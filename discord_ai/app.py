from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from .config import ConfigurationError, Settings
from .conversation_log import TestConversationLog
from .social import SOCIAL_HELP_TEXT
from .user_notes import (
    UserNoteCommand,
    UserNoteStore,
    execute_user_note_command,
    parse_user_note_command,
)
from .voice import VoiceManager
from .warlune import AnswerService, create_model_gateway


LOGGER = logging.getLogger(__name__)
BOT_RESTART_DELAY_SECONDS = 10.0


def extract_text_call_request(text: str, call_words: tuple[str, ...]) -> str | None:
    alternatives = "|".join(
        re.escape(word) for word in sorted(call_words, key=len, reverse=True)
    )
    match = re.match(
        rf"^\s*(?:(?:hey|hi|hello|yo|ok|okay)[\s,.:;!?-]+)?(?:{alternatives})(?!\w)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    request = text[match.end() :].strip(" \t,.:;!?-")
    return request or "Say a brief, friendly hello."


def text_session_key(guild_id: int | None, channel_id: int | None, user_id: int) -> str:
    return f"text:{guild_id}:{channel_id}:{user_id}"


def split_discord_message(text: str, limit: int = 1900) -> list[str]:
    remaining = text.strip()
    if not remaining:
        return ["I did not get an answer back."]
    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = max(
            remaining.rfind("\n\n", 0, limit),
            remaining.rfind("\n", 0, limit),
            remaining.rfind(". ", 0, limit),
            remaining.rfind(" ", 0, limit),
        )
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks[:6]


class DiscordAIBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents(
            guilds=True,
            voice_states=True,
            messages=True,
            message_content=True,
        )
        super().__init__(
            command_prefix=commands.when_mentioned_or(settings.bot_prefix),
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.settings = settings
        self.conversations = TestConversationLog(
            settings.test_mode,
            settings.conversation_log_path,
            max_bytes=settings.conversation_log_max_bytes,
        )
        self.gateway = create_model_gateway(settings)
        self.user_notes = UserNoteStore(settings.user_notes_state_path)
        self.answers = AnswerService(settings, self.gateway, self.conversations)
        self.voices = VoiceManager(settings, self.answers, self.user_notes)

    async def setup_hook(self) -> None:
        LOGGER.info("Discord login accepted; syncing application commands")
        if self.settings.dev_guild_id is not None:
            guild = discord.Object(id=self.settings.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        LOGGER.info("Discord application commands synced")
        asyncio.create_task(self._warm_model())
        asyncio.create_task(self.voices.warm())

    async def _warm_model(self) -> None:
        try:
            model = await self.answers.warm()
            LOGGER.info(
                "%s model is ready: %s",
                self.gateway.provider_name,
                model or self.gateway.model,
            )
        except Exception:
            LOGGER.warning("Model warmup failed; requests will retry when used", exc_info=True)

    async def close(self) -> None:
        await self.voices.close()
        await super().close()


def create_bot(settings: Settings) -> DiscordAIBot:
    bot = DiscordAIBot(settings)

    def text_log_context(
        user: discord.abc.User,
        guild_id: int | None,
        channel_id: int | None,
        entrypoint: str,
    ) -> dict[str, Any]:
        personality = bot.voices.personality_choice(guild_id)
        return {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "user_id": user.id,
            "user_name": str(getattr(user, "display_name", user.name))[:100],
            "entrypoint": entrypoint,
            "personality_key": personality.key,
            "personality_name": personality.name,
        }

    async def answer_text_request(
        session_key: str,
        prompt: str,
        *,
        guild_id: int,
        user_id: int,
        force_search: bool = False,
        intelligence_level: str = "Instant",
        log_context: dict[str, Any] | None = None,
    ) -> str:
        note_command = parse_user_note_command(prompt)
        if note_command is not None:
            response = await asyncio.to_thread(
                execute_user_note_command,
                bot.user_notes,
                guild_id,
                user_id,
                note_command,
            )
            bot.answers.record_event(
                "user_note_command",
                guild_id=guild_id,
                user_id=user_id,
                action=note_command.action,
                note_count=bot.user_notes.count(guild_id, user_id),
                entrypoint=(log_context or {}).get("entrypoint", "text"),
            )
            return response

        notes_context = await asyncio.to_thread(
            bot.user_notes.prompt_context,
            guild_id,
            user_id,
        )
        return await bot.answers.answer(
            session_key,
            prompt,
            force_search=force_search,
            intelligence_level=intelligence_level,
            runtime_context=notes_context,
            personality_prompt=bot.voices.personality_prompt(guild_id),
            log_context=log_context,
        )

    async def interaction_allowed(interaction: discord.Interaction[Any]) -> bool:
        channel_name = str(getattr(interaction.channel, "name", "") or "")
        if settings.guild_is_allowed(interaction.guild_id) and settings.text_channel_is_allowed(
            interaction.channel_id, channel_name
        ):
            return True
        message = "This bot is currently restricted to the `TEST text` channel."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return False

    async def voice_control_allowed(interaction: discord.Interaction[Any]) -> bool:
        if not settings.admin_role_ids:
            return True
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        allowed = member is not None and (
            member.guild_permissions.manage_guild
            or any(role.id in settings.admin_role_ids for role in member.roles)
        )
        if allowed:
            return True
        message = "You need an allowlisted bot-control role to do that."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return False

    async def send_interaction(
        interaction: discord.Interaction[Any], text: str, *, ephemeral: bool = False
    ) -> None:
        chunks = split_discord_message(text)
        for chunk in chunks:
            await interaction.followup.send(
                chunk,
                ephemeral=ephemeral,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def answer_interaction(
        interaction: discord.Interaction[Any],
        prompt: str,
        *,
        force_search: bool = False,
        intelligence_level: str = "Instant",
    ) -> None:
        if not await interaction_allowed(interaction):
            return
        await interaction.response.defer(thinking=True)
        try:
            session_key = text_session_key(
                interaction.guild_id,
                interaction.channel_id,
                interaction.user.id,
            )
            answer = await answer_text_request(
                session_key,
                prompt,
                guild_id=int(interaction.guild_id or 0),
                user_id=interaction.user.id,
                force_search=force_search,
                intelligence_level=intelligence_level,
                log_context=text_log_context(
                    interaction.user,
                    interaction.guild_id,
                    interaction.channel_id,
                    "slash_command",
                ),
            )
            await send_interaction(interaction, answer)
        except Exception:
            LOGGER.exception("Discord text request failed")
            await interaction.followup.send(
                "I could not complete that request. Check the bot console and local model server.",
                ephemeral=True,
            )

    @bot.event
    async def on_ready() -> None:
        if bot.user is not None and bot.user.name != settings.bot_display_name:
            try:
                await bot.user.edit(username=settings.bot_display_name)
                LOGGER.info("Discord bot username changed to %s", settings.bot_display_name)
            except discord.HTTPException:
                LOGGER.warning("Could not change the Discord bot username", exc_info=True)
        for guild in bot.guilds:
            member = guild.me
            if member is None or member.display_name == settings.bot_display_name:
                continue
            try:
                await member.edit(nick=settings.bot_display_name, reason="Jangle bot identity")
            except discord.HTTPException:
                LOGGER.warning("Could not set bot nickname in guild %s", guild.id, exc_info=True)
        LOGGER.info("Discord AI ready as %s in %d guild(s)", bot.user, len(bot.guilds))
        for label, channel_ids in (
            ("text", settings.text_channel_ids),
            ("voice", settings.voice_channel_ids),
        ):
            for channel_id in sorted(channel_ids):
                channel = bot.get_channel(channel_id)
                if channel is None:
                    LOGGER.warning("Configured %s channel is not visible to the bot: %s", label, channel_id)
                    continue
                LOGGER.info(
                    "Configured %s channel resolved: %s / %s (%s)",
                    label,
                    getattr(getattr(channel, "guild", None), "name", "unknown guild"),
                    getattr(channel, "name", "unknown channel"),
                    channel_id,
                )

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return
        if not settings.guild_is_allowed(message.guild.id):
            return
        channel_name = str(getattr(message.channel, "name", "") or "")
        if not settings.text_channel_is_allowed(message.channel.id, channel_name):
            return
        if message.content.startswith(settings.bot_prefix):
            await bot.process_commands(message)
            return

        mentioned = bot.user is not None and bot.user in message.mentions
        if mentioned and bot.user is not None:
            prompt = message.content
            prompt = re.sub(rf"<@!?{bot.user.id}>", "", prompt).strip()
            text_trigger = "mention"
        elif settings.text_require_call_word:
            prompt = extract_text_call_request(message.content, settings.text_call_words)
            if prompt is None:
                return
            text_trigger = "call_word"
        else:
            prompt = message.content.strip()
            text_trigger = "automatic"
        if not prompt:
            return
        try:
            async with message.channel.typing():
                session_key = text_session_key(
                    message.guild.id,
                    message.channel.id,
                    message.author.id,
                )
                context = text_log_context(
                    message.author,
                    message.guild.id,
                    message.channel.id,
                    "automatic_message",
                )
                context["trigger"] = text_trigger
                answer = await answer_text_request(
                    session_key,
                    prompt,
                    guild_id=message.guild.id,
                    user_id=message.author.id,
                    log_context=context,
                )
            for chunk in split_discord_message(answer):
                await message.channel.send(
                    chunk,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        except Exception:
            LOGGER.exception("Automatic Discord text request failed")
            await message.channel.send(
                "I could not complete that request. Check the bot console and local model server."
            )

    @bot.command(name="ask")
    async def ask_prefix(ctx: commands.Context[DiscordAIBot], *, question: str) -> None:
        if ctx.guild is None or not settings.guild_is_allowed(ctx.guild.id):
            return
        if not settings.text_channel_is_allowed(ctx.channel.id, str(getattr(ctx.channel, "name", "") or "")):
            return
        async with ctx.typing():
            key = text_session_key(ctx.guild.id, ctx.channel.id, ctx.author.id)
            answer = await answer_text_request(
                key,
                question,
                guild_id=ctx.guild.id,
                user_id=ctx.author.id,
                log_context=text_log_context(ctx.author, ctx.guild.id, ctx.channel.id, "prefix_ask"),
            )
        for chunk in split_discord_message(answer):
            await ctx.send(chunk, allowed_mentions=discord.AllowedMentions.none())

    @bot.command(name="search")
    async def search_prefix(ctx: commands.Context[DiscordAIBot], *, query: str) -> None:
        if ctx.guild is None or not settings.guild_is_allowed(ctx.guild.id):
            return
        if not settings.text_channel_is_allowed(ctx.channel.id, str(getattr(ctx.channel, "name", "") or "")):
            return
        async with ctx.typing():
            key = text_session_key(ctx.guild.id, ctx.channel.id, ctx.author.id)
            answer = await answer_text_request(
                key,
                query,
                guild_id=ctx.guild.id,
                user_id=ctx.author.id,
                force_search=True,
                log_context=text_log_context(
                    ctx.author, ctx.guild.id, ctx.channel.id, "prefix_search"
                ),
            )
        for chunk in split_discord_message(answer):
            await ctx.send(chunk, allowed_mentions=discord.AllowedMentions.none())

    @bot.command(name="medium")
    async def medium_prefix(ctx: commands.Context[DiscordAIBot], *, question: str) -> None:
        if ctx.guild is None or not settings.guild_is_allowed(ctx.guild.id):
            return
        if not settings.text_channel_is_allowed(ctx.channel.id, str(getattr(ctx.channel, "name", "") or "")):
            return
        async with ctx.typing():
            key = text_session_key(ctx.guild.id, ctx.channel.id, ctx.author.id)
            answer = await answer_text_request(
                key,
                question,
                guild_id=ctx.guild.id,
                user_id=ctx.author.id,
                intelligence_level="Medium",
                log_context=text_log_context(
                    ctx.author, ctx.guild.id, ctx.channel.id, "prefix_medium"
                ),
            )
        for chunk in split_discord_message(answer):
            await ctx.send(chunk, allowed_mentions=discord.AllowedMentions.none())

    @bot.command(name="pro")
    async def pro_prefix(ctx: commands.Context[DiscordAIBot], *, question: str) -> None:
        if ctx.guild is None or not settings.guild_is_allowed(ctx.guild.id):
            return
        if not settings.text_channel_is_allowed(ctx.channel.id, str(getattr(ctx.channel, "name", "") or "")):
            return
        async with ctx.typing():
            key = text_session_key(ctx.guild.id, ctx.channel.id, ctx.author.id)
            answer = await answer_text_request(
                key,
                question,
                guild_id=ctx.guild.id,
                user_id=ctx.author.id,
                intelligence_level="Pro",
                log_context=text_log_context(ctx.author, ctx.guild.id, ctx.channel.id, "prefix_pro"),
            )
        for chunk in split_discord_message(answer):
            await ctx.send(chunk, allowed_mentions=discord.AllowedMentions.none())

    @bot.tree.command(name="ask", description="Ask the configured local model a question")
    @app_commands.describe(question="What you want to ask")
    async def ask_slash(interaction: discord.Interaction[Any], question: str) -> None:
        await answer_interaction(interaction, question)

    @bot.tree.command(name="search", description="Search configured public web context and answer")
    @app_commands.describe(query="What to search for")
    async def search_slash(interaction: discord.Interaction[Any], query: str) -> None:
        await answer_interaction(interaction, query, force_search=True)

    @bot.tree.command(name="medium", description="Ask with the balanced reasoning level")
    @app_commands.describe(question="What you want the model to reason about")
    async def medium_slash(interaction: discord.Interaction[Any], question: str) -> None:
        await answer_interaction(interaction, question, intelligence_level="Medium")

    @bot.tree.command(name="pro", description="Ask with the deepest practical reasoning level")
    @app_commands.describe(question="The complex question or task")
    async def pro_slash(interaction: discord.Interaction[Any], question: str) -> None:
        await answer_interaction(interaction, question, intelligence_level="Pro")

    async def run_note_interaction(
        interaction: discord.Interaction[Any],
        command: UserNoteCommand,
    ) -> None:
        if not await interaction_allowed(interaction):
            return
        if interaction.guild_id is None:
            await interaction.response.send_message(
                "Use this inside a Discord server.",
                ephemeral=True,
            )
            return
        response = await asyncio.to_thread(
            execute_user_note_command,
            bot.user_notes,
            interaction.guild_id,
            interaction.user.id,
            command,
        )
        bot.answers.record_event(
            "user_note_command",
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
            user_id=interaction.user.id,
            action=command.action,
            note_count=bot.user_notes.count(
                interaction.guild_id,
                interaction.user.id,
            ),
            entrypoint="slash_command",
        )
        await interaction.response.send_message(
            response,
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @bot.tree.command(name="remember", description="Save one small note in your Jangle notepad")
    @app_commands.describe(note="A harmless preference, hobby, game fact, or running joke")
    async def remember_slash(interaction: discord.Interaction[Any], note: str) -> None:
        await run_note_interaction(interaction, UserNoteCommand("add", note=note))

    @bot.tree.command(name="memories", description="Privately show your Jangle notepad")
    async def memories_slash(interaction: discord.Interaction[Any]) -> None:
        await run_note_interaction(interaction, UserNoteCommand("list"))

    @bot.tree.command(name="forget_memory", description="Remove one Jangle note or clear them all")
    @app_commands.describe(target="A note number, a matching phrase, or 'all'")
    async def forget_memory_slash(
        interaction: discord.Interaction[Any],
        target: str,
    ) -> None:
        clean_target = target.strip()
        if clean_target.casefold() in {"all", "everything"}:
            command = UserNoteCommand("clear")
        elif clean_target.isdigit():
            command = UserNoteCommand("remove_index", index=int(clean_target))
        else:
            command = UserNoteCommand("remove_matching", note=clean_target)
        await run_note_interaction(interaction, command)

    @bot.tree.command(name="activities", description="Show Jangle's voice games and social activities")
    async def activities_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        await interaction.response.send_message(
            SOCIAL_HELP_TEXT,
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @bot.tree.command(name="join", description="Join your voice channel and listen for the wake word")
    async def join_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        if not await voice_control_allowed(interaction):
            return
        if interaction.guild is None or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("Use this inside a Discord server.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            status = await bot.voices.join(interaction.user, interaction.channel)
            await interaction.followup.send(status)
        except ValueError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception:
            LOGGER.exception("Could not join Discord voice")
            await interaction.followup.send("I could not join that voice channel.", ephemeral=True)

    @bot.tree.command(name="leave", description="Disconnect the voice AI from this server")
    async def leave_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        if not await voice_control_allowed(interaction):
            return
        if interaction.guild is None:
            await interaction.response.send_message("Use this inside a Discord server.", ephemeral=True)
            return
        disconnected = await bot.voices.leave(interaction.guild)
        await interaction.response.send_message(
            "Disconnected." if disconnected else "I am not in a voice channel.",
            ephemeral=True,
        )

    @bot.tree.command(name="voice_debug", description="Temporarily echo voice transcripts for tuning")
    @app_commands.describe(enabled="Show or hide recognized prompts and answers in TEST text")
    async def voice_debug_slash(
        interaction: discord.Interaction[Any], enabled: bool
    ) -> None:
        if not await interaction_allowed(interaction):
            return
        if not await voice_control_allowed(interaction):
            return
        if interaction.guild_id is None:
            await interaction.response.send_message("Use this inside a Discord server.", ephemeral=True)
            return
        bot.voices.set_debug(interaction.guild_id, enabled)
        await interaction.response.send_message(
            (
                "Voice debug enabled. Recognized requests and answers will appear in this text channel; "
                "audio is still not saved or sent to model memory."
                if enabled
                else "Voice debug disabled."
            ),
            ephemeral=True,
        )

    @bot.tree.command(name="reset", description="Forget your temporary conversation context here")
    async def reset_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        key = text_session_key(
            interaction.guild_id,
            interaction.channel_id,
            interaction.user.id,
        )
        bot.answers.sessions.reset(key)
        await interaction.response.send_message(
            "Temporary context cleared. Your Jangle notepad was not changed.",
            ephemeral=True,
        )

    @bot.tree.command(name="privacy", description="Show what this Discord adapter can access and retain")
    async def privacy_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        await interaction.response.send_message(
            "Discord guest mode sends only the current request, short-lived history, relevant explicit "
            "Jangle notes, and configured public search context to the selected model endpoint. It does "
            "not load Warlune profile data, durable memory, saved chats, files, projects, or credentials. "
            "Short conversation history exists only in RAM until reset or "
            "restart. Each user may explicitly keep up to five harmless 100-character notes in a "
            "plugin-owned local notepad keyed by Discord server and user ID; users can privately view "
            "or delete their own notes with slash commands. "
            + (
                "Test mode saves bot-directed text, accepted voice transcripts, answers, and timing to a "
                "rotating local Jangle log. Ignored room conversation and raw audio are not saved. "
                if settings.test_mode
                else "Voice transcripts and answers are not saved by Jangle. "
            )
            + "Pocket TTS stays local. Edge TTS receives only generated answer text when enabled.",
            ephemeral=True,
        )

    @bot.tree.command(name="status", description="Show the local model and voice connection status")
    async def status_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        guild_status = bot.voices.status(interaction.guild_id or 0)
        model = bot.gateway.model or "auto-detecting"
        note_count = (
            bot.user_notes.count(interaction.guild_id, interaction.user.id)
            if interaction.guild_id is not None
            else 0
        )
        await interaction.response.send_message(
            f"Provider: `{bot.gateway.provider_name}`\nModel: `{model}`\nVoice: {guild_status}\n"
            f"Your Jangle notes: {note_count}/5\n"
            f"Test logging: {'enabled' if settings.test_mode else 'disabled'}\n"
            f"Active model requests: {bot.answers.active_requests}",
            ephemeral=True,
        )

    @bot.tree.command(name="voice_members", description="List people in your current voice channel")
    async def voice_members_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        channel = member.voice.channel if member is not None and member.voice is not None else None
        if channel is None or not settings.voice_channel_is_allowed(channel.id, channel.name):
            await interaction.response.send_message("Join `TEST voice` first.", ephemeral=True)
            return
        names = [item.display_name for item in channel.members if not item.bot]
        text = ", ".join(names) if names else "No human members are present."
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @bot.tree.command(name="channel_info", description="Show read-only information about this channel")
    async def channel_info_slash(interaction: discord.Interaction[Any]) -> None:
        if not await interaction_allowed(interaction):
            return
        channel = interaction.channel
        name = getattr(channel, "name", "unknown")
        topic = getattr(channel, "topic", None) or "none"
        slowmode = getattr(channel, "slowmode_delay", 0)
        await interaction.response.send_message(
            f"Name: `{name}`\nTopic: {str(topic)[:1200]}\nSlowmode: {slowmode} seconds",
            ephemeral=True,
        )

    return bot


def run_bot_with_restart(
    settings: Settings,
    *,
    restart_delay_seconds: float = BOT_RESTART_DELAY_SECONDS,
) -> None:
    while True:
        try:
            bot = create_bot(settings)
            bot.run(settings.bot_token, log_handler=None)
            return
        except Exception:
            LOGGER.exception(
                "Jangle stopped unexpectedly; retrying Discord in %.0f seconds",
                restart_delay_seconds,
            )
            time.sleep(restart_delay_seconds)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.ERROR)
    try:
        settings = Settings.from_env()
        LOGGER.info("Starting Jangle Discord AI with configured text/voice channel locks")
        if settings.test_mode:
            LOGGER.info(
                "Jangle test conversation logging is enabled at %s",
                settings.conversation_log_path,
            )
        run_bot_with_restart(settings)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
