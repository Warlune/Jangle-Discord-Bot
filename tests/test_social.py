from __future__ import annotations

import random

from discord_ai.social import (
    PARTY_AMBIENT_COOLDOWN_SECONDS,
    GameState,
    answer_matches,
    award_category_is_safe,
    choose_game_question,
    extract_nomination_target,
    match_option,
    parse_poll_spec,
    parse_social_command,
    should_accept_party_ambient,
    twenty_question_guess_matches,
)


def test_party_game_commands_cover_each_mode() -> None:
    wow = parse_social_command("start WoW trivia")
    general = parse_social_command("play general trivia game")
    riddle = parse_social_command("start a riddle game")
    would = parse_social_command("begin would you rather")
    twenty = parse_social_command("start 20 questions")

    assert wow is not None and (wow.activity, wow.action, wow.mode, wow.argument) == (
        "game",
        "start",
        "trivia",
        "wow",
    )
    assert general is not None and general.argument == "general"
    assert riddle is not None and riddle.mode == "riddle"
    assert would is not None and would.mode == "would"
    assert twenty is not None and twenty.mode == "twenty"
    assert parse_social_command("game score").action == "status"  # type: ignore[union-attr]
    assert parse_social_command("give us a hint").action == "hint"  # type: ignore[union-attr]
    assert parse_social_command("stop game").action == "stop"  # type: ignore[union-attr]
    assert parse_social_command("social help").activity == "social"  # type: ignore[union-attr]


def test_poll_dnd_award_and_party_commands_parse_naturally() -> None:
    poll = parse_social_command("start a poll raid or keys or battlegrounds")
    detailed_poll = parse_social_command(
        "start a vote What should we run tonight: raid, keys, or battlegrounds"
    )
    dnd = parse_social_command("start DND campaign about a cursed raid portal")
    fresh_dnd = parse_social_command("start a new D and D campaign about haunted mines")
    award = parse_social_command("start an award for most likely to stand in fire")
    party = parse_social_command("enable party mode for 20 minutes")

    assert poll is not None and poll.options == ("raid", "keys", "battlegrounds")
    assert detailed_poll is not None
    assert detailed_poll.argument == "What should we run tonight"
    assert detailed_poll.options == ("raid", "keys", "battlegrounds")
    assert dnd is not None and (dnd.activity, dnd.action, dnd.argument) == (
        "dnd",
        "start",
        "a cursed raid portal",
    )
    assert fresh_dnd is not None and fresh_dnd.action == "new"
    assert parse_social_command("resume DND").action == "resume"  # type: ignore[union-attr]
    assert parse_social_command("join DND").action == "join"  # type: ignore[union-attr]
    assert parse_social_command("my stats").action == "stats"  # type: ignore[union-attr]
    assert parse_social_command("party stats").action == "party_stats"  # type: ignore[union-attr]
    assert parse_social_command("DND journal").action == "journal"  # type: ignore[union-attr]
    assert parse_social_command("start story mode about a portal") is None
    assert award is not None and award.argument == "most likely to stand in fire"
    assert party is not None and party.duration_minutes == 20
    assert parse_social_command("party mode off").action == "stop"  # type: ignore[union-attr]
    assert parse_social_command("award results").action == "status"  # type: ignore[union-attr]


def test_poll_spec_requires_distinct_spoken_options() -> None:
    assert parse_poll_spec("raid or keys") == (
        "What should we choose?",
        ("raid", "keys"),
    )
    assert parse_poll_spec("Favorite role: tank, healer, or damage") == (
        "Favorite role",
        ("tank", "healer", "damage"),
    )


def test_answer_and_option_matching_tolerate_voice_variation() -> None:
    assert answer_matches("I think the answer is Frostmourne", ("frostmourne",)) is True
    assert answer_matches("Jupiter", ("jupiter",)) is True
    assert match_option("I vote for battlegrounds", ("raid", "keys", "battlegrounds")) == 2
    assert match_option("option two", ("raid", "keys")) == 1
    assert match_option("something unrelated", ("raid", "keys")) is None


def test_twenty_questions_only_accepts_an_actual_guess() -> None:
    assert twenty_question_guess_matches("piano", ("piano",)) is True
    assert twenty_question_guess_matches("is it a piano", ("piano",)) is True
    assert twenty_question_guess_matches("is it bigger than a piano", ("piano",)) is False


def test_game_questions_do_not_repeat_within_a_short_round() -> None:
    state = GameState("trivia", "general", 1, "Host")
    rng = random.Random(7)

    prompts = {choose_game_question(state, rng).prompt for _ in range(5)}

    assert len(prompts) == 5


def test_award_targets_and_sensitive_categories_are_bounded() -> None:
    assert extract_nomination_target("I nominate Raid Dad") == "Raid Dad"
    assert extract_nomination_target("ordinary conversation") is None
    assert award_category_is_safe("most likely to stand in fire") is True
    assert award_category_is_safe("worst medical diagnosis") is False


def test_party_ambient_requires_delay_cooldown_and_a_successful_roll() -> None:
    base = dict(
        transcript="that raid pull was an absolute disaster",
        duration_seconds=1.2,
        enabled_at=100.0,
        deadline=2000.0,
        last_reaction_at=0.0,
    )

    assert should_accept_party_ambient(now=120.0, roll=0.1, **base) is True
    assert (
        should_accept_party_ambient(
            now=120.0,
            roll=0.9,
            **base,
        )
        is False
    )
    assert (
        should_accept_party_ambient(
            now=110.0,
            roll=0.1,
            **base,
        )
        is False
    )
    cooldown = dict(base)
    cooldown["last_reaction_at"] = 120.0 - PARTY_AMBIENT_COOLDOWN_SECONDS + 1
    assert should_accept_party_ambient(now=120.0, roll=0.1, **cooldown) is False
    question = dict(base)
    question["transcript"] = "what happened during that raid pull"
    assert should_accept_party_ambient(now=120.0, roll=0.1, **question) is False
