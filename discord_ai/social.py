from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


SOCIAL_ACTIVITY_SECONDS = 20 * 60
GAME_ANSWER_WINDOW_SECONDS = 15.0
TRIVIA_CORRECT_SETTLE_SECONDS = 0.8
PARTY_MODE_DEFAULT_MINUTES = 15
PARTY_MODE_MAX_MINUTES = 30
PARTY_AMBIENT_COOLDOWN_SECONDS = 75.0
PARTY_AMBIENT_MIN_DELAY_SECONDS = 15.0
PARTY_AMBIENT_CHANCE = 0.35

SOCIAL_HELP_TEXT = """**Jangle voice activities**
Games: `start WoW trivia`, `start general trivia`, `start riddles`, `start Would You Rather`, `start Twenty Questions`, `hint`, `game score`, `next question`, `stop game`
Polls: `start poll raid or keys or battlegrounds`, `poll results`, `end poll`, `cancel poll`
Stories: `start story mode about <theme>`, `story status`, `end story`, `cancel story`
Awards: `start an award for <category>`, then say `I nominate <Discord name>`; use `award results`, `end nominations`, or `cancel award`
Party Mode: administrator `enable party mode for 15 minutes`; anyone `party mode off`

Say `Hey Jangle` before setup and control commands. Participant answers, votes, story turns, and nominations are wake-free while their activity is active."""


@dataclass(frozen=True)
class SocialCommand:
    activity: str
    action: str
    mode: str = ""
    argument: str = ""
    options: tuple[str, ...] = ()
    duration_minutes: int = 0


@dataclass(frozen=True)
class QuizQuestion:
    prompt: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class WouldQuestion:
    first: str
    second: str


@dataclass(frozen=True)
class TwentyQuestionSecret:
    answer: str
    aliases: tuple[str, ...]
    category: str


@dataclass
class GameState:
    kind: str
    category: str
    host_user_id: int
    host_name: str
    started_at: float = field(default_factory=time.monotonic)
    expires_at: float = field(
        default_factory=lambda: time.monotonic() + SOCIAL_ACTIVITY_SECONDS
    )
    round_number: int = 0
    rounds_total: int = 5
    used_questions: set[int] = field(default_factory=set)
    current_question: QuizQuestion | WouldQuestion | None = None
    scores: dict[int, int] = field(default_factory=dict)
    player_names: dict[int, str] = field(default_factory=dict)
    votes: dict[int, int] = field(default_factory=dict)
    secret: TwentyQuestionSecret | None = None
    question_count: int = 0
    answer_attempt: int = 1
    accepting_answers: bool = False
    eligible_user_ids: set[int] = field(default_factory=set)
    attempted_user_ids: set[int] = field(default_factory=set)
    correct_user_ids: set[int] = field(default_factory=set)
    submission_started_at: dict[int, float] = field(default_factory=dict)
    window_token: int = 0
    hint_used: bool = False


@dataclass
class PollState:
    question: str
    options: tuple[str, ...]
    host_user_id: int
    host_name: str
    started_at: float = field(default_factory=time.monotonic)
    expires_at: float = field(
        default_factory=lambda: time.monotonic() + SOCIAL_ACTIVITY_SECONDS
    )
    votes: dict[int, int] = field(default_factory=dict)
    voter_names: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StoryParticipant:
    user_id: int
    name: str


@dataclass
class StoryState:
    theme: str
    host_user_id: int
    host_name: str
    participants: tuple[StoryParticipant, ...]
    started_at: float = field(default_factory=time.monotonic)
    expires_at: float = field(
        default_factory=lambda: time.monotonic() + SOCIAL_ACTIVITY_SECONDS
    )
    turn_index: int = 0
    turns_completed: int = 0
    max_turns: int = 6


@dataclass
class AwardState:
    category: str
    host_user_id: int
    host_name: str
    started_at: float = field(default_factory=time.monotonic)
    expires_at: float = field(
        default_factory=lambda: time.monotonic() + SOCIAL_ACTIVITY_SECONDS
    )
    nominations: dict[int, int] = field(default_factory=dict)
    voter_names: dict[int, str] = field(default_factory=dict)
    nominee_names: dict[int, str] = field(default_factory=dict)


WOW_TRIVIA: tuple[QuizQuestion, ...] = (
    QuizQuestion("What is the name of the Lich King's runeblade?", ("frostmourne",)),
    QuizQuestion("What was the orcs' original homeworld called?", ("draenor",)),
    QuizQuestion("Which dragon aspect is known as the Life-Binder?", ("alexstrasza",)),
    QuizQuestion("Who famously says, 'You are not prepared'?", ("illidan", "illidan stormrage")),
    QuizQuestion("What is the blood elf capital city?", ("silvermoon", "silvermoon city")),
    QuizQuestion("Which resource do rogues spend on most abilities?", ("energy",)),
    QuizQuestion("Which city was named after Orgrim Doomhammer?", ("orgrimmar",)),
    QuizQuestion("What is the draenei home city on Azuremyst Isle?", ("the exodar", "exodar")),
    QuizQuestion("Which Old God is imprisoned beneath Ahn'Qiraj?", ("cthun", "c thun")),
    QuizQuestion("What class uses soul shards?", ("warlock", "warlocks")),
)

GENERAL_TRIVIA: tuple[QuizQuestion, ...] = (
    QuizQuestion("What is the largest planet in our solar system?", ("jupiter",)),
    QuizQuestion("Which chemical element uses the symbol Au?", ("gold",)),
    QuizQuestion("What is the capital of Japan?", ("tokyo",)),
    QuizQuestion("Which ocean is the largest?", ("pacific", "pacific ocean")),
    QuizQuestion("How many pieces does each player begin with in chess?", ("16", "sixteen")),
    QuizQuestion("Who wrote The Hobbit?", ("tolkien", "jrr tolkien", "j r r tolkien")),
    QuizQuestion("What is the fastest land animal?", ("cheetah",)),
    QuizQuestion("How many sides does a triangle have?", ("3", "three")),
    QuizQuestion("Which mammal is capable of true sustained flight?", ("bat", "bats")),
    QuizQuestion("What instrument has 88 keys on a standard modern version?", ("piano",)),
)

RIDDLES: tuple[QuizQuestion, ...] = (
    QuizQuestion("What has keys but cannot open locks?", ("piano", "a piano")),
    QuizQuestion("What gets wetter the more it dries?", ("towel", "a towel")),
    QuizQuestion("What has hands and a face but no arms or legs?", ("clock", "a clock")),
    QuizQuestion("What can travel around the world while staying in one corner?", ("stamp", "a stamp")),
    QuizQuestion("What has a neck but no head?", ("bottle", "a bottle")),
    QuizQuestion("What has many teeth but cannot bite?", ("comb", "a comb")),
    QuizQuestion("What belongs to you but other people use more than you do?", ("name", "your name", "my name")),
    QuizQuestion("What can fill a room but takes up no space?", ("light",)),
    QuizQuestion("What has one eye but cannot see?", ("needle", "a needle")),
    QuizQuestion("The more you take, the more you leave behind. What are they?", ("footsteps", "steps")),
)

WOULD_QUESTIONS: tuple[WouldQuestion, ...] = (
    WouldQuestion("tank a raid with no addons", "heal a raid with everyone standing in fire"),
    WouldQuestion("only communicate in raid warnings", "only communicate through emotes"),
    WouldQuestion("have perfect gear but terrible luck", "bad gear but impossible luck"),
    WouldQuestion("fight one giant murloc", "fight one hundred tiny murlocs"),
    WouldQuestion("always arrive ten minutes early", "always arrive one pull late"),
    WouldQuestion("lose access to music", "lose access to video games"),
    WouldQuestion("know every language", "play every instrument"),
    WouldQuestion("explore deep space", "explore the deepest ocean"),
    WouldQuestion("have unlimited free travel", "have unlimited free food"),
    WouldQuestion("restart your favorite game fresh", "keep your current collection forever"),
)

TWENTY_QUESTION_SECRETS: tuple[TwentyQuestionSecret, ...] = (
    TwentyQuestionSecret("a piano", ("piano",), "an object"),
    TwentyQuestionSecret("a volcano", ("volcano",), "a natural place"),
    TwentyQuestionSecret("a pineapple", ("pineapple",), "food"),
    TwentyQuestionSecret("a lighthouse", ("lighthouse",), "a structure"),
    TwentyQuestionSecret("a snowman", ("snowman",), "a familiar figure"),
    TwentyQuestionSecret("a dragon", ("dragon",), "a creature"),
    TwentyQuestionSecret("a microwave", ("microwave", "microwave oven"), "an object"),
    TwentyQuestionSecret("the moon", ("moon", "the moon"), "a place"),
    TwentyQuestionSecret("a cactus", ("cactus",), "a living thing"),
    TwentyQuestionSecret("an umbrella", ("umbrella",), "an object"),
)


def _clean_request(value: str) -> str:
    return " ".join(value.strip(" ,.!?").split())


def normalize_social_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", value).casefold().split())


def parse_social_command(request: str) -> SocialCommand | None:
    clean = _clean_request(request)
    normalized = normalize_social_text(clean)
    if not clean:
        return None

    if normalized in {
        "social help",
        "activity help",
        "activities",
        "list activities",
        "show activities",
        "list party games",
        "show party games",
        "what games can we play",
    }:
        return SocialCommand("social", "help")

    party_start = re.fullmatch(
        r"(?:(?:enable|start|activate|turn\s+on)\s+party\s+mode|party\s+mode\s+on)"
        r"(?:\s+for\s+(?P<minutes>\d+)\s+minutes?)?",
        clean,
        flags=re.IGNORECASE,
    )
    if party_start is not None:
        minutes = int(party_start.group("minutes") or PARTY_MODE_DEFAULT_MINUTES)
        return SocialCommand("party", "start", duration_minutes=minutes)
    if normalized in {
        "party mode off",
        "disable party mode",
        "stop party mode",
        "end party mode",
        "turn off party mode",
    }:
        return SocialCommand("party", "stop")
    if normalized in {"party mode", "party mode status", "is party mode on"}:
        return SocialCommand("party", "status")

    game_start = re.fullmatch(
        r"(?:start|begin|play)\s+(?:a\s+|the\s+)?"
        r"(?P<game>wow\s+trivia|general\s+trivia|trivia|riddles?|would\s+you\s+rather|"
        r"twenty\s+questions|20\s+questions)(?:\s+game)?",
        clean,
        flags=re.IGNORECASE,
    )
    if game_start is not None:
        game = normalize_social_text(game_start.group("game"))
        if "trivia" in game:
            category = "wow" if game.startswith("wow") else "general" if game.startswith("general") else "mixed"
            return SocialCommand("game", "start", mode="trivia", argument=category)
        if game.startswith("riddle"):
            return SocialCommand("game", "start", mode="riddle")
        if "would you rather" in game:
            return SocialCommand("game", "start", mode="would")
        return SocialCommand("game", "start", mode="twenty")
    if normalized in {
        "stop game",
        "end game",
        "cancel game",
        "stop the game",
        "end the game",
        "game mode off",
        "exit game mode",
    }:
        return SocialCommand("game", "stop")
    if normalized in {"game score", "game scores", "score", "scoreboard", "game status"}:
        return SocialCommand("game", "status")
    if normalized in {
        "hint",
        "game hint",
        "give me a hint",
        "give us a hint",
        "can we have a hint",
        "can i have a hint",
    }:
        return SocialCommand("game", "hint")
    if normalized in {"next question", "next round", "skip question", "skip round"}:
        return SocialCommand("game", "next")

    poll_start = re.fullmatch(
        r"(?:start|begin|create)\s+(?:a\s+|the\s+)?(?:poll|vote)(?:\s+(?P<body>.+))?",
        clean,
        flags=re.IGNORECASE,
    )
    if poll_start is not None:
        body = (poll_start.group("body") or "").strip()
        question, options = parse_poll_spec(body)
        return SocialCommand("poll", "start", argument=question, options=options)
    if normalized in {"poll results", "vote results", "show poll", "poll status"}:
        return SocialCommand("poll", "status")
    if normalized in {"end poll", "finish poll", "close poll", "end vote", "finish vote"}:
        return SocialCommand("poll", "finish")
    if normalized in {"cancel poll", "stop poll", "cancel vote", "stop vote"}:
        return SocialCommand("poll", "stop")

    story_start = re.fullmatch(
        r"(?:start|begin)\s+(?:a\s+|the\s+)?(?:story|scenario)(?:\s+mode)?"
        r"(?:\s+(?:about|with|called|set\s+in)\s+)?(?P<theme>.*)",
        clean,
        flags=re.IGNORECASE,
    )
    if story_start is not None:
        theme = story_start.group("theme").strip() or "a chaotic fantasy tavern adventure"
        return SocialCommand("story", "start", argument=theme[:120])
    if normalized in {"stop story", "end story", "finish story", "stop scenario", "end scenario"}:
        return SocialCommand("story", "finish")
    if normalized in {"cancel story", "cancel scenario"}:
        return SocialCommand("story", "stop")
    if normalized in {"story status", "whose turn is it", "whose turn", "scenario status"}:
        return SocialCommand("story", "status")

    award_start = re.fullmatch(
        r"(?:start|begin|create)\s+(?:an?\s+|the\s+)?awards?"
        r"(?:\s+(?:for|called|named))?\s*(?P<category>.*)",
        clean,
        flags=re.IGNORECASE,
    )
    if award_start is not None:
        return SocialCommand("award", "start", argument=award_start.group("category").strip()[:100])
    if normalized in {"award results", "show award", "award status", "show nominations"}:
        return SocialCommand("award", "status")
    if normalized in {"end nominations", "finish award", "end award", "announce award"}:
        return SocialCommand("award", "finish")
    if normalized in {"cancel award", "stop award", "cancel awards", "stop awards"}:
        return SocialCommand("award", "stop")
    return None


def parse_poll_spec(body: str) -> tuple[str, tuple[str, ...]]:
    clean = _clean_request(body)
    if not clean:
        return "", ()
    if ":" in clean:
        question, raw_options = clean.split(":", 1)
        question = question.strip(" ,.!?") or "What should we choose?"
    else:
        question = "What should we choose?"
        raw_options = clean
    raw_options = re.sub(r"^(?:options?|choices?)\s+(?:are\s+)?", "", raw_options, flags=re.IGNORECASE)
    raw_options = re.sub(r",\s*or\s+", ", ", raw_options, flags=re.IGNORECASE)
    if "," in raw_options:
        parts = re.split(r"\s*,\s*|\s+or\s+", raw_options, flags=re.IGNORECASE)
    else:
        parts = re.split(r"\s+or\s+", raw_options, flags=re.IGNORECASE)
    options = tuple(
        dict.fromkeys(
            part.strip(" ,.!?")[:40]
            for part in parts
            if part.strip(" ,.!?")
        )
    )
    return question[:120], options[:5]


def game_question_bank(kind: str, category: str = "mixed") -> tuple[Any, ...]:
    if kind == "riddle":
        return RIDDLES
    if kind == "would":
        return WOULD_QUESTIONS
    if category == "wow":
        return WOW_TRIVIA
    if category == "general":
        return GENERAL_TRIVIA
    return WOW_TRIVIA + GENERAL_TRIVIA


def choose_game_question(state: GameState, rng: random.Random | Any = random) -> Any:
    bank = game_question_bank(state.kind, state.category)
    available = [index for index in range(len(bank)) if index not in state.used_questions]
    if not available:
        state.used_questions.clear()
        available = list(range(len(bank)))
    index = rng.choice(available)
    state.used_questions.add(index)
    state.current_question = bank[index]
    return state.current_question


def choose_twenty_question_secret(
    rng: random.Random | Any = random,
) -> TwentyQuestionSecret:
    return rng.choice(TWENTY_QUESTION_SECRETS)


def answer_matches(value: str, accepted: tuple[str, ...]) -> bool:
    candidate = normalize_social_text(value)
    candidate = re.sub(
        r"^(?:the\s+answer\s+is|answer|is\s+it|i\s+guess|my\s+guess\s+is|i\s+think\s+it\s+is)\s+",
        "",
        candidate,
    ).strip()
    for answer in accepted:
        expected = normalize_social_text(answer)
        if candidate == expected:
            return True
        if len(expected) >= 4 and re.search(rf"\b{re.escape(expected)}\b", candidate):
            return True
        if len(expected) >= 5 and SequenceMatcher(None, candidate, expected).ratio() >= 0.84:
            return True
    return False


def twenty_question_guess_matches(value: str, accepted: tuple[str, ...]) -> bool:
    candidate = normalize_social_text(value)
    guess_pattern = re.fullmatch(
        r"(?:(?:is\s+it|i\s+guess|my\s+guess\s+is|i\s+think\s+it\s+is)\s+)?(?P<guess>.+)",
        candidate,
    )
    if guess_pattern is None:
        return False
    explicit = bool(
        re.match(
            r"^(?:is\s+it|i\s+guess|my\s+guess\s+is|i\s+think\s+it\s+is)\b",
            candidate,
        )
    )
    guess = re.sub(r"^(?:a|an|the)\s+", "", guess_pattern.group("guess")).strip()
    for answer in accepted:
        expected = re.sub(
            r"^(?:a|an|the)\s+",
            "",
            normalize_social_text(answer),
        ).strip()
        if guess == expected:
            return True
        if explicit and len(expected) >= 5 and SequenceMatcher(None, guess, expected).ratio() >= 0.86:
            return True
    return False


_NUMBER_OPTIONS = {
    "1": 0,
    "one": 0,
    "first": 0,
    "a": 0,
    "2": 1,
    "two": 1,
    "second": 1,
    "b": 1,
    "3": 2,
    "three": 2,
    "third": 2,
    "c": 2,
    "4": 3,
    "four": 3,
    "fourth": 3,
    "d": 3,
    "5": 4,
    "five": 4,
    "fifth": 4,
    "e": 4,
}


def match_option(value: str, options: tuple[str, ...]) -> int | None:
    candidate = normalize_social_text(value)
    candidate = re.sub(
        r"^(?:i\s+vote\s+(?:for\s+)?|vote\s+(?:for\s+)?|my\s+vote\s+is\s+|"
        r"i\s+choose\s+|choose\s+|option\s+)",
        "",
        candidate,
    ).strip()
    numbered = _NUMBER_OPTIONS.get(candidate)
    if numbered is not None and numbered < len(options):
        return numbered
    normalized_options = [normalize_social_text(option) for option in options]
    exact = [index for index, option in enumerate(normalized_options) if candidate == option]
    if len(exact) == 1:
        return exact[0]
    partial = [
        index
        for index, option in enumerate(normalized_options)
        if len(candidate) >= 3 and (candidate in option or option in candidate)
    ]
    if len(partial) == 1:
        return partial[0]
    scored = sorted(
        (
            (SequenceMatcher(None, candidate, option).ratio(), index)
            for index, option in enumerate(normalized_options)
        ),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.78 and (
        len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.12
    ):
        return scored[0][1]
    return None


def extract_nomination_target(value: str) -> str | None:
    clean = _clean_request(value)
    match = re.fullmatch(
        r"(?:i\s+)?(?:nominate|vote\s+for|pick|choose)\s+(?P<target>.+)",
        clean,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    target = match.group("target").strip(" ,.!?-'\"")
    return target[:100] or None


_UNSAFE_AWARD_PATTERN = re.compile(
    r"\b(?:race|ethnicity|religion|sexual\s+orientation|gender\s+identity|disability|"
    r"medical|diagnosis|suicide|self\s*harm|real\s+name|address|phone|dox|criminal)\b",
    flags=re.IGNORECASE,
)


def award_category_is_safe(value: str) -> bool:
    clean = _clean_request(value)
    return bool(clean) and _UNSAFE_AWARD_PATTERN.search(clean) is None


_PARTY_SENSITIVE_PATTERN = re.compile(
    r"\b(?:password|passcode|address|phone\s+number|credit\s+card|social\s+security|"
    r"diagnos(?:is|ed)|medical|suicide|self\s*harm)\b",
    flags=re.IGNORECASE,
)
_PARTY_LOOKUP_PATTERN = re.compile(
    r"\b(?:weather|forecast|news|latest|today|tomorrow|stock|price|score|search|look\s+up)\b",
    flags=re.IGNORECASE,
)


def should_accept_party_ambient(
    transcript: str,
    duration_seconds: float,
    *,
    now: float,
    enabled_at: float,
    deadline: float,
    last_reaction_at: float,
    roll: float,
) -> bool:
    clean = _clean_request(transcript)
    words = clean.split()
    if now >= deadline or now - enabled_at < PARTY_AMBIENT_MIN_DELAY_SECONDS:
        return False
    if now - last_reaction_at < PARTY_AMBIENT_COOLDOWN_SECONDS:
        return False
    if duration_seconds < 0.8 or not 4 <= len(words) <= 45:
        return False
    if clean.endswith("?") or normalize_social_text(clean).split(" ", 1)[0] in {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
    }:
        return False
    if _PARTY_SENSITIVE_PATTERN.search(clean) or _PARTY_LOOKUP_PATTERN.search(clean):
        return False
    return roll < PARTY_AMBIENT_CHANCE
