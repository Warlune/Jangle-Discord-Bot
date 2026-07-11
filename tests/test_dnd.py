from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from discord_ai.dnd import (
    DND_MAX_JOURNAL_ENTRIES,
    DndCampaignStore,
    DndCharacter,
    campaign_context,
    choose_check,
    roll_check,
    scene_guidance,
)


class FixedRoller:
    def __init__(self, *values: int) -> None:
        self.values = iter(values)

    def __call__(self, _expression: str) -> SimpleNamespace:
        return SimpleNamespace(total=next(self.values))


def test_checks_begin_easy_and_build_to_a_fair_hard_scene() -> None:
    character = DndCharacter.create(1, "Arden", 0)

    opening = choose_check("I attack the wooden gate", character, 1)
    climax = choose_check("I attack the legendary boss", character, 3)

    assert (opening.ability, opening.dc, opening.risky) == ("strength", 12, True)
    assert climax.dc == 16
    assert "local problem" in scene_guidance(1)
    assert "Pay off an earlier clue" in scene_guidance(3)


def test_roll_result_and_character_changes_are_owned_by_code() -> None:
    character = DndCharacter.create(1, "Arden", 0)
    check = choose_check("I attack the guard", character, 1)

    success = roll_check(character, check, "I attack the guard", roller=FixedRoller(12))
    failure = roll_check(character, check, "I attack the guard", roller=FixedRoller(1, 3))

    assert success.success is True
    assert success.total == 17
    assert failure.fumble is True
    assert failure.damage == 5
    assert character.hp == character.max_hp - 5
    assert character.successes == 1
    assert character.failures == 1


def test_xp_levels_up_and_restores_the_character() -> None:
    character = DndCharacter.create(1, "Arden", 2)
    character.hp = 1

    levels = character.award_xp(60)

    assert levels == 1
    assert character.level == 2
    assert character.hp == character.max_hp


def test_campaign_and_characters_survive_reload_and_name_changes(tmp_path: Path) -> None:
    path = tmp_path / "dnd.json"
    store = DndCampaignStore(path)
    bundle = store.start_campaign(
        10,
        1,
        "Host",
        [(1, "Host"), (2, "Guest")],
        "haunted mines",
    )
    bundle.characters[1].award_xp(60)
    bundle.campaign.add_journal("The party found three scratched symbols beneath the lift.")
    store.save(10, bundle)

    reloaded = DndCampaignStore(path).load_active(10)

    assert reloaded is not None
    assert reloaded.campaign.theme == "haunted mines"
    assert reloaded.characters[1].level == 2
    assert reloaded.campaign.journal[-1].startswith("The party found")

    DndCampaignStore(path).finish(10, reloaded, "campaign completed")
    fresh_store = DndCampaignStore(path)
    assert fresh_store.load_active(10) is None
    assert fresh_store.load_characters(10)[1].level == 2
    renamed = fresh_store.start_campaign(10, 1, "New Host", [(1, "New Host")], "old ruins")
    assert renamed.characters[1].name == "New Host"
    assert renamed.characters[1].level == 2


def test_journal_is_bounded_before_it_reaches_the_model(tmp_path: Path) -> None:
    store = DndCampaignStore(tmp_path / "dnd.json")
    bundle = store.start_campaign(10, 1, "Host", [(1, "Host")], "ruins")

    for index in range(DND_MAX_JOURNAL_ENTRIES + 8):
        bundle.campaign.add_journal(f"Event {index}")
    store.save(10, bundle)

    reloaded = DndCampaignStore(tmp_path / "dnd.json").load_active(10)
    assert reloaded is not None
    assert len(reloaded.campaign.journal) == DND_MAX_JOURNAL_ENTRIES
    assert reloaded.campaign.journal[0] == "Event 8"


def test_model_context_keeps_recent_journal_and_party_inside_gateway_limit(tmp_path: Path) -> None:
    store = DndCampaignStore(tmp_path / "dnd.json")
    participants = [(index, f"Character With A Long Name {index}") for index in range(1, 11)]
    bundle = store.start_campaign(10, 1, "Host", participants, "a remote border village")
    bundle.campaign.opening = "A mill stops turning while the river continues to run."
    for index in range(8):
        bundle.campaign.add_journal(f"Consequence {index}: " + "a fresh development " * 20)

    context = campaign_context(bundle)

    assert len(context) <= 2000
    assert "Opening anchor: A mill stops turning" in context
    assert "Consequence 7" in context
    assert "Character With A Long Name 10" in context
