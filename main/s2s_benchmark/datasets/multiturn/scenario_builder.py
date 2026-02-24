"""
Scenario builder — loads and validates multi-turn scenario YAML files.
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


@dataclass
class ContextCheck:
    description: str
    check_type: str  # "response_contains_all", "response_contains_any", "llm_judge"
    expected: List[str] = field(default_factory=list)
    prompt: str = ""  # for llm_judge type


@dataclass
class UserTurn:
    turn_index: int
    text: str
    speech_style: str = "neutral"
    context_check: Optional[ContextCheck] = None


@dataclass
class ContextProbe:
    inject_after_turn: int
    probe_text: str
    expected_contains: List[str]
    measures: str = "context_retention"


@dataclass
class Scenario:
    scenario_id: str
    category: str
    difficulty: str
    description: str
    n_turns: int
    system_prompt: str
    user_turns: List[UserTurn]
    context_probes: List[ContextProbe] = field(default_factory=list)
    success_criteria: Dict = field(default_factory=dict)
    voice_checks: List[Dict] = field(default_factory=list)
    max_turns: int = 15


def load_scenario(path: Path) -> Scenario:
    """Load a single scenario from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    user_turns = []
    for ut in data.get("user_turns", []):
        cc = None
        if "context_check" in ut:
            cc_data = ut["context_check"]
            cc = ContextCheck(
                description=cc_data.get("description", ""),
                check_type=cc_data.get("check_type", "response_contains_any"),
                expected=cc_data.get("expected", []),
                prompt=cc_data.get("prompt", ""),
            )
        user_turns.append(UserTurn(
            turn_index=ut["turn_index"],
            text=ut["text"],
            speech_style=ut.get("speech_style", "neutral"),
            context_check=cc,
        ))

    probes = []
    for p in data.get("context_probes", []):
        probes.append(ContextProbe(
            inject_after_turn=p["inject_after_turn"],
            probe_text=p["probe_text"],
            expected_contains=p.get("expected_contains", []),
            measures=p.get("measures", "context_retention"),
        ))

    return Scenario(
        scenario_id=data["scenario_id"],
        category=data.get("category", "general"),
        difficulty=data.get("difficulty", "medium"),
        description=data.get("description", ""),
        n_turns=data.get("n_turns", len(user_turns) * 2),
        system_prompt=data.get("system_prompt", ""),
        user_turns=user_turns,
        context_probes=probes,
        success_criteria=data.get("success_criteria", {}),
        voice_checks=data.get("voice_checks", []),
        max_turns=data.get("max_turns", 15),
    )


def load_all_scenarios(scenarios_dir) -> List[Scenario]:
    """Load all scenario YAML files from a directory."""
    scenarios_dir = Path(scenarios_dir)
    scenarios = []
    for yaml_file in sorted(scenarios_dir.glob("*.yaml")):
        if yaml_file.name.startswith("_"):
            continue
        try:
            scenarios.append(load_scenario(yaml_file))
        except Exception as exc:
            print(f"  ⚠ Failed to load scenario {yaml_file.name}: {exc}")
    return scenarios
