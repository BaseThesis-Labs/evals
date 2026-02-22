"""src/config.py — YAML config loader with pydantic v2 validation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


# ── Model config ──────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    name: str
    cls: str = Field(alias="class")
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    api_key_env: str | None = None
    cost_per_minute_usd: float | None = None

    model_config = {"populate_by_name": True}


class ModelsConfig(BaseModel):
    models: list[ModelConfig]

    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]


# ── Dataset config ────────────────────────────────────────────────────────────

class DatasetConfig(BaseModel):
    type: str                          # "manifest" or "directory"
    manifest_path: str | None = None
    audio_dir: str | None = None
    description: str = ""
    download_url: str | None = None
    subset: str | None = None
    group_by: list[str] = Field(default_factory=list)
    auto_generate: bool = False


class DatasetsConfig(BaseModel):
    datasets: dict[str, DatasetConfig]


# ── Evaluation config ─────────────────────────────────────────────────────────

class NormalizationConfig(BaseModel):
    lowercase: bool = True
    expand_contractions: bool = True
    remove_fillers: bool = True
    remove_punctuation: bool = True
    filler_words: list[str] = Field(
        default_factory=lambda: ["uh", "um", "hmm", "mhm", "ah", "er"]
    )


class SemanticConfig(BaseModel):
    enabled: bool = True
    semdist_model: str = "all-MiniLM-L6-v2"
    bertscore: bool = True
    bertscore_model: str = "roberta-large"
    meaning_preservation_llm: bool = False


class MetricsConfig(BaseModel):
    surface: dict[str, Any] = Field(default_factory=dict)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    formatting: dict[str, Any] = Field(default_factory=dict)
    entity: dict[str, Any] = Field(default_factory=dict)
    latency: dict[str, Any] = Field(default_factory=dict)
    hallucination: dict[str, Any] = Field(default_factory=dict)


class StatisticalConfig(BaseModel):
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    blockwise_bootstrap: bool = True
    block_field: str = "speaker_id"
    seed: int = 42


class AdvancedConfig(BaseModel):
    enabled: bool = True
    semascore: bool = True
    bleu: bool = True
    meteor: bool = True
    her: bool = True
    per: bool = True
    error_severity: bool = True
    shallow: bool = True
    keywords: list[str] = Field(default_factory=list)
    llm_judge: bool = False
    llm_judge_model: str = "gpt-4o-mini"


class EvaluationConfig(BaseModel):
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    reporting: dict[str, Any] = Field(default_factory=dict)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)


# ── Case study config ─────────────────────────────────────────────────────────

class WeightsConfig(BaseModel):
    intelligibility: float = 0.25
    semantic: float = 0.20
    latency: float = 0.20
    formatting: float = 0.15
    hallucination: float = 0.10
    entity: float = 0.10
    safety: float = 0.00   # avg_error_severity + impact_score + krr


class MetricBoundConfig(BaseModel):
    lower_is_better: bool
    min: float
    max: float


class CaseStudyConfig(BaseModel):
    description: str = ""
    datasets: list[str]
    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    primary_metrics: list[str] = Field(default_factory=list)


class CaseStudiesConfig(BaseModel):
    case_studies: dict[str, CaseStudyConfig]
    metric_bounds: dict[str, MetricBoundConfig] = Field(default_factory=dict)


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        log.warning(f"Config not found: {p}")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_models_config(path: str = "configs/models.yaml") -> ModelsConfig:
    return ModelsConfig(**_load_yaml(path))


def load_datasets_config(path: str = "configs/datasets.yaml") -> DatasetsConfig:
    return DatasetsConfig(**_load_yaml(path))


def load_evaluation_config(path: str = "configs/evaluation.yaml") -> EvaluationConfig:
    return EvaluationConfig(**_load_yaml(path))


def load_case_studies_config(path: str = "configs/case_studies.yaml") -> CaseStudiesConfig:
    return CaseStudiesConfig(**_load_yaml(path))


def load_all_configs(config_dir: str = "configs") -> dict:
    d = Path(config_dir)
    return {
        "models":       load_models_config(d / "models.yaml"),
        "datasets":     load_datasets_config(d / "datasets.yaml"),
        "evaluation":   load_evaluation_config(d / "evaluation.yaml"),
        "case_studies": load_case_studies_config(d / "case_studies.yaml"),
    }
