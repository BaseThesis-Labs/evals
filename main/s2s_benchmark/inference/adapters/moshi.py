"""
Moshi S2S adapter — STUB.

Enabled: false in eval_config.yaml.
Moshi (Kyutai, 2024) is a full-duplex speech LM.

References:
  - https://kyutai.org/moshi
  - Model: moshi-v0.1 (~8B, requires GPU for reasonable speed)
  - Python client: pip install moshi
"""
from __future__ import annotations

from inference.adapters.base import BaseS2SAdapter, S2SResult


class MoshiAdapter(BaseS2SAdapter):
    """Moshi full-duplex Speech LM adapter (stub).

    Config keys:
        model      (str): e.g. "moshi-v0.1"
        device     (str): "cuda" (required for real-time), "cpu" (slow)
        server_url (str): optional; connect to running moshi server
    """

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.moshi_model = config.get("model", "moshi-v0.1")
        self.device = config.get("device", "cpu")
        self.server_url = config.get("server_url")

    def process(
        self,
        audio_in_path: str,
        reference_text: str,
        utterance_id: str,
        output_dir: str,
    ) -> S2SResult:
        raise NotImplementedError(
            "MoshiAdapter is a stub. "
            "Set enabled: false in config/eval_config.yaml or implement the "
            "Moshi client (pip install moshi) here."
        )

    def cleanup(self) -> None:
        pass
