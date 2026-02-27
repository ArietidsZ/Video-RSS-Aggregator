from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from threading import Lock

from vllm import LLM, SamplingParams

from .config import Config

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a concise video content summarizer. Given a transcript, produce a JSON object with exactly two keys:
- "summary": a single paragraph summarizing the content.
- "key_points": a JSON array of 3-7 short bullet strings capturing the main ideas.
Output ONLY valid JSON, no markdown fences, no extra text."""

_USER_TEMPLATE = "Transcript:\n\n{text}\n\nSummarize the above transcript."


@dataclass(slots=True)
class SummaryResult:
    summary: str
    key_points: list[str] = field(default_factory=list)


class SummarizationEngine:
    """Singleton wrapper around Qwen3-8B-AWQ via vLLM."""

    _instance: SummarizationEngine | None = None
    _lock = Lock()

    def __init__(self, config: Config) -> None:
        log.info("Loading LLM %s (gpu_mem=%.2f)", config.llm_model, config.gpu_memory_utilization)
        self._llm = LLM(
            model=config.llm_model,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=8192,
            enforce_eager=True,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._max_tokens = config.llm_max_tokens
        log.info("LLM loaded")

    @classmethod
    def get(cls, config: Config) -> SummarizationEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    def summarize(self, text: str) -> SummaryResult:
        """Run summarization. Blocking — call from a worker thread."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=self._max_tokens,
        )
        outputs = self._llm.generate([prompt], params, use_tqdm=False)
        raw = outputs[0].outputs[0].text.strip()
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> SummaryResult:
        # Strip markdown fences if model adds them despite instructions
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            return SummaryResult(
                summary=data.get("summary", text),
                key_points=data.get("key_points", []),
            )
        except json.JSONDecodeError:
            return SummaryResult(summary=text, key_points=[])
