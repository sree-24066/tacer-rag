"""
LLM Client for Evidence-Constrained Correction.

Provides interfaces to local Ollama models and OpenAI-compatible APIs.
Safely falls back to rule-based correction when external services are unreachable.
Never hardcodes API keys or passwords.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from tacer_rag.utils import get_logger

logger = get_logger("llm_client")


class LLMCorrectionClient:
    """
    Unified LLM interface for correction generation.
    Supports local Ollama (default) and OpenAI API.
    """

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.2",
        temperature: float = 0.0,
        ollama_url: str = "http://localhost:11434",
    ):
        self.backend = backend.lower()
        self.model = model
        self.temperature = temperature
        self.ollama_url = ollama_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Generate completion using configured backend.
        Returns None if backend is unavailable (triggering rule-based fallback).
        """
        if self.backend == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        elif self.backend == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.backend == "none" or self.backend == "rule_based":
            return None
        else:
            logger.warning("Unknown LLM backend '%s'. Using rule-based fallback.", self.backend)
            return None

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call local Ollama /api/chat endpoint."""
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
            },
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError) as e:
            logger.debug("Ollama is not reachable at %s (%s). Using rule-based fallback.", url, e)
            return None

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call OpenAI chat completion using environment API key."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.debug("OPENAI_API_KEY not set. Using rule-based fallback.")
            return None

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model if "gpt" in self.model else "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.warning("OpenAI API call failed (%s). Using rule-based fallback.", e)
            return None
        return None
