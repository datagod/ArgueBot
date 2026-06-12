"""OpenAI-compatible LLM client (Ollama / llama-server)."""

from __future__ import annotations

import requests

from services.settings_store import get_settings


class LLMError(Exception):
    pass


def _base_url() -> str:
    return get_settings()["llm_base_url"].rstrip("/")


def list_models() -> list[str]:
    base = _base_url()
    models: list[str] = []

    try:
        resp = requests.get(f"{base}/v1/models", timeout=10)
        if resp.ok:
            data = resp.json()
            for item in data.get("data", []):
                model_id = item.get("id")
                if model_id:
                    models.append(model_id)
    except requests.RequestException:
        pass

    if not models:
        try:
            resp = requests.get(f"{base}/api/tags", timeout=10)
            if resp.ok:
                data = resp.json()
                for item in data.get("models", []):
                    name = item.get("name")
                    if name:
                        models.append(name)
        except requests.RequestException:
            pass

    return sorted(set(models))


def test_connection() -> tuple[bool, str]:
    try:
        models = list_models()
        if models:
            return True, f"Connected. {len(models)} model(s) available."
        return False, "Reachable but no models found. Check your LLM server."
    except requests.RequestException as exc:
        return False, f"Connection failed: {exc}"


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    settings = get_settings()
    payload = {
        "model": model or settings["llm_model"],
        "messages": messages,
        "temperature": temperature if temperature is not None else settings["llm_temperature"],
        "max_tokens": max_tokens if max_tokens is not None else settings["llm_max_tokens"],
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{_base_url()}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
    except requests.RequestException as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc

    if not resp.ok:
        raise LLMError(f"LLM error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response: {data}") from exc