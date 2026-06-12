"""Client for devnen Chatterbox-TTS-Server (Docker on port 8004)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from services.settings_store import get_settings

# Known HuggingFace repo IDs supported by Chatterbox-TTS-Server
KNOWN_MODELS = [
    "chatterbox-turbo",
    "ResembleAI/chatterbox",
    "ResembleAI/chatterbox-multilingual",
]


class ChatterboxError(Exception):
    pass


def _base_url() -> str:
    return get_settings()["chatterbox_base_url"].rstrip("/")


def get_model_info() -> dict[str, Any]:
    resp = requests.get(f"{_base_url()}/api/model-info", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_initial_data() -> dict[str, Any]:
    resp = requests.get(f"{_base_url()}/api/ui/initial-data", timeout=15)
    resp.raise_for_status()
    return resp.json()


def list_predefined_voices() -> list[str]:
    try:
        data = get_initial_data()
        voices = data.get("predefined_voices", [])
        return [v.get("filename", v) if isinstance(v, dict) else str(v) for v in voices]
    except requests.RequestException:
        resp = requests.get(f"{_base_url()}/get_predefined_voices", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("voices", [])


def list_reference_voices() -> list[str]:
    try:
        data = get_initial_data()
        return list(data.get("reference_files", []))
    except requests.RequestException:
        resp = requests.get(f"{_base_url()}/get_reference_files", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("files", [])


def list_all_voices() -> dict[str, list[str]]:
    return {
        "predefined": list_predefined_voices(),
        "reference": list_reference_voices(),
    }


def get_current_model() -> str:
    try:
        data = get_initial_data()
        return data.get("config", {}).get("model", {}).get("repo_id", "chatterbox-turbo")
    except requests.RequestException:
        return "chatterbox-turbo"


def test_connection() -> tuple[bool, str]:
    try:
        info = get_model_info()
        if info.get("loaded"):
            model_type = info.get("type", "unknown")
            device = info.get("device", "unknown")
            return True, f"Chatterbox ready ({model_type} on {device})."
        return False, "Chatterbox reachable but model not loaded."
    except requests.RequestException as exc:
        return False, f"Chatterbox connection failed: {exc}"


def synthesize(text: str, output_path: Path | None = None) -> tuple[str, bytes]:
    """Generate speech WAV bytes. Returns (path, bytes)."""
    settings = get_settings()
    voice_mode = settings.get("chatterbox_voice_mode", "clone")
    payload: dict[str, Any] = {
        "text": text,
        "voice_mode": voice_mode,
        "output_format": "wav",
        "split_text": True,
        "temperature": settings.get("chatterbox_temperature"),
        "exaggeration": settings.get("chatterbox_exaggeration"),
        "cfg_weight": settings.get("chatterbox_cfg_weight"),
        "speed_factor": settings.get("chatterbox_speed_factor", 1.0),
    }

    if voice_mode == "predefined":
        payload["predefined_voice_id"] = settings.get(
            "chatterbox_predefined_voice", "Olivia.wav"
        )
    else:
        payload["reference_audio_filename"] = settings.get(
            "chatterbox_reference_voice", "kryten2.mp3"
        )

    try:
        resp = requests.post(
            f"{_base_url()}/tts",
            json=payload,
            timeout=600,
        )
    except requests.RequestException as exc:
        raise ChatterboxError(f"TTS request failed: {exc}") from exc

    if not resp.ok:
        raise ChatterboxError(f"TTS error {resp.status_code}: {resp.text[:500]}")

    audio = resp.content
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "last_reply.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio)
    return str(output_path), audio

