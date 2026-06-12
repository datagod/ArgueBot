#!/usr/bin/env python3
"""
ArgueBot — Style-trained chatbot with Chatterbox TTS.

Upload text to teach the bot a voice/style, chat via local LLM (Ollama),
and hear replies spoken through your local Chatterbox TTS Docker server.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from prompts import build_system_prompt
from services import chatterbox, corpus, llm, rag
from services.corpus import get_stats
from services.settings_store import get_settings, init_db, update_settings

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "data"
AVATAR_DIR = DATA_DIR / "avatar"
DEFAULT_AVATAR = None

CUSTOM_CSS = """
.gradio-container { max-width: 1000px !important; margin: auto; }
.main-title { text-align: center; margin-bottom: 0.2em; }
.subtitle { text-align: center; color: #666; margin-bottom: 1.2em; }
.status-ok { color: #16a34a; font-weight: 600; }
.status-bad { color: #dc2626; font-weight: 600; }
.bot-profile-row { align-items: center; gap: 1rem; margin-bottom: 0.5rem; }
.bot-header { padding: 0.25rem 0 0 0.5rem; }
.bot-name { font-size: 1.75rem; font-weight: 700; margin: 0 0 0.35rem 0; line-height: 1.2; }
.bot-persona { color: #555; font-size: 0.95rem; margin: 0; line-height: 1.4; font-style: italic; }
.chat-input-row { margin-top: 0.5rem; }
"""


def _apply_env_defaults() -> None:
    """Seed settings from .env on first run."""
    mapping = {
        "LLM_BASE_URL": "llm_base_url",
        "LLM_MODEL": "llm_model",
        "LLM_TEMPERATURE": ("llm_temperature", float),
        "LLM_MAX_TOKENS": ("llm_max_tokens", int),
        "CHATTERBOX_BASE_URL": "chatterbox_base_url",
        "CHATTERBOX_VOICE_MODE": "chatterbox_voice_mode",
        "CHATTERBOX_PREDEFINED_VOICE": "chatterbox_predefined_voice",
        "CHATTERBOX_REFERENCE_VOICE": "chatterbox_reference_voice",
        "CHATTERBOX_TEMPERATURE": ("chatterbox_temperature", float),
        "CHATTERBOX_EXAGGERATION": ("chatterbox_exaggeration", float),
        "CHATTERBOX_CFG_WEIGHT": ("chatterbox_cfg_weight", float),
        "BOT_NAME": "bot_name",
        "BOT_PERSONA_BLURB": "bot_persona_blurb",
    }
    updates: dict = {}
    for env_key, target in mapping.items():
        value = os.getenv(env_key)
        if value is None:
            continue
        if isinstance(target, tuple):
            key, caster = target
            try:
                updates[key] = caster(value)
            except ValueError:
                continue
        else:
            updates[target] = value
    if updates:
        update_settings(updates)


def _bot_name() -> str:
    return get_settings().get("bot_name", "ArgueBot") or "ArgueBot"


def _avatar_path() -> str | None:
    settings = get_settings()
    path = settings.get("avatar_path")
    if path and Path(path).exists():
        return path
    for candidate in AVATAR_DIR.glob("*"):
        if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return str(candidate)
    return None


def _bot_header_html() -> str:
    name = _bot_name()
    persona = get_settings().get("bot_persona_blurb", "").strip()
    persona_html = (
        f'<p class="bot-persona">{persona}</p>' if persona else ""
    )
    return (
        f'<div class="bot-header">'
        f'<h2 class="bot-name">{name}</h2>'
        f"{persona_html}"
        f"</div>"
    )


def _question_input_kwargs() -> dict:
    name = _bot_name()
    return {
        "label": f"Ask {name}",
        "placeholder": f"Type your question for {name}…",
    }


def _corpus_status_html() -> str:
    stats = get_stats()
    words = stats.get("words", 0)
    chunks = stats.get("chunks", 0)
    if words == 0:
        return (
            '<span class="status-bad">No training text yet — '
            "upload samples on the Training tab for better style matching.</span>"
        )
    quality = "excellent" if words >= 10000 else "good" if words >= 2000 else "basic"
    return (
        f'<span class="status-ok">Style corpus: {words:,} words, {chunks} chunks '
        f"({quality} coverage).</span>"
    )


def _stats_markdown() -> str:
    stats = get_stats()
    lines = [
        f"- **Documents:** {stats['documents']}",
        f"- **Chunks:** {stats['chunks']}",
        f"- **Words:** {stats['words']:,}",
        f"- **Characters:** {stats['chars']:,}",
    ]
    if stats["document_list"]:
        lines.append("\n**Recent uploads:**")
        for doc in stats["document_list"][:10]:
            lines.append(
                f"- {doc['name']} ({doc['word_count']:,} words, {doc['source_type']})"
            )
    return "\n".join(lines)


def _settings_to_form() -> tuple:
    s = get_settings()
    return (
        s["llm_base_url"],
        s["llm_model"],
        s["llm_temperature"],
        s["llm_max_tokens"],
        s["chatterbox_base_url"],
        s.get("chatterbox_model", "chatterbox-turbo"),
        s.get("chatterbox_voice_mode", "clone"),
        s.get("chatterbox_predefined_voice", "Olivia.wav"),
        s.get("chatterbox_reference_voice", "kryten2.mp3"),
        s.get("chatterbox_temperature", 0.8),
        s.get("chatterbox_exaggeration", 0.5),
        s.get("chatterbox_cfg_weight", 0.5),
        s.get("chatterbox_speed_factor", 1.0),
        s["bot_name"],
        s.get("bot_persona_blurb", ""),
    )


def _voice_choices() -> tuple[list[str], list[str]]:
    try:
        voices = chatterbox.list_all_voices()
        predefined = voices.get("predefined") or ["Olivia.wav"]
        reference = voices.get("reference") or ["kryten2.mp3"]
    except Exception:
        predefined = ["Olivia.wav"]
        reference = ["kryten2.mp3"]
    return predefined, reference


def chat_respond(
    message: str,
    history: list,
) -> tuple[list, str | None, str, str | None, str]:
    if not message or not message.strip():
        return history, None, _corpus_status_html(), _avatar_path(), _bot_header_html()

    settings = get_settings()
    stats = get_stats()
    style_chunks = rag.retrieve(message)
    system_prompt = build_system_prompt(
        bot_name=settings["bot_name"],
        persona_blurb=settings.get("bot_persona_blurb", ""),
        style_excerpts=style_chunks,
        corpus_word_count=stats.get("words", 0),
    )

    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message.strip()})

    try:
        reply = llm.chat(messages)
    except llm.LLMError as exc:
        reply = f"⚠️ LLM error: {exc}"

    history = history + [[message.strip(), reply]]
    audio_path = None
    status = _corpus_status_html()

    if not reply.startswith("⚠️"):
        try:
            audio_path, _ = chatterbox.synthesize(reply)
            status += " &nbsp;|&nbsp; <span class='status-ok'>Speech generated.</span>"
        except chatterbox.ChatterboxError as exc:
            status += (
                f" &nbsp;|&nbsp; <span class='status-bad'>TTS failed: {exc}</span>"
            )

    return history, audio_path, status, _avatar_path(), _bot_header_html()


def add_pasted_text(text: str) -> str:
    if not text or not text.strip():
        return "⚠️ Paste some text first."
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        corpus.add_text(f"pasted_{ts}.txt", text, source_type="paste")
        return f"✅ Added pasted text.\n\n{_stats_markdown()}"
    except Exception as exc:
        return f"⚠️ Failed: {exc}"


def add_uploaded_files(files) -> str:
    if not files:
        return "⚠️ No files selected."
    results = []
    for file_obj in files:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        try:
            corpus.add_file(path)
            results.append(f"✅ {Path(path).name}")
        except Exception as exc:
            results.append(f"⚠️ {Path(path).name}: {exc}")
    return "\n".join(results) + f"\n\n{_stats_markdown()}"


def clear_corpus_action() -> str:
    corpus.clear_corpus()
    return f"Corpus cleared.\n\n{_stats_markdown()}"


def reindex_action() -> str:
    corpus.reindex()
    return f"Index rebuilt.\n\n{_stats_markdown()}"


def save_settings_action(
    llm_base_url,
    llm_model,
    llm_temperature,
    llm_max_tokens,
    chatterbox_base_url,
    chatterbox_model,
    voice_mode,
    predefined_voice,
    reference_voice,
    cb_temperature,
    cb_exaggeration,
    cb_cfg_weight,
    cb_speed_factor,
    bot_name,
    bot_persona,
    avatar_file,
) -> str:
    updates = {
        "llm_base_url": llm_base_url.strip(),
        "llm_model": llm_model.strip(),
        "llm_temperature": float(llm_temperature),
        "llm_max_tokens": int(llm_max_tokens),
        "chatterbox_base_url": chatterbox_base_url.strip(),
        "chatterbox_model": chatterbox_model,
        "chatterbox_voice_mode": voice_mode,
        "chatterbox_predefined_voice": predefined_voice,
        "chatterbox_reference_voice": reference_voice,
        "chatterbox_temperature": float(cb_temperature),
        "chatterbox_exaggeration": float(cb_exaggeration),
        "chatterbox_cfg_weight": float(cb_cfg_weight),
        "chatterbox_speed_factor": float(cb_speed_factor),
        "bot_name": bot_name.strip() or "ArgueBot",
        "bot_persona_blurb": bot_persona.strip(),
    }

    if avatar_file is not None:
        AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        src = avatar_file if isinstance(avatar_file, str) else avatar_file
        dest = AVATAR_DIR / f"avatar{Path(src).suffix or '.png'}"
        shutil.copy(src, dest)
        updates["avatar_path"] = str(dest)

    update_settings(updates)
    name = updates["bot_name"]
    return (
        "✅ Settings saved.",
        _avatar_path(),
        _bot_header_html(),
        gr.Textbox(
            label=f"Ask {name}",
            placeholder=f"Type your question for {name}…",
        ),
        gr.Audio(label=f"{name} speaking…"),
    )


def test_llm_action(llm_base_url, llm_model) -> str:
    update_settings({"llm_base_url": llm_base_url.strip(), "llm_model": llm_model.strip()})
    ok, msg = llm.test_connection()
    return f"{'✅' if ok else '⚠️'} {msg}"


def test_chatterbox_action(chatterbox_base_url) -> str:
    update_settings({"chatterbox_base_url": chatterbox_base_url.strip()})
    ok, msg = chatterbox.test_connection()
    return f"{'✅' if ok else '⚠️'} {msg}"


def test_voice_action(
    chatterbox_base_url,
    voice_mode,
    predefined_voice,
    reference_voice,
    cb_temperature,
    cb_exaggeration,
    cb_cfg_weight,
    cb_speed_factor,
) -> tuple[str, str | None]:
    update_settings(
        {
            "chatterbox_base_url": chatterbox_base_url.strip(),
            "chatterbox_voice_mode": voice_mode,
            "chatterbox_predefined_voice": predefined_voice,
            "chatterbox_reference_voice": reference_voice,
            "chatterbox_temperature": float(cb_temperature),
            "chatterbox_exaggeration": float(cb_exaggeration),
            "chatterbox_cfg_weight": float(cb_cfg_weight),
            "chatterbox_speed_factor": float(cb_speed_factor),
        }
    )
    try:
        path, _ = chatterbox.synthesize(
            "Hello! This is ArgueBot testing the Chatterbox voice."
        )
        return "✅ Voice test succeeded.", path
    except chatterbox.ChatterboxError as exc:
        return f"⚠️ Voice test failed: {exc}", None


def refresh_models_action(llm_base_url) -> gr.Dropdown:
    update_settings({"llm_base_url": llm_base_url.strip()})
    models = llm.list_models()
    if not models:
        models = [get_settings()["llm_model"]]
    return gr.Dropdown(choices=models, value=models[0])


def refresh_voices_action(chatterbox_base_url) -> tuple[gr.Dropdown, gr.Dropdown]:
    update_settings({"chatterbox_base_url": chatterbox_base_url.strip()})
    predefined, reference = _voice_choices()
    return (
        gr.Dropdown(choices=predefined, value=predefined[0] if predefined else None),
        gr.Dropdown(choices=reference, value=reference[0] if reference else None),
    )


def create_app() -> gr.Blocks:
    init_db()
    _apply_env_defaults()
    rag.rebuild_index()

    predefined_voices, reference_voices = _voice_choices()
    llm_models = llm.list_models() or [get_settings()["llm_model"]]
    form = _settings_to_form()

    with gr.Blocks(title="ArgueBot") as demo:
        gr.HTML(
            """
            <div style="text-align:center; padding: 0.5em 0 1em;">
                <h1 class="main-title" style="font-size: 2.4em; margin:0;">🗣️ ArgueBot</h1>
                <p class="subtitle" style="font-size:1.1em;">
                    Train a chatbot on <strong>your text</strong>, chat via <strong>local LLM</strong>,
                    speak through <strong>Chatterbox TTS</strong> (Docker)
                </p>
            </div>
            """
        )

        with gr.Tabs():
            # ── Chat ──────────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                corpus_status = gr.HTML(value=_corpus_status_html())

                with gr.Row(elem_classes=["bot-profile-row"]):
                    avatar_img = gr.Image(
                        value=_avatar_path(),
                        height=200,
                        width=200,
                        interactive=False,
                        show_label=False,
                        container=False,
                    )
                    bot_header = gr.HTML(value=_bot_header_html())

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=360,
                )

                with gr.Row(elem_classes=["chat-input-row"]):
                    q_kwargs = _question_input_kwargs()
                    msg_input = gr.Textbox(
                        label=q_kwargs["label"],
                        placeholder=q_kwargs["placeholder"],
                        scale=5,
                        lines=2,
                        max_lines=6,
                    )
                    send_btn = gr.Button("Ask", variant="primary", scale=1, min_width=100)

                reply_audio = gr.Audio(
                    label=f"{_bot_name()} speaking…",
                    type="filepath",
                    autoplay=True,
                    interactive=False,
                )

                send_btn.click(
                    chat_respond,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, reply_audio, corpus_status, avatar_img, bot_header],
                ).then(lambda: "", outputs=msg_input)

                msg_input.submit(
                    chat_respond,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, reply_audio, corpus_status, avatar_img, bot_header],
                ).then(lambda: "", outputs=msg_input)

                demo.load(
                    fn=lambda: (_avatar_path(), _bot_header_html()),
                    outputs=[avatar_img, bot_header],
                )

            # ── Training ────────────────────────────────────────────────────
            with gr.Tab("📚 Training"):
                gr.Markdown(
                    "Upload writing samples — posts, transcripts, articles. "
                    "**More text = better style matching.**"
                )
                with gr.Row():
                    file_upload = gr.File(
                        label="Upload files (.txt, .md, .csv, .pdf)",
                        file_count="multiple",
                        type="filepath",
                    )
                    paste_box = gr.Textbox(
                        label="Or paste text",
                        lines=12,
                        placeholder="Paste large blocks of text here…",
                    )

                with gr.Row():
                    upload_btn = gr.Button("Add files to corpus", variant="primary")
                    paste_btn = gr.Button("Add pasted text")
                    reindex_btn = gr.Button("Reindex")
                    clear_btn = gr.Button("Clear corpus", variant="stop")

                stats_md = gr.Markdown(value=_stats_markdown())

                upload_btn.click(add_uploaded_files, inputs=file_upload, outputs=stats_md)
                paste_btn.click(add_pasted_text, inputs=paste_box, outputs=stats_md)
                reindex_btn.click(reindex_action, outputs=stats_md)
                clear_btn.click(clear_corpus_action, outputs=stats_md)

            # ── Settings ──────────────────────────────────────────────────
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### LLM (Ollama / llama-server)")
                with gr.Row():
                    llm_url = gr.Textbox(label="LLM base URL", value=form[0])
                    llm_model_dd = gr.Dropdown(
                        label="Model",
                        choices=llm_models,
                        value=form[1],
                        allow_custom_value=True,
                    )
                    refresh_models_btn = gr.Button("Refresh models")

                with gr.Row():
                    llm_temp = gr.Slider(0.0, 2.0, value=form[2], label="Temperature")
                    llm_max_tok = gr.Slider(64, 4096, value=form[3], step=64, label="Max tokens")

                llm_test_btn = gr.Button("Test LLM connection")
                llm_test_out = gr.Textbox(label="LLM status", interactive=False)

                gr.Markdown(
                    "### Chatterbox TTS (Docker — default `http://127.0.0.1:8004`)\n"
                    "Manage voices and reference audio in the "
                    "[Chatterbox UI](http://127.0.0.1:8004) — ArgueBot only selects from what's already there."
                )
                with gr.Row():
                    cb_url = gr.Textbox(
                        label="Chatterbox base URL",
                        value=form[4],
                        info="Your chatterbox-tts-server-cu128 container",
                    )
                    cb_model = gr.Dropdown(
                        label="Model (informational)",
                        choices=chatterbox.KNOWN_MODELS,
                        value=form[5],
                        info="Change model in Chatterbox TTS Server UI if needed",
                    )

                with gr.Row():
                    voice_mode = gr.Radio(
                        ["clone", "predefined"],
                        value=form[6],
                        label="Voice mode",
                        info="clone = your reference audio; predefined = built-in voices",
                    )
                    predefined_dd = gr.Dropdown(
                        label="Predefined voice",
                        choices=predefined_voices,
                        value=form[7] if form[7] in predefined_voices else (
                            predefined_voices[0] if predefined_voices else None
                        ),
                    )
                    reference_dd = gr.Dropdown(
                        label="Reference / cloned voice",
                        choices=reference_voices,
                        value=form[8] if form[8] in reference_voices else (
                            reference_voices[0] if reference_voices else None
                        ),
                    )
                    refresh_voices_btn = gr.Button("Refresh voices")

                with gr.Row():
                    cb_temp = gr.Slider(0.05, 2.0, value=form[9], label="Temperature")
                    cb_exag = gr.Slider(0.25, 2.0, value=form[10], label="Exaggeration")
                    cb_cfg = gr.Slider(0.0, 1.0, value=form[11], label="CFG weight")
                    cb_speed = gr.Slider(0.5, 2.0, value=form[12], step=0.05, label="Speed")

                with gr.Row():
                    cb_test_btn = gr.Button("Test Chatterbox connection")
                    voice_test_btn = gr.Button("Test voice", variant="primary")

                cb_test_out = gr.Textbox(label="Chatterbox status", interactive=False)
                voice_test_audio = gr.Audio(label="Voice test playback", type="filepath")

                gr.Markdown("### Persona")
                with gr.Row():
                    bot_name_in = gr.Textbox(label="Bot name", value=form[13])
                    avatar_upload = gr.Image(label="Avatar image", type="filepath")
                bot_persona_in = gr.Textbox(
                    label="Persona / tone description",
                    value=form[14],
                    lines=3,
                )

                save_btn = gr.Button("💾 Save all settings", variant="primary", size="lg")
                save_out = gr.Textbox(label="Save status", interactive=False)

                # Wire up settings events
                refresh_models_btn.click(
                    refresh_models_action,
                    inputs=llm_url,
                    outputs=llm_model_dd,
                )
                refresh_voices_btn.click(
                    refresh_voices_action,
                    inputs=cb_url,
                    outputs=[predefined_dd, reference_dd],
                )
                llm_test_btn.click(
                    test_llm_action,
                    inputs=[llm_url, llm_model_dd],
                    outputs=llm_test_out,
                )
                cb_test_btn.click(
                    test_chatterbox_action,
                    inputs=cb_url,
                    outputs=cb_test_out,
                )
                voice_test_btn.click(
                    test_voice_action,
                    inputs=[
                        cb_url, voice_mode, predefined_dd, reference_dd,
                        cb_temp, cb_exag, cb_cfg, cb_speed,
                    ],
                    outputs=[cb_test_out, voice_test_audio],
                )
                save_btn.click(
                    save_settings_action,
                    inputs=[
                        llm_url, llm_model_dd, llm_temp, llm_max_tok,
                        cb_url, cb_model, voice_mode, predefined_dd, reference_dd,
                        cb_temp, cb_exag, cb_cfg, cb_speed,
                        bot_name_in, bot_persona_in, avatar_upload,
                    ],
                    outputs=[save_out, avatar_img, bot_header, msg_input, reply_audio],
                )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArgueBot — style-trained chatbot")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link (gradio.live)",
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="rose", secondary_hue="slate"),
        css=CUSTOM_CSS,
    )