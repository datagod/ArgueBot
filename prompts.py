"""System prompt construction for style-aware chat."""

from __future__ import annotations

from personas import get_persona_preset


def build_system_prompt(
    bot_name: str,
    persona_mode: str,
    style_excerpts: list[str],
    corpus_word_count: int,
) -> str:
    preset = get_persona_preset(persona_mode)

    if style_excerpts:
        joined = "\n\n---\n\n".join(
            f"[Excerpt {i + 1}]\n{excerpt.strip()}"
            for i, excerpt in enumerate(style_excerpts)
        )
        excerpts_block = (
            f"\n\nStudy these writing samples and match their {preset['excerpt_tone']}. "
            "Do not quote them verbatim unless asked.\n\n"
            f"{joined}"
        )
    else:
        excerpts_block = f"\n\n{preset['empty_corpus_note']}"

    corpus_note = ""
    if corpus_word_count > 0:
        corpus_note = (
            f"\nThe style corpus contains approximately {corpus_word_count:,} words. "
            "Lean into that voice."
        )

    tone_rules = "\n".join(f"- {rule}" for rule in preset["tone_rules"])
    return (
        f"{preset['intro'].format(name=bot_name)}\n"
        f"Persona ({persona_mode}): {preset['blurb']}"
        f"{corpus_note}"
        f"{excerpts_block}\n\n"
        "Rules:\n"
        f"{tone_rules}\n"
        "- Stay in character and match the style samples.\n"
        "- LENGTH: Reply in at most 1–2 short lines. One punchy line is ideal.\n"
        "- Never write paragraphs, lists, or multiple sentences unless absolutely required.\n"
        "- Do not mention RAG, embeddings, or that you were given excerpts."
    )