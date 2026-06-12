"""System prompt construction for style-aware chat."""

from __future__ import annotations


def build_system_prompt(
    bot_name: str,
    persona_blurb: str,
    style_excerpts: list[str],
    corpus_word_count: int,
) -> str:
    excerpts_block = ""
    if style_excerpts:
        joined = "\n\n---\n\n".join(
            f"[Excerpt {i + 1}]\n{excerpt.strip()}"
            for i, excerpt in enumerate(style_excerpts)
        )
        excerpts_block = (
            "\n\nStudy these writing samples and match their style — vocabulary, "
            "sentence rhythm, tone, rhetorical patterns, and argumentative energy. "
            "Do not quote them verbatim unless asked.\n\n"
            f"{joined}"
        )
    else:
        excerpts_block = (
            "\n\nNo style corpus uploaded yet. Use the persona description and "
            "respond helpfully until training text is added."
        )

    corpus_note = ""
    if corpus_word_count > 0:
        corpus_note = (
            f"\nThe style corpus contains approximately {corpus_word_count:,} words. "
            "Lean into that voice."
        )

    return (
        f"You are {bot_name}, a conversational AI that speaks in a distinctive style.\n"
        f"Persona: {persona_blurb.strip() or 'A sharp, opinionated conversationalist.'}"
        f"{corpus_note}"
        f"{excerpts_block}\n\n"
        "Rules:\n"
        "- Answer the user's actual question or request.\n"
        "- Stay in character and match the style samples.\n"
        "- LENGTH: Reply in at most 1–2 short lines. One punchy line is ideal.\n"
        "- Never write paragraphs, lists, or multiple sentences unless absolutely required.\n"
        "- Do not mention RAG, embeddings, or that you were given excerpts."
    )