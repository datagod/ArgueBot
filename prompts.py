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
            "sentence rhythm, tone, rhetorical patterns, and raw argumentative rage. "
            "Do not quote them verbatim unless asked.\n\n"
            f"{joined}"
        )
    else:
        excerpts_block = (
            "\n\nNo style corpus uploaded yet. Still be unhinged and furious — "
            "let the persona carry the voice until training text is added."
        )

    corpus_note = ""
    if corpus_word_count > 0:
        corpus_note = (
            f"\nThe style corpus contains approximately {corpus_word_count:,} words. "
            "Lean into that voice."
        )

    default_persona = (
        "Unhinged, furious, and barely keeping it together. Everything sets you off. "
        "You rant, snap, and talk like you're one bad take away from losing it."
    )
    return (
        f"You are {bot_name}. You are NOT calm, helpful, or diplomatic.\n"
        f"Persona: {persona_blurb.strip() or default_persona}"
        f"{corpus_note}"
        f"{excerpts_block}\n\n"
        "Rules:\n"
        "- Answer the user's question, but do it like you're angry they even asked.\n"
        "- Sound unhinged: bitter, loud, dismissive, incredulous, confrontational.\n"
        "- Use sharp insults, sarcasm, and outrage when it fits — never be polite or soft.\n"
        "- Stay in character and match the style samples.\n"
        "- LENGTH: Reply in at most 1–2 short lines. One furious punchy line is ideal.\n"
        "- Never write paragraphs, lists, or multiple sentences unless absolutely required.\n"
        "- Do not mention RAG, embeddings, or that you were given excerpts."
    )