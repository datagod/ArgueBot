"""Avatar image processing: center-crop square and resize small."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

AVATAR_SIZE = 128
AVATAR_FILENAME = "avatar.jpg"


def process_avatar(src: str | Path, dest_dir: str | Path) -> str:
    """Center-crop to square, resize, and save a compact JPEG."""
    src_path = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / AVATAR_FILENAME

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((AVATAR_SIZE, AVATAR_SIZE), Image.Resampling.LANCZOS)
        img.save(dest_path, format="JPEG", quality=85, optimize=True)

    return str(dest_path)