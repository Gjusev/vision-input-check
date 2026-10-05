"""Built-in image variants: identical re-encodes and lossy stress variants.

``original`` is the canonical PNG: the suite source image decoded and
re-encoded as RGB PNG. It is not the raw bytes of the file on disk; every
variant starts from the same decoded source, so deltas measure the
variant's encoding, not the source file's container.
"""

from __future__ import annotations

import io

from PIL import Image, ImageChops

from .types import VariantSpec

IDENTICAL_VARIANTS = ("original", "png-reencode", "exif-strip")
LOSSY_VARIANTS = ("jpeg-75", "jpeg-40", "resize-50", "grayscale", "webp")

_MIME = {
    "original": "image/png",
    "png-reencode": "image/png",
    "exif-strip": "image/png",
    "jpeg-75": "image/jpeg",
    "jpeg-40": "image/jpeg",
    "resize-50": "image/png",
    "grayscale": "image/png",
    "webp": "image/webp",
}


def media_type_for(variant: str) -> str:
    """MIME type the API would receive for one variant's bytes."""
    try:
        return _MIME[variant]
    except KeyError as error:
        raise KeyError(f"unknown variant {variant!r}") from error


def _load(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _to_bytes(image: Image.Image, format_name: str, **kwargs: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name, **kwargs)
    return buffer.getvalue()


def apply_variant(image: Image.Image, name: str) -> bytes:
    """Return the encoded bytes the API would receive for one variant."""
    if name == "original":
        return _png_bytes(image)
    if name in ("png-reencode", "exif-strip"):
        clean = image.copy()
        clean.info = {}  # drop EXIF and metadata before re-encoding
        return _png_bytes(clean)
    if name == "jpeg-75":
        return _to_bytes(image, "JPEG", quality=75)
    if name == "jpeg-40":
        return _to_bytes(image, "JPEG", quality=40)
    if name == "resize-50":
        # Half each side; floor at 1 pixel so very small images still encode.
        small = image.resize((max(1, image.width // 2), max(1, image.height // 2)))
        return _png_bytes(small)
    if name == "grayscale":
        return _png_bytes(image.convert("L").convert("RGB"))
    if name == "webp":
        return _to_bytes(image, "WEBP", quality=80)
    raise KeyError(f"unknown variant {name!r}")


def is_pixel_identical(a: bytes, b: bytes) -> bool:
    """True when two encoded images carry exactly the same pixels.

    Compares dimensions, band set (mode family), and per-pixel values.
    A mode change such as RGBA -> RGB or RGB -> L is a destructive
    conversion and therefore never classified as pixel-identical, even if
    the RGB values happen to survive.
    """
    with Image.open(io.BytesIO(a)) as ia, Image.open(io.BytesIO(b)) as ib:
        if ia.size != ib.size or ia.getbands() != ib.getbands():
            return False
        return ImageChops.difference(ia, ib).getbbox() is None


def verify_variant_class(name: str, source: Image.Image, encoded: bytes) -> bool:
    """Verify the identical-class contract: pixels unchanged, bytes may differ.

    Lossy and unknown variants are never identical by definition, so they
    fail this check without an image comparison.
    """
    if name not in IDENTICAL_VARIANTS:
        return False
    return is_pixel_identical(_png_bytes(source), encoded)


BUILTIN_VARIANTS: list[VariantSpec] = [
    VariantSpec(name="original", kind="identical"),
    VariantSpec(name="png-reencode", kind="identical"),
    VariantSpec(name="exif-strip", kind="identical"),
    VariantSpec(name="jpeg-75", kind="lossy"),
    VariantSpec(name="jpeg-40", kind="lossy"),
    VariantSpec(name="resize-50", kind="lossy"),
    VariantSpec(name="grayscale", kind="lossy"),
    VariantSpec(name="webp", kind="lossy"),
]
