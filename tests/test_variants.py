"""Variant encodings and pixel identity rules."""

import io

import pytest
from PIL import Image

from vision_input_check.variants import (
    BUILTIN_VARIANTS,
    apply_variant,
    is_pixel_identical,
    media_type_for,
    verify_variant_class,
)


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _source():
    image = Image.new("RGB", (64, 48), "white")
    for x in range(0, 64, 4):
        for y in range(0, 48, 4):
            image.putpixel((x, y), (x * 4 % 256, y * 5 % 256, 128))
    return image


def _encode_all(image):
    return {spec.name: apply_variant(image, spec.name) for spec in BUILTIN_VARIANTS}


class TestIdentity:
    def test_identical_variants_preserve_pixels(self):
        image = _source()
        encoded = _encode_all(image)
        original = encoded["original"]
        for name in ("png-reencode", "exif-strip"):
            assert is_pixel_identical(original, encoded[name]), name
            assert verify_variant_class(name, image, encoded[name])

    def test_original_contract(self):
        image = _source()
        assert verify_variant_class("original", image, apply_variant(image, "original"))

    def test_lossy_variants_are_not_identical(self):
        image = _source()
        encoded = _encode_all(image)
        original = encoded["original"]
        for name in ("jpeg-75", "jpeg-40", "resize-50", "grayscale", "webp"):
            assert not is_pixel_identical(original, encoded[name]), name
            assert not verify_variant_class(name, image, encoded[name])

    def test_lossy_kind_never_counts_as_identical(self):
        # Even byte-identical output would not make a lossy variant identical.
        gray = _png(_source().convert("L").convert("RGB"))
        assert not verify_variant_class("grayscale", _source(), gray)

    def test_destructive_mode_change_is_not_identical(self):
        # RGB bytes vs RGBA bytes with the same RGB values: alpha loss is a
        # real change, so this must not classify as pixel-identical.
        rgb = _png(_source())
        rgba = _source().convert("RGBA")
        with_alpha = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        with_alpha.alpha_composite(rgba)
        assert not is_pixel_identical(rgb, _png(with_alpha))

    def test_size_change_is_not_identical(self):
        image = _source()
        smaller = image.resize((32, 24))
        assert not is_pixel_identical(_png(image), _png(smaller))


class TestEncoding:
    def test_tiny_image_resize_does_not_crash(self):
        image = Image.new("RGB", (2, 2), "red")
        small = Image.open(io.BytesIO(apply_variant(image, "resize-50")))
        assert small.size == (1, 1)

    def test_one_pixel_image(self):
        image = Image.new("RGB", (1, 1), "blue")
        for spec in BUILTIN_VARIANTS:
            data = apply_variant(image, spec.name)
            assert Image.open(io.BytesIO(data)).size == (1, 1), spec.name

    def test_unknown_variant(self):
        with pytest.raises(KeyError):
            apply_variant(_source(), "avif-9000")

    def test_media_types(self):
        assert media_type_for("jpeg-40") == "image/jpeg"
        assert media_type_for("webp") == "image/webp"
        assert media_type_for("original") == "image/png"
        with pytest.raises(KeyError):
            media_type_for("nope")
