"""Crop geometry shared by the Yale B preparation and evaluation code."""

YALE_CROP_OFFSETS = range(8)
YALE_CROP_COUNT = len(YALE_CROP_OFFSETS) ** 2


def yale_crops(image):
    width, height = image.size
    crop_width = width - len(YALE_CROP_OFFSETS)
    crop_height = height - len(YALE_CROP_OFFSETS)
    if min(crop_width, crop_height) < 1:
        raise ValueError("The Yale B image is too small for the benchmark crops.")
    for y in YALE_CROP_OFFSETS:
        for x in YALE_CROP_OFFSETS:
            yield image.crop((x, y, x + crop_width, y + crop_height))
