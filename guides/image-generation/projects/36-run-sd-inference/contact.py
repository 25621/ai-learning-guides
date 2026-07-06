"""Tiny helper: paste a row of PIL images into one labeled contact sheet."""

from PIL import Image, ImageDraw, ImageFont


def contact_sheet(images, labels, pad: int = 6, label_h: int = 22) -> Image.Image:
    w, h = images[0].size
    sheet = Image.new("RGB", (len(images) * (w + pad) - pad, h + label_h),
                      color=(252, 252, 251))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (w + pad)
        sheet.paste(img, (x, label_h))
        draw.text((x + 4, 3), str(label), fill=(11, 11, 11), font=font)
    return sheet


def stack_sheets(sheets, pad: int = 8) -> Image.Image:
    w = max(s.size[0] for s in sheets)
    h = sum(s.size[1] for s in sheets) + pad * (len(sheets) - 1)
    out = Image.new("RGB", (w, h), color=(252, 252, 251))
    y = 0
    for s in sheets:
        out.paste(s, (0, y))
        y += s.size[1] + pad
    return out
