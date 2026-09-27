"""Instagram Reels (1080x1920, 30s) room film from images/*.jpg.

Every photo is shown uncropped (letterboxed over a blurred, darkened copy of
itself), with slow zoom, varied transitions and Japanese/English captions.

Usage: python3 video/make_video.py [--preview]
Requires: pillow, numpy, imageio-ffmpeg, fonts-noto-cjk.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images")
OUT = os.path.join(ROOT, "video", "room_seasons_reel.mp4")
FONT_JP = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc"
FONT_EN = os.environ.get("FONT_EN", os.path.join(ROOT, "video", "fonts", "cormorant-garamond-500.woff"))

W, H, FPS, DURATION = 1080, 1920, 30, 30.0
GOLD = (201, 178, 140)
IVORY = (242, 236, 226)

# (file, English label, Japanese caption, zoom direction)
SCENES = [
    ("7C1A4141.jpg", "ONE ROOM, MANY FACES", "ひとつの部屋に、\nいくつもの表情。", +1),
    ("7C1A4142.jpg", "IN PRAISE OF SHADOWS", "灯りを落として、\n光と影を愉しむ", -1),
    ("7C1A4143.jpg", "MORNING LIGHT", "障子越しに、やわらかな朝の光", +1),
    ("7C1A4149.jpg", "OPENING", "障子をひらけば、庭の緑", -1),
    ("7C1A4150.jpg", "SEASONS", "窓辺に、季節がうつろう", +1),
    ("7C1A4159.jpg", "AFTERNOON", "木漏れ日が満ちる、昼下がり", -1),
    ("7C1A4161.jpg", "TIME", "ゆるやかに流れる時間も、\n贅沢のひとつ", +1),
    ("7C1A4162.jpg", "LIGHT  ·  SEASON  ·  TIME", "光と、季節と、時間と。\n訪れるたび、新しい表情に。", -1),
]
# transition into scene i+1
TRANSITIONS = ["black", "bloom", "shoji", "dissolve", "blur", "wipe", "dissolve"]
OVERLAP = 0.9  # seconds each transition lasts
SCENE_LEN = (DURATION + OVERLAP * (len(SCENES) - 1)) / len(SCENES)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def tracked_text(draw, xy_center, text, font, fill, tracking):
    """Draw one line centered at xy_center with extra letter spacing."""
    widths = [draw.textlength(c, font=font) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = xy_center[0] - total / 2
    for c, w in zip(text, widths):
        draw.text((x, xy_center[1]), c, font=font, fill=fill, anchor="lm")
        x += w + tracking


def caption_layer(en, jp, y_top):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_en = ImageFont.truetype(FONT_EN, 30)
    f_jp = ImageFont.truetype(FONT_JP, 50, index=0)
    y = y_top
    tracked_text(d, (W / 2, y), en, f_en, GOLD + (255,), 7)
    y += 42
    d.line([(W / 2 - 34, y), (W / 2 + 34, y)], fill=GOLD + (200,), width=1)
    y += 64
    for line in jp.split("\n"):
        tracked_text(d, (W / 2, y), line, f_jp, IVORY + (255,), 6)
        y += 82
    shadow = layer.filter(ImageFilter.GaussianBlur(6))
    sa = np.asarray(shadow)[..., 3].astype(np.float32) * 0.8
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    base.putalpha(Image.fromarray(sa.astype(np.uint8)))
    return Image.alpha_composite(base, layer)


class Scene:
    def __init__(self, fname, en, jp, zdir):
        src = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        self.zdir = zdir
        portrait = src.height > src.width
        # fit whole photo inside frame, leaving room for captions
        self.max_w = W * (0.90 if portrait else 0.97)
        scale = self.max_w / src.width
        self.fg_w, self.fg_h = self.max_w, src.height * scale
        pad = 14
        fg = src.resize((round(self.fg_w), round(self.fg_h)), Image.LANCZOS)
        framed = Image.new("RGBA", (fg.width + 2 * pad, fg.height + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(framed).rectangle(
            [2, 2, framed.width - 3, framed.height - 3], outline=GOLD + (120,), width=1)
        framed.paste(fg, (pad, pad))
        self.fg = framed
        self.pad = pad
        # centre of photo sits a bit above middle; captions go below
        self.cy = H * (0.43 if portrait else 0.42)
        bottom = self.cy + self.fg_h / 2
        self.caption = caption_layer(en, jp, bottom + 110)
        # ambient background: blurred, darkened, warm-toned copy
        bg = src.copy()
        bg.thumbnail((360, 360))
        r = max(W / bg.width, H / bg.height)
        bg = bg.resize((int(bg.width * r) + 2, int(bg.height * r) + 2), Image.BICUBIC)
        left, top = (bg.width - W) // 2, (bg.height - H) // 2
        bg = bg.crop((left, top, left + W, top + H)).filter(ImageFilter.GaussianBlur(40))
        a = np.asarray(bg).astype(np.float32) * 0.22
        a = a * np.array([1.02, 0.98, 0.92]) + np.array([8, 6, 4])
        yy, xx = np.mgrid[0:H, 0:W]
        vign = 1 - 0.55 * (((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
        a *= np.clip(vign, 0.3, 1)[..., None]
        self.bg = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")

    def render(self, t):
        """t = seconds since scene start."""
        p = t / SCENE_LEN
        z = 0.90 + 0.10 * (p if self.zdir > 0 else 1 - p)  # never exceeds full fit
        s = z  # output-per-source-pixel scale
        cx = W / 2
        cy = self.cy + 10 * (p - 0.5) * self.zdir
        x0 = cx - self.fg.width * s / 2
        y0 = cy - self.fg.height * s / 2
        inv = 1 / s
        fg = self.fg.transform((W, H), Image.AFFINE,
                               (inv, 0, -x0 * inv, 0, inv, -y0 * inv),
                               resample=Image.BICUBIC)
        frame = Image.alpha_composite(self.bg, fg)
        # caption fade/drift
        a = ease((t - 0.45) / 0.8) * (1 - ease((t - (SCENE_LEN - 1.05)) / 0.6))
        if a > 0:
            cap = self.caption
            dy = int(round(14 * (1 - ease((t - 0.45) / 1.0))))
            if dy:
                cap = cap.transform((W, H), Image.AFFINE, (1, 0, 0, 0, 1, -dy))
            ca = np.asarray(cap).copy()
            ca[..., 3] = (ca[..., 3].astype(np.float32) * a).astype(np.uint8)
            frame = Image.alpha_composite(frame, Image.fromarray(ca))
        return np.asarray(frame.convert("RGB")).astype(np.float32)


def transition(kind, a, b, p):
    """Blend frame a -> b, p in [0,1]."""
    e = ease(p)
    if kind == "dissolve":
        return a * (1 - e) + b * e
    if kind == "black":
        return a * max(0, 1 - 2 * e) ** 1.3 if e < 0.5 else b * (2 * e - 1) ** 0.8
    if kind == "bloom":  # warm light floods in, then settles
        glow = np.sin(np.pi * p)
        mix = a * (1 - e) + b * e
        warm = np.array([255, 238, 212], np.float32)
        return mix * (1 - 0.5 * glow) + warm * 0.5 * glow
    if kind == "blur":
        r = float(18 * np.sin(np.pi * p))
        ia = Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
        ib = Image.fromarray(b.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
        return np.asarray(ia, np.float32) * (1 - e) + np.asarray(ib, np.float32) * e
    xs = np.arange(W, dtype=np.float32)
    feather = 160
    if kind == "shoji":  # opens from the centre outwards like sliding doors
        half = e * (W / 2 + feather)
        m = np.clip((half - np.abs(xs - W / 2)) / feather, 0, 1)
    else:  # soft left-to-right wipe
        edge = e * (W + 2 * feather) - feather
        m = np.clip((edge - xs) / feather + 0.5, 0, 1)
    m = m[None, :, None]
    return a * (1 - m) + b * m


def main():
    preview = "--preview" in sys.argv
    scenes = [Scene(*s) for s in SCENES]
    starts = [i * (SCENE_LEN - OVERLAP) for i in range(len(scenes))]
    n_frames = int(DURATION * FPS)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
           "-shortest", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "128k", OUT]
    proc = None if preview else subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    for f in range(n_frames):
        if preview and f % 45:
            continue
        t = f / FPS
        active = [i for i, s0 in enumerate(starts) if s0 <= t < s0 + SCENE_LEN]
        if len(active) == 2:
            i, j = active
            p = (t - starts[j]) / OVERLAP
            img = transition(TRANSITIONS[i], scenes[i].render(t - starts[i]),
                             scenes[j].render(t - starts[j]), p)
        else:
            i = active[0] if active else len(scenes) - 1
            img = scenes[i].render(t - starts[i])
        # fade in from / out to black
        img = img * ease(t / 0.8) * (1 - ease((t - (DURATION - 1.2)) / 1.1))
        img += rng.normal(0, 1.6, (H, W, 1)).astype(np.float32)  # fine film grain
        out = np.clip(img, 0, 255).astype(np.uint8)
        if preview:
            Image.fromarray(out).save(os.path.join(os.environ.get("PREVIEW_DIR", "."), f"f{f:04d}.jpg"))
        else:
            proc.stdin.write(out.tobytes())
        if f % 90 == 0:
            print(f"frame {f}/{n_frames}", flush=True)
    if proc:
        proc.stdin.close()
        proc.wait()
        print("wrote", OUT)


if __name__ == "__main__":
    main()
