#!/usr/bin/env python3
"""
RESONO v2 — Indigenous Language Robotic Archivist (WRO 2026)
============================================================
Upgraded headless-friendly GUI for Raspberry Pi 5.

Two display surfaces:
  1. Monitor (tkinter GUI via VNC/screenshare) — mode selection, camera, controls
  2. ST7789 1.9" SPI display (320×170) — eye animations, prompts, fire/water

Modes:
  • Prompt Mode   — select fire/water animation → display on ST7789 → record word
  • Object Mode   — MobileNetV3 detects object for 3s → prompt on ST7789 → record
  • Conversation  — Bangla prompt on ST7789 → record until 5s silence
"""

import os
import sys
import time
import math
import json
import sqlite3
import threading
import urllib.request
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# NCNN lightweight inference engine (optimised for ARM / Raspberry Pi)
import ncnn

# Audio capture & playback
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import pygame

# Hardware SPI & GPIO (ST7789)
try:
    import spidev
    import lgpio
    HARDWARE_SPI_AVAILABLE = True
except ImportError:
    HARDWARE_SPI_AVAILABLE = False
    print("[WARN] spidev/lgpio not found. Running in Display Simulation Mode.")

# ============================================================
# PATHS & DATABASE SETUP
# ============================================================
BASE_DIR = os.path.expanduser("~")
AUDIO_DIR = os.path.join(BASE_DIR, "resono_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "resono_archive.db")

pygame.mixer.init()

# ============================================================
# NCNN MODEL CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NCNN_PARAM  = os.path.join(SCRIPT_DIR, "resono_mobilenetv3_large_fp16.ncnn.param")
NCNN_BIN    = os.path.join(SCRIPT_DIR, "resono_mobilenetv3_large_fp16.ncnn.bin")
LABELS_FILE = os.path.join(SCRIPT_DIR, "labels.json")

NCNN_INPUT_BLOB  = "in0"
NCNN_OUTPUT_BLOB = "out0"

# Font for Bangla text rendering on ST7789
BANGLA_FONT_PATH = os.path.join(SCRIPT_DIR, "NotoSansBengali-Regular.ttf")

# Santali dictionary suggestions (keys match the training class labels)
SANTALI_OBJECT_MAP = {
    "WATER_BOTTLE": "Dak' Botol",
    "MUG":          "Dak' Bati",
    "PHONE":        "Phon",
    "WATCH":        "Ghori",
    "PEN":          "Kalam",
    "GLASSES":      "Chosma",
    "SPOON":        "Chamoch"
}


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_word TEXT NOT NULL,
            translation_en TEXT NOT NULL,
            translation_bn TEXT NOT NULL,
            category TEXT,
            mode TEXT,
            audio_file TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM archive")
    if cursor.fetchone()[0] == 0:
        samples = [
            ("Johar", "Hello / Greetings", "নমস্কার / হ্যালো", "Greeting", "Prompt Mode", ""),
            ("Dak'", "Water", "পানি / জল", "Everyday Object", "Object Mode", ""),
            ("Daka", "Cooked Rice / Meal", "ভাত / খাবার", "Food & Sustenance", "Prompt Mode", ""),
            ("Horo", "Tortoise / Turtle", "কচ্ছপ", "Folklore & Animals", "Conversation", "")
        ]
        cursor.executemany("""
            INSERT INTO archive (target_word, translation_en, translation_bn, category, mode, audio_file)
            VALUES (?, ?, ?, ?, ?, ?)
        """, samples)
    conn.commit()
    conn.close()

init_database()


# ============================================================
# NCNN MOBILENETV3-LARGE FP16 CLASSIFIER
# ============================================================
class PretrainedClassifier:
    """Runs object classification via the NCNN MobileNetV3-Large FP16 model
    exported from the Resono training pipeline.
    """

    # ImageNet normalisation (same values used during training)
    _MEAN = [123.675, 116.28, 103.53]        # RGB pixel means (0-255 scale)
    _NORM = [1/58.395, 1/57.12, 1/57.375]    # 1 / std  (0-255 scale)
    _INPUT_SIZE = 224

    def __init__(self):
        # --- Load class labels ---
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "classes" in data:
                    self.classes = list(data["classes"])
                elif isinstance(data, dict):
                    first_val = next(iter(data.values()))
                    if isinstance(first_val, int):
                        self.classes = [None] * len(data)
                        for name, idx in data.items():
                            self.classes[idx] = name
                    else:
                        self.classes = [data[str(i)] for i in range(len(data))]
                else:
                    self.classes = list(data)
        else:
            print(f"[WARN] {LABELS_FILE} not found — using hard-coded class order.")
            self.classes = ["glasses", "mug", "pen", "phone", "spoon", "watch", "water_bottle"]

        self.num_classes = len(self.classes)
        print(f"[AI] Classes ({self.num_classes}): {self.classes}")

        # --- Load NCNN network ---
        if not os.path.exists(NCNN_PARAM) or not os.path.exists(NCNN_BIN):
            raise FileNotFoundError(
                f"NCNN model files not found.\n"
                f"  Expected param : {NCNN_PARAM}\n"
                f"  Expected bin   : {NCNN_BIN}\n"
                f"Copy the exported model files into the same folder as main_v2.py."
            )

        self.net = ncnn.Net()
        self.net.opt.num_threads = 4
        self.net.load_param(NCNN_PARAM)
        self.net.load_model(NCNN_BIN)
        print(f"[AI] NCNN MobileNetV3-Large FP16 loaded from {SCRIPT_DIR}")

    @staticmethod
    def _softmax(x):
        """Numerically stable softmax."""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def classify_crop(self, crop_cv2):
        """Classify a BGR crop and return (clean_label, confidence, raw_label)."""
        resized = cv2.resize(crop_cv2, (self._INPUT_SIZE, self._INPUT_SIZE))
        mat_in = ncnn.Mat.from_pixels(
            resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self._INPUT_SIZE, self._INPUT_SIZE
        )
        mat_in.substract_mean_normalize(self._MEAN, self._NORM)
        ex = self.net.create_extractor()
        ex.input(NCNN_INPUT_BLOB, mat_in)
        ret, mat_out = ex.extract(NCNN_OUTPUT_BLOB)
        logits = np.array(mat_out).flatten()
        probs = self._softmax(logits)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        raw_label = self.classes[idx]
        clean_label = raw_label.upper()
        return clean_label, conf, raw_label


# ============================================================
# PROCEDURAL ANIMATION ENGINE (Fire & Water for ST7789)
# ============================================================
class AnimationEngine:
    """Generates procedural fire and water animation frames for the 320x170 ST7789."""

    WIDTH = 320
    HEIGHT = 170

    def __init__(self):
        # Pre-compute fire color palette (256 entries: black -> red -> orange -> yellow -> white)
        self.fire_palette = self._build_fire_palette()
        # Pre-compute water color palette (256 entries: deep blue -> cyan -> white)
        self.water_palette = self._build_water_palette()

    @staticmethod
    def _build_fire_palette():
        """Build a 256-color fire gradient: black -> dark red -> red -> orange -> yellow -> white."""
        palette = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.2:
                s = t / 0.2
                palette[i] = [int(s * 80), 0, 0]
            elif t < 0.4:
                s = (t - 0.2) / 0.2
                palette[i] = [80 + int(s * 175), 0, 0]
            elif t < 0.6:
                s = (t - 0.4) / 0.2
                palette[i] = [255, int(s * 160), 0]
            elif t < 0.8:
                s = (t - 0.6) / 0.2
                palette[i] = [255, 160 + int(s * 95), int(s * 40)]
            else:
                s = (t - 0.8) / 0.2
                palette[i] = [255, 255, 40 + int(s * 215)]
        return palette

    @staticmethod
    def _build_water_palette():
        """Build a 256-color water gradient: black -> deep blue -> blue -> cyan -> white."""
        palette = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.25:
                s = t / 0.25
                palette[i] = [0, 0, int(s * 60)]
            elif t < 0.5:
                s = (t - 0.25) / 0.25
                palette[i] = [0, int(s * 40), 60 + int(s * 120)]
            elif t < 0.75:
                s = (t - 0.5) / 0.25
                palette[i] = [int(s * 80), 40 + int(s * 140), 180 + int(s * 55)]
            else:
                s = (t - 0.75) / 0.25
                palette[i] = [80 + int(s * 175), 180 + int(s * 75), 235 + int(s * 20)]
        return palette

    def generate_fire_frame(self, heat_buffer):
        """Advance the doom-fire simulation by one step and return an RGB PIL image."""
        h, w = heat_buffer.shape
        # Seed bottom row with random hot values
        heat_buffer[h - 1, :] = np.random.randint(180, 256, size=w)
        for _ in range(w // 20):
            cx = np.random.randint(0, w)
            span = np.random.randint(4, 16)
            x0 = max(0, cx - span)
            x1 = min(w, cx + span)
            heat_buffer[h - 1, x0:x1] = np.random.randint(60, 140)

        # Propagate fire upward: each pixel = avg of neighbors below - random decay
        for y in range(0, h - 1):
            left  = np.roll(heat_buffer[y + 1], 1)
            right = np.roll(heat_buffer[y + 1], -1)
            below = heat_buffer[y + 1]
            if y + 2 < h:
                below2 = heat_buffer[y + 2]
            else:
                below2 = below
            avg = (left + right + below + below2) / 4.0
            decay = np.random.uniform(0.5, 2.5, size=w)
            heat_buffer[y] = np.clip(avg - decay, 0, 255)

        indices = np.clip(heat_buffer.astype(np.int32), 0, 255)
        rgb = self.fire_palette[indices]
        return Image.fromarray(rgb.astype(np.uint8), "RGB")

    def generate_water_frames(self, display_controller, duration=5.0):
        """Run a water ripple simulation on the ST7789 for `duration` seconds."""
        w, h = self.WIDTH, self.HEIGHT
        scale = 2
        sw, sh = w // scale, h // scale

        buf_curr = np.zeros((sh, sw), dtype=np.float64)
        buf_prev = np.zeros((sh, sw), dtype=np.float64)
        damping = 0.97

        t_start = time.time()
        frame_count = 0

        while time.time() - t_start < duration:
            if frame_count % 4 == 0:
                dx = np.random.randint(3, sw - 3)
                dy = np.random.randint(3, sh - 3)
                radius = np.random.randint(1, 3)
                for ry in range(-radius, radius + 1):
                    for rx in range(-radius, radius + 1):
                        if 0 <= dy+ry < sh and 0 <= dx+rx < sw:
                            buf_curr[dy+ry, dx+rx] = 255.0

            buf_new = np.zeros_like(buf_curr)
            buf_new[1:-1, 1:-1] = (
                (buf_curr[:-2, 1:-1] + buf_curr[2:, 1:-1] +
                 buf_curr[1:-1, :-2] + buf_curr[1:-1, 2:]) / 2.0
                - buf_prev[1:-1, 1:-1]
            ) * damping

            buf_prev = buf_curr.copy()
            buf_curr = buf_new

            intensity = np.clip(
                (buf_curr + 128).astype(np.int32), 0, 255
            )
            rgb_small = self.water_palette[intensity]
            img_small = Image.fromarray(rgb_small.astype(np.uint8), "RGB")
            img = img_small.resize((w, h), Image.NEAREST)

            display_controller.display_image(img)
            frame_count += 1
            time.sleep(0.05)

    def generate_fire_animation(self, display_controller, duration=5.0):
        """Run the doom-fire animation on the ST7789 for `duration` seconds."""
        scale = 2
        sw, sh = self.WIDTH // scale, self.HEIGHT // scale
        heat = np.zeros((sh, sw), dtype=np.float64)

        t_start = time.time()
        while time.time() - t_start < duration:
            frame_img_small = self.generate_fire_frame(heat)
            frame_img = frame_img_small.resize(
                (self.WIDTH, self.HEIGHT), Image.NEAREST
            )
            display_controller.display_image(frame_img)
            time.sleep(0.05)


# ============================================================
# ST7789 EYES DISPLAY ENGINE (Enhanced with pause/resume & font auto-handling)
# ============================================================
class EyesDisplayController:
    def __init__(self, rotation=180):
        self.width = 320
        self.height = 170
        self.x_offset = 0
        self.y_offset = 35
        self.dc_pin = 25
        self.rst_pin = 24
        self.bl_pin = 18
        self.rotation = rotation  # 180 = rotate display 180 deg for robot mounting
        self.state = "normal"
        self.running = True

        # Pause/resume mechanism
        self._paused = threading.Event()  # SET = paused, CLEAR = running

        self.active = False

        if HARDWARE_SPI_AVAILABLE:
            try:
                self.spi = spidev.SpiDev()
                self.spi.open(0, 0)
                self.spi.max_speed_hz = 40000000
                self.spi.mode = 0

                self.chip = lgpio.gpiochip_open(0)
                lgpio.gpio_claim_output(self.chip, self.dc_pin)
                lgpio.gpio_claim_output(self.chip, self.rst_pin)
                lgpio.gpio_claim_output(self.chip, self.bl_pin)
                lgpio.gpio_write(self.chip, self.bl_pin, 1)

                self.active = True
                self.init_display()
            except Exception as e:
                print(f"[DISPLAY ERROR] {e}")
                self.active = False
        else:
            self.active = False

        # Load fonts for text rendering
        self._load_fonts()

    def _ensure_bangla_font(self):
        """Check local folder, system font paths, or download NotoSansBengali if missing."""
        if os.path.exists(BANGLA_FONT_PATH) and os.path.getsize(BANGLA_FONT_PATH) > 10000:
            return BANGLA_FONT_PATH

        # Check common Linux/Debian/RPi font directories
        candidate_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansBengali-Regular.otf",
            "/usr/share/fonts/truetype/fonts-beng-extra/kalpurush.ttf",
            "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
            "C:/Windows/Fonts/Vrinda.ttf",
            "C:/Windows/Fonts/Nirmala.ttf",
            "C:/Windows/Fonts/NirmalaB.ttf"
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                print(f"[DISPLAY] Found system Bengali font: {path}")
                return path

        # Auto-download from Google Fonts repository
        download_urls = [
            "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf",
            "https://github.com/notofonts/bengali/raw/main/fonts/NotoSansBengali-Regular.ttf"
        ]
        for url in download_urls:
            try:
                print(f"[DISPLAY] Auto-downloading Bengali font from {url}...")
                urllib.request.urlretrieve(url, BANGLA_FONT_PATH)
                if os.path.exists(BANGLA_FONT_PATH) and os.path.getsize(BANGLA_FONT_PATH) > 10000:
                    print(f"[DISPLAY] Successfully downloaded NotoSansBengali-Regular.ttf!")
                    return BANGLA_FONT_PATH
            except Exception as e:
                print(f"[DISPLAY WARN] Download failed from {url}: {e}")

        return None

    def _load_fonts(self):
        """Load fonts for English and Bangla text rendering."""
        self.font_en = None
        self.font_en_sm = None
        self.font_title = None
        self.font_bn = None
        self.has_bangla_font = False

        # Load English fonts
        for font_name in ["DejaVuSans-Bold.ttf", "FreeSansBold.ttf", "arial.ttf", "segoeui.ttf"]:
            try:
                self.font_en = ImageFont.truetype(font_name, 16)
                self.font_en_sm = ImageFont.truetype(font_name, 12)
                self.font_title = ImageFont.truetype(font_name, 20)
                break
            except (IOError, OSError):
                continue

        if self.font_en is None:
            self.font_en = ImageFont.load_default()
            self.font_en_sm = ImageFont.load_default()
            self.font_title = ImageFont.load_default()

        # Load or download Bangla font
        found_font = self._ensure_bangla_font()
        if found_font:
            try:
                self.font_bn = ImageFont.truetype(found_font, 18)
                self.font_title = ImageFont.truetype(found_font, 20)
                self.font_en_sm = ImageFont.truetype(found_font, 12)
                self.has_bangla_font = True
                print(f"[DISPLAY] Bangla font active: {found_font}")
            except Exception as e:
                print(f"[DISPLAY ERROR] Failed loading Bangla font: {e}")
                self.font_bn = self.font_en
                self.has_bangla_font = False
        else:
            print("[DISPLAY WARN] No Bangla font available. Will use English fallback prompt.")
            self.font_bn = self.font_en
            self.has_bangla_font = False

    def command(self, val):
        if self.active:
            lgpio.gpio_write(self.chip, self.dc_pin, 0)
            self.spi.writebytes([val])

    def data(self, val):
        if self.active:
            lgpio.gpio_write(self.chip, self.dc_pin, 1)
            self.spi.writebytes([val] if isinstance(val, int) else val)

    def init_display(self):
        lgpio.gpio_write(self.chip, self.rst_pin, 1); time.sleep(0.05)
        lgpio.gpio_write(self.chip, self.rst_pin, 0); time.sleep(0.05)
        lgpio.gpio_write(self.chip, self.rst_pin, 1); time.sleep(0.05)
        self.command(0x01); time.sleep(0.15)
        self.command(0x11); time.sleep(0.12)
        self.command(0x3A); self.data(0x55)
        self.command(0x36); self.data(0x60)
        self.command(0x21)
        self.command(0x13)
        self.command(0x29); time.sleep(0.05)

    def display_image(self, image):
        """Send a PIL RGB image to the ST7789 display."""
        if not self.active:
            return
        self.command(0x2A)
        self.data([(self.x_offset >> 8) & 0xFF, self.x_offset & 0xFF,
                   ((self.x_offset + self.width - 1) >> 8) & 0xFF,
                   (self.x_offset + self.width - 1) & 0xFF])
        self.command(0x2B)
        self.data([(self.y_offset >> 8) & 0xFF, self.y_offset & 0xFF,
                   ((self.y_offset + self.height - 1) >> 8) & 0xFF,
                   (self.y_offset + self.height - 1) & 0xFF])
        self.command(0x2C)

        # Convert to RGB565 using numpy vectorization
        img_arr = np.array(image)
        if self.rotation == 180:
            img_arr = img_arr[::-1, ::-1, :]

        r = (img_arr[:, :, 0].astype(np.uint16) & 0xF8) << 8
        g = (img_arr[:, :, 1].astype(np.uint16) & 0xFC) << 3
        b = img_arr[:, :, 2].astype(np.uint16) >> 3
        rgb565 = r | g | b
        high = ((rgb565 >> 8) & 0xFF).astype(np.uint8)
        low = (rgb565 & 0xFF).astype(np.uint8)
        buffer = np.stack([high, low], axis=-1).flatten().tobytes()

        lgpio.gpio_write(self.chip, self.dc_pin, 1)
        for i in range(0, len(buffer), 4096):
            self.spi.writebytes(list(buffer[i:i + 4096]))

    def display_prompt_card(self, title, bangla_text, english_text, hint="🎤 Speak now..."):
        """Render a high-contrast, modern prompt card on the ST7789 display.
        Handles missing Bangla font gracefully to completely avoid square boxes/tofu.
        """
        canvas = Image.new("RGB", (self.width, self.height), (11, 17, 32))
        draw = ImageDraw.Draw(canvas)

        # Draw rounded card container
        draw.rounded_rectangle(
            [6, 6, self.width - 6, self.height - 6],
            radius=8, fill=(30, 41, 59), outline=(56, 189, 248), width=2
        )

        # 1. Title
        if title:
            bbox = draw.textbbox((0, 0), title, font=self.font_title)
            tw = bbox[2] - bbox[0]
            draw.text(((self.width - tw) // 2, 16), title, fill=(248, 113, 113), font=self.font_title)

        # 2. Main Question / Prompt
        main_prompt = bangla_text if self.has_bangla_font else english_text
        bbox = draw.textbbox((0, 0), main_prompt, font=self.font_bn)
        tw = bbox[2] - bbox[0]
        # Word wrap if too long
        if tw > self.width - 24:
            words = main_prompt.split()
            lines = []
            curr = ""
            for w in words:
                t = f"{curr} {w}".strip() if curr else w
                if draw.textbbox((0, 0), t, font=self.font_bn)[2] > self.width - 24:
                    if curr: lines.append(curr)
                    curr = w
                else:
                    curr = t
            if curr: lines.append(curr)
            y = 56
            for l in lines:
                b = draw.textbbox((0, 0), l, font=self.font_bn)
                draw.text(((self.width - (b[2]-b[0])) // 2, y), l, fill=(255, 255, 255), font=self.font_bn)
                y += 24
        else:
            draw.text(((self.width - tw) // 2, 60), main_prompt, fill=(255, 255, 255), font=self.font_bn)

        # 3. Subtitle
        if self.has_bangla_font and english_text:
            bbox = draw.textbbox((0, 0), english_text, font=self.font_en_sm)
            tw = bbox[2] - bbox[0]
            draw.text(((self.width - tw) // 2, 98), english_text, fill=(148, 163, 184), font=self.font_en_sm)

        # 4. Action Hint
        if hint:
            bbox = draw.textbbox((0, 0), hint, font=self.font_en_sm)
            tw = bbox[2] - bbox[0]
            draw.text(((self.width - tw) // 2, 130), hint, fill=(56, 189, 248), font=self.font_en_sm)

        self.display_image(canvas)

    def display_text(self, lines, color=(255, 255, 255), bg_color=(11, 17, 32)):
        """Render centered English text lines on the ST7789."""
        canvas = Image.new("RGB", (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(canvas)

        total_height = len(lines) * 24
        y_start = max(0, (self.height - total_height) // 2)

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=self.font_en)
            tw = bbox[2] - bbox[0]
            x = max(0, (self.width - tw) // 2)
            y = y_start + i * 24
            draw.text((x, y), line, fill=color, font=self.font_en)

        self.display_image(canvas)

    def display_bangla(self, text, color=(255, 255, 255), bg_color=(11, 17, 32)):
        """Render Bangla text centered, or fall back to English if font unavailable."""
        if not self.has_bangla_font:
            self.display_text([text], color=color, bg_color=bg_color)
            return

        canvas = Image.new("RGB", (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(canvas)

        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=self.font_bn)
            tw = bbox[2] - bbox[0]
            if tw > self.width - 20:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)

        line_height = 28
        total_height = len(lines) * line_height
        y_start = max(0, (self.height - total_height) // 2)

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=self.font_bn)
            tw = bbox[2] - bbox[0]
            x = max(0, (self.width - tw) // 2)
            y = y_start + i * line_height
            draw.text((x, y), line, fill=color, font=self.font_bn)

        self.display_image(canvas)

    # --- Pause/Resume for eye animation ---
    def pause(self):
        """Pause the eye animation loop so the display can be used for content."""
        self._paused.set()
        time.sleep(0.15)
        print("[EYES] Animation paused")

    def resume(self):
        """Resume the eye animation loop."""
        self._paused.clear()
        print("[EYES] Animation resumed")

    def is_paused(self):
        return self._paused.is_set()

    # --- Eye rendering ---
    def create_eye(self, width, height, angle, color):
        pad = 10
        canvas = Image.new("RGBA", (width + pad * 2, max(height + pad * 2, pad * 2 + 6)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.ellipse([pad, pad, pad + width, pad + height], fill=color)
        return canvas.rotate(angle, resample=Image.BICUBIC, expand=True)

    def render_eyes(self, look=0.0, blink=0.0):
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 255))
        base_w, base_h = 68, 92
        color = (255, 255, 255, 255)

        if self.state == "detect":
            base_w, base_h = 76, 98
            color = (56, 189, 248, 255)
        elif self.state == "listening":
            color = (239, 68, 68, 255)

        cur_h = int(4 + (base_h - 4) * (1.0 - blink))
        l_eye = self.create_eye(base_w, cur_h, angle=-14, color=color)
        r_eye = self.create_eye(base_w, cur_h, angle=14, color=color)

        cy = self.height // 2
        lx = 105 + int(16 * look)
        rx = 215 + int(16 * look)

        canvas.paste(l_eye, (lx - l_eye.width // 2, cy - l_eye.height // 2), l_eye)
        canvas.paste(r_eye, (rx - r_eye.width // 2, cy - r_eye.height // 2), r_eye)
        self.display_image(canvas.convert("RGB"))

    def animation_loop(self):
        """Main eye animation loop — pauses when _paused event is set."""
        if not self.active:
            return
        while self.running:
            if self._paused.is_set():
                time.sleep(0.1)
                continue

            self.render_eyes(look=-0.8); time.sleep(0.4)
            if self._paused.is_set(): continue
            self.render_eyes(look=0.0); time.sleep(0.5)
            if self._paused.is_set(): continue
            self.render_eyes(look=0.0, blink=1.0); time.sleep(0.08)
            if self._paused.is_set(): continue
            self.render_eyes(look=0.0, blink=0.0); time.sleep(0.7)
            if self._paused.is_set(): continue
            self.render_eyes(look=0.8); time.sleep(0.4)
            if self._paused.is_set(): continue
            self.render_eyes(look=0.0); time.sleep(0.5)

    def close(self):
        self.running = False
        if self.active:
            try:
                lgpio.gpio_write(self.chip, self.bl_pin, 0)
                self.spi.close()
                lgpio.gpiochip_close(self.chip)
            except:
                pass


# ============================================================
# AUDIO RECORDER (with silence detection)
# ============================================================
class AudioRecorder:
    """Handles audio recording with manual stop and auto-stop on silence."""

    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_frames = []
        self.latest_path = None
        # Silence detection
        self.silence_threshold = 200  # RMS threshold for int16 audio
        self.silence_duration = 0.0
        self.max_silence = 5.0  # seconds
        self.auto_stop_callback = None
        self._auto_stop_enabled = False

    def start_recording(self, auto_stop=False, on_auto_stop=None):
        """Start recording audio. If auto_stop=True, stops after 5s of silence."""
        if self.is_recording:
            return
        self.is_recording = True
        self.audio_frames = []
        self.silence_duration = 0.0
        self._auto_stop_enabled = auto_stop
        self.auto_stop_callback = on_auto_stop

        def _record():
            try:
                with sd.InputStream(samplerate=self.sample_rate, channels=1,
                                    dtype='int16', blocksize=1024,
                                    callback=self._audio_callback):
                    while self.is_recording:
                        sd.sleep(100)
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")
                self.is_recording = False

        threading.Thread(target=_record, daemon=True).start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream."""
        self.audio_frames.append(indata.copy())

        if self._auto_stop_enabled:
            rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
            chunk_duration = frames / self.sample_rate

            if rms < self.silence_threshold:
                self.silence_duration += chunk_duration
                if self.silence_duration >= self.max_silence:
                    self.is_recording = False
                    if self.auto_stop_callback:
                        self.auto_stop_callback()
            else:
                self.silence_duration = 0.0

    def stop_recording(self, mode_prefix="rec"):
        """Stop recording and save to WAV. Returns the file path."""
        self.is_recording = False
        time.sleep(0.15)

        if self.audio_frames:
            data = np.concatenate(self.audio_frames, axis=0)
            path = os.path.join(AUDIO_DIR, f"{mode_prefix}_{int(time.time())}.wav")
            wav_write(path, self.sample_rate, data)
            self.latest_path = path
            print(f"[AUDIO] Saved: {path}")
            return path
        return None

    def play_latest(self):
        """Play the most recently recorded audio."""
        if self.latest_path and os.path.exists(self.latest_path):
            pygame.mixer.music.load(self.latest_path)
            pygame.mixer.music.play()
            return True
        return False

    def get_silence_progress(self):
        """Return current silence duration (for UI display)."""
        return self.silence_duration


# ============================================================
# ROBOT TEXT-TO-SPEECH (TTS) ENGINE
# ============================================================
class TTSController:
    """Provides natural robot voice synthesis with offline caching and multi-tier fallback.

    Tier 1: Local MP3 file cache in ~/resono_audio/tts_cache/ (instant playback, 0 network lag)
    Tier 2: Google Translate TTS endpoint via urllib (natural neural voice)
    Tier 3: Offline OS fallback (espeak/espeak-ng on Linux/RPi, SAPI on Windows)
    """

    def __init__(self):
        self.cache_dir = os.path.join(AUDIO_DIR, "tts_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _get_cache_path(self, text):
        """Generate a deterministic filename in cache for the text."""
        import hashlib
        safe_hash = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:10]
        safe_prefix = "".join(c for c in text[:16] if c.isalnum() or c in (" ", "_")).rstrip().replace(" ", "_").lower()
        return os.path.join(self.cache_dir, f"tts_{safe_prefix}_{safe_hash}.mp3")

    def _synthesize_online(self, text, dest_path):
        """Fetch TTS MP3 audio stream from Google TTS endpoint."""
        import urllib.parse
        clean_text = text.strip()
        url = "https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=en&q=" + urllib.parse.quote(clean_text)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux armv7l)"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = resp.read()
            if len(data) > 500:
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True
        return False

    def _fallback_speak_os(self, text):
        """Offline fallback using OS-level TTS (espeak on Linux/Pi, SAPI on Windows)."""
        import subprocess
        try:
            if sys.platform.startswith("linux"):
                for cmd in ["espeak-ng", "espeak", "spd-say"]:
                    if subprocess.run(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                        subprocess.run([cmd, text], check=False)
                        return
            elif sys.platform == "win32":
                ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False)
        except Exception as e:
            print(f"[TTS OS FALLBACK ERROR] {e}")

    def speak(self, text, block=False, on_complete=None):
        """Speak text in a non-blocking background thread with instant caching."""
        def _worker():
            with self._lock:
                cache_file = self._get_cache_path(text)
                played = False

                # 1. Try local cache or online synthesis
                try:
                    if not (os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000):
                        self._synthesize_online(text, cache_file)

                    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
                        pygame.mixer.music.load(cache_file)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.05)
                        played = True
                except Exception as e:
                    print(f"[TTS WARN] Online/Cache playback error: {e}")

                # 2. Offline OS Fallback
                if not played:
                    self._fallback_speak_os(text)

                if on_complete:
                    on_complete()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        if block:
            t.join()


# ============================================================
# MAIN RESONO v2 GUI
# ============================================================
class ResonoApp:
    # Color scheme
    BG_DARK     = "#0b1120"
    BG_CARD     = "#1e293b"
    BG_INPUT    = "#0f172a"
    BG_FIELD    = "#334155"
    FG_PRIMARY  = "#f8fafc"
    FG_MUTED    = "#94a3b8"
    ACCENT_CYAN = "#38bdf8"
    ACCENT_GREEN= "#10b981"
    ACCENT_RED  = "#ef4444"
    ACCENT_BLUE = "#0284c7"
    ACCENT_AMBER= "#f59e0b"

    def __init__(self, root):
        self.root = root
        self.root.title("RESONO v2 — Indigenous Language Robotic Archivist")
        self.root.geometry("900x620")
        self.root.configure(bg=self.BG_DARK)
        self.root.resizable(True, True)

        # Core systems
        self.classifier = PretrainedClassifier()
        self.eyes = EyesDisplayController(rotation=180)
        self.animations = AnimationEngine()
        self.recorder = AudioRecorder()
        self.tts = TTSController()

        # Start eye animation thread
        threading.Thread(target=self.eyes.animation_loop, daemon=True).start()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.camera_running = False

        # Object mode tracking
        self.sustained_class = None
        self.sustained_time = 0.0
        self.sustained_threshold = 3.0  # seconds
        self.last_frame_time = time.time()
        self.object_triggered = False

        # Current frame reference (to prevent GC)
        self._current_frame = None

        # Build UI
        self._build_header()
        self._content_area = tk.Frame(self.root, bg=self.BG_DARK)
        self._content_area.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        # Track active frame
        self._active_frame = None

        # Show mode selection
        self.show_mode_selection()

    # ---------------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------------
    def _build_header(self):
        head = tk.Frame(self.root, bg=self.BG_CARD, height=54, padx=16)
        head.pack(side=tk.TOP, fill=tk.X)

        tk.Label(head, text="RESONO", font=("Segoe UI", 16, "bold"),
                 fg=self.ACCENT_CYAN, bg=self.BG_CARD).pack(side=tk.LEFT, pady=10)

        badge = tk.Label(head, text=" PILOT: SANTALI ",
                         font=("Segoe UI", 8, "bold"),
                         fg=self.ACCENT_GREEN, bg="#064e3b", padx=6, pady=2)
        badge.pack(side=tk.LEFT, padx=14)

        self.db_counter_lbl = tk.Label(head, text="Words: 0",
                                        font=("Segoe UI", 10),
                                        fg=self.FG_MUTED, bg=self.BG_CARD)
        self.db_counter_lbl.pack(side=tk.RIGHT, pady=10)
        self._update_db_count()

    def _update_db_count(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM archive")
            count = c.fetchone()[0]
            conn.close()
            self.db_counter_lbl.config(text=f"Words: {count}")
        except:
            pass

    # ---------------------------------------------------------------
    # FRAME SWAPPING
    # ---------------------------------------------------------------
    def _clear_content(self):
        """Remove the current active frame from the content area."""
        if self._active_frame:
            self._active_frame.destroy()
            self._active_frame = None
        self.camera_running = False

    def _make_exit_button(self, parent):
        """Create a styled exit button that returns to mode selection."""
        btn = tk.Button(parent, text="✕ EXIT", font=("Segoe UI", 10, "bold"),
                        bg="#475569", fg="white", activebackground="#64748b",
                        relief=tk.FLAT, padx=12, pady=4,
                        command=self._exit_mode)
        return btn

    def _exit_mode(self):
        """Exit current mode, resume eyes, return to mode selection."""
        self.camera_running = False
        self.recorder.is_recording = False
        self.eyes.state = "normal"
        self.eyes.resume()
        self.object_triggered = False
        self.sustained_class = None
        self.sustained_time = 0.0
        self.show_mode_selection()

    # ---------------------------------------------------------------
    # MODE SELECTION PANEL
    # ---------------------------------------------------------------
    def show_mode_selection(self):
        self._clear_content()
        frame = tk.Frame(self._content_area, bg=self.BG_DARK)
        frame.pack(fill=tk.BOTH, expand=True)
        self._active_frame = frame

        tk.Label(frame, text="SELECT OPERATING MODE",
                 font=("Segoe UI", 18, "bold"),
                 fg=self.FG_PRIMARY, bg=self.BG_DARK).pack(pady=(28, 20))

        # Fixed width container to prevent any horizontal resizing/jitter
        cards = tk.Frame(frame, bg=self.BG_DARK, width=540)
        cards.pack(expand=True, pady=(0, 20))

        self._make_mode_card(
            cards,
            icon="🎯", title="PROMPT MODE",
            subtitle="Visual prompt → Record pronunciation",
            desc="Show fire/water animation, ask for the word",
            color=self.ACCENT_CYAN,
            command=self.show_prompt_mode
        )

        self._make_mode_card(
            cards,
            icon="📷", title="OBJECT MODE",
            subtitle="Detect object → Name → Record",
            desc="Camera identifies objects, record their names",
            color=self.ACCENT_GREEN,
            command=self.show_object_mode
        )

        self._make_mode_card(
            cards,
            icon="💬", title="CONVERSATION MODE",
            subtitle="Free narrative recording",
            desc="Record stories, auto-stops after 5s silence",
            color=self.ACCENT_AMBER,
            command=self.show_conversation_mode
        )

    def _make_mode_card(self, parent, icon, title, subtitle, desc, color, command):
        """Create a rock-solid, flutter-free clickable mode selection card."""
        card = tk.Frame(parent, bg=self.BG_CARD, cursor="hand2",
                        padx=20, pady=12, bd=0, highlightthickness=2,
                        highlightbackground="#334155", highlightcolor=color,
                        width=520)
        card.pack(fill=tk.X, pady=8)

        top = tk.Frame(card, bg=self.BG_CARD)
        top.pack(fill=tk.X)
        i_lbl = tk.Label(top, text=icon, font=("Segoe UI", 22),
                         bg=self.BG_CARD, fg=color)
        i_lbl.pack(side=tk.LEFT)
        t_lbl = tk.Label(top, text=f"  {title}", font=("Segoe UI", 14, "bold"),
                         bg=self.BG_CARD, fg=color)
        t_lbl.pack(side=tk.LEFT)

        s_lbl = tk.Label(card, text=subtitle, font=("Segoe UI", 10),
                         bg=self.BG_CARD, fg=self.FG_PRIMARY)
        s_lbl.pack(anchor=tk.W, padx=40)
        d_lbl = tk.Label(card, text=desc, font=("Segoe UI", 9),
                         bg=self.BG_CARD, fg=self.FG_MUTED)
        d_lbl.pack(anchor=tk.W, padx=40)

        all_widgets = [card, top, i_lbl, t_lbl, s_lbl, d_lbl]

        def _on_click(e):
            command()

        def _on_enter(e):
            card.config(highlightbackground=color, bg="#24334a")
            for w in all_widgets:
                try:
                    w.config(bg="#24334a")
                except Exception:
                    pass

        def _on_leave(e):
            # Check pointer coordinates to avoid oscillation when moving across child labels
            try:
                x, y = card.winfo_pointerxy()
                cx = card.winfo_rootx()
                cy = card.winfo_rooty()
                cw = card.winfo_width()
                ch = card.winfo_height()
                if cx <= x < cx + cw and cy <= y < cy + ch:
                    return  # Cursor is still inside the card, ignore sub-widget leave event
            except Exception:
                pass
            card.config(highlightbackground="#334155", bg=self.BG_CARD)
            for w in all_widgets:
                try:
                    w.config(bg=self.BG_CARD)
                except Exception:
                    pass

        for w in all_widgets:
            w.bind("<Button-1>", _on_click)
            w.bind("<Enter>", _on_enter)
            w.bind("<Leave>", _on_leave)

        return card

    # ---------------------------------------------------------------
    # PROMPT MODE
    # ---------------------------------------------------------------
    def show_prompt_mode(self):
        self._clear_content()
        frame = tk.Frame(self._content_area, bg=self.BG_DARK)
        frame.pack(fill=tk.BOTH, expand=True)
        self._active_frame = frame

        hdr = tk.Frame(frame, bg=self.BG_CARD, padx=12, pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="🎯  PROMPT MODE", font=("Segoe UI", 14, "bold"),
                 fg=self.ACCENT_CYAN, bg=self.BG_CARD).pack(side=tk.LEFT)
        self._make_exit_button(hdr).pack(side=tk.RIGHT)

        anim_frame = tk.Frame(frame, bg=self.BG_DARK, pady=16)
        anim_frame.pack(fill=tk.X)

        tk.Label(anim_frame, text="Select animation to display on robot:",
                 font=("Segoe UI", 11), fg=self.FG_PRIMARY,
                 bg=self.BG_DARK).pack(anchor=tk.W, padx=8, pady=(0, 10))

        btn_row = tk.Frame(anim_frame, bg=self.BG_DARK)
        btn_row.pack()

        tk.Button(btn_row, text="🔥  FIRE", font=("Segoe UI", 14, "bold"),
                  bg="#991b1b", fg="white", activebackground="#dc2626",
                  width=14, height=2, relief=tk.FLAT,
                  command=lambda: self._prompt_play_animation("fire")
                  ).pack(side=tk.LEFT, padx=12)

        tk.Button(btn_row, text="💧  WATER", font=("Segoe UI", 14, "bold"),
                  bg="#1e3a5f", fg="white", activebackground="#2563eb",
                  width=14, height=2, relief=tk.FLAT,
                  command=lambda: self._prompt_play_animation("water")
                  ).pack(side=tk.LEFT, padx=12)

        self.prompt_status = tk.Label(frame, text="Select an animation above to begin",
                                       font=("Segoe UI", 11),
                                       fg=self.FG_MUTED, bg=self.BG_DARK)
        self.prompt_status.pack(pady=12)

        ctrl = tk.Frame(frame, bg=self.BG_FIELD, padx=12, pady=10)
        ctrl.pack(fill=tk.X, padx=8)

        self.prompt_rec_btn = tk.Button(
            ctrl, text="🎤 RECORD", font=("Segoe UI", 11, "bold"),
            bg=self.ACCENT_RED, fg="white", state=tk.DISABLED,
            command=self._prompt_toggle_record)
        self.prompt_rec_btn.pack(side=tk.LEFT, padx=6)

        self.prompt_play_btn = tk.Button(
            ctrl, text="▶ PLAY", font=("Segoe UI", 11, "bold"),
            bg="#475569", fg="white", state=tk.DISABLED,
            command=self._prompt_play_audio)
        self.prompt_play_btn.pack(side=tk.LEFT, padx=6)

        self.prompt_save_btn = tk.Button(
            ctrl, text="💾 SAVE", font=("Segoe UI", 11, "bold"),
            bg=self.ACCENT_GREEN, fg="white", state=tk.DISABLED,
            command=self._prompt_save)
        self.prompt_save_btn.pack(side=tk.LEFT, padx=6)

        entry_frame = tk.Frame(frame, bg=self.BG_DARK, pady=8)
        entry_frame.pack(fill=tk.X, padx=8)

        tk.Label(entry_frame, text="Santali Word:", font=("Segoe UI", 10, "bold"),
                 fg=self.ACCENT_CYAN, bg=self.BG_DARK).pack(side=tk.LEFT)
        self.prompt_word_entry = tk.Entry(entry_frame, width=30,
                                          bg=self.BG_FIELD, fg="white",
                                          font=("Segoe UI", 11),
                                          insertbackground="white")
        self.prompt_word_entry.pack(side=tk.LEFT, padx=8)

        self._prompt_current_anim = None

    def _prompt_play_animation(self, anim_type):
        """Play fire or water animation on ST7789, then show question."""
        self._prompt_current_anim = anim_type
        self.prompt_status.config(
            text=f"Playing {anim_type} animation on display...",
            fg=self.ACCENT_AMBER)

        def _run():
            self.eyes.pause()

            if anim_type == "fire":
                self.animations.generate_fire_animation(self.eyes, duration=5.0)
                card_title = "🔥 FIRE / আগুন"
            else:
                self.animations.generate_water_frames(self.eyes, duration=5.0)
                card_title = "💧 WATER / পানি"

            # Display high-contrast prompt card with Bangla & English
            self.eyes.display_prompt_card(
                title=card_title,
                bangla_text="এটাকে তোমার ভাষায় কী বলে?",
                english_text="What is this called in your language?",
                hint="🎤 Speak now..."
            )

            # Robot voice speaks the prompt question
            self.tts.speak("What is this called in your language?")

            self.root.after(0, self._prompt_animation_done)

        threading.Thread(target=_run, daemon=True).start()

    def _prompt_animation_done(self):
        """Called after animation finishes — enable recording."""
        self.prompt_status.config(
            text="Say the word in your language now!",
            fg=self.ACCENT_GREEN)
        self.prompt_rec_btn.config(state=tk.NORMAL)

    def _prompt_toggle_record(self):
        if not self.recorder.is_recording:
            self.recorder.start_recording()
            self.eyes.state = "listening"
            self.prompt_rec_btn.config(text="⏹ STOP", bg="#b91c1c")
            self.prompt_status.config(text="🔴 Recording...", fg=self.ACCENT_RED)
        else:
            path = self.recorder.stop_recording(
                mode_prefix=f"prompt_{self._prompt_current_anim or 'unknown'}")
            self.eyes.state = "normal"
            self.eyes.resume()
            self.prompt_rec_btn.config(text="🎤 RECORD", bg=self.ACCENT_RED)
            self.prompt_play_btn.config(state=tk.NORMAL)
            self.prompt_save_btn.config(state=tk.NORMAL)
            if path:
                self.prompt_status.config(
                    text="Recorded! Enter the word and save.",
                    fg=self.ACCENT_GREEN)
            else:
                self.prompt_status.config(
                    text="No audio captured. Try again.",
                    fg=self.ACCENT_RED)

    def _prompt_play_audio(self):
        if not self.recorder.play_latest():
            messagebox.showinfo("Audio", "No recording to play.")

    def _prompt_save(self):
        word = self.prompt_word_entry.get().strip()
        if not word:
            messagebox.showwarning("Missing", "Enter the Santali word first.")
            return
        concept = self._prompt_current_anim.title() if self._prompt_current_anim else "Unknown"
        self._save_to_db(word, concept, "", "Prompt Mode", "Visual Concept")
        self.prompt_status.config(
            text=f"Saved '{word}' to archive!", fg=self.ACCENT_GREEN)
        self.prompt_rec_btn.config(state=tk.DISABLED)
        self.prompt_play_btn.config(state=tk.DISABLED)
        self.prompt_save_btn.config(state=tk.DISABLED)

    # ---------------------------------------------------------------
    # OBJECT MODE
    # ---------------------------------------------------------------
    def show_object_mode(self):
        self._clear_content()
        frame = tk.Frame(self._content_area, bg=self.BG_DARK)
        frame.pack(fill=tk.BOTH, expand=True)
        self._active_frame = frame

        hdr = tk.Frame(frame, bg=self.BG_CARD, padx=12, pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="📷  OBJECT MODE", font=("Segoe UI", 14, "bold"),
                 fg=self.ACCENT_GREEN, bg=self.BG_CARD).pack(side=tk.LEFT)
        self._make_exit_button(hdr).pack(side=tk.RIGHT)

        body = tk.Frame(frame, bg=self.BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, pady=8)

        cam_box = tk.Frame(body, bg=self.BG_INPUT, padx=4, pady=4)
        cam_box.pack(side=tk.LEFT, padx=(8, 8))

        tk.Label(cam_box, text="LIVE CAMERA", font=("Segoe UI", 9, "bold"),
                 fg=self.FG_MUTED, bg=self.BG_INPUT).pack()
        self.obj_cam_label = tk.Label(cam_box, bg=self.BG_INPUT)
        self.obj_cam_label.pack(padx=4, pady=4)

        info = tk.Frame(body, bg=self.BG_DARK)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.obj_detected_lbl = tk.Label(info, text="Detected: (waiting...)",
                                          font=("Segoe UI", 13, "bold"),
                                          fg=self.FG_MUTED, bg=self.BG_DARK)
        self.obj_detected_lbl.pack(anchor=tk.W, pady=4)

        self.obj_conf_lbl = tk.Label(info, text="Confidence: —",
                                      font=("Segoe UI", 10),
                                      fg=self.ACCENT_CYAN, bg=self.BG_DARK)
        self.obj_conf_lbl.pack(anchor=tk.W)

        self.obj_sustained_lbl = tk.Label(
            info, text="Sustained: 0.0s / 3.0s",
            font=("Segoe UI", 10), fg=self.FG_MUTED, bg=self.BG_DARK)
        self.obj_sustained_lbl.pack(anchor=tk.W, pady=2)

        self.obj_progress_canvas = tk.Canvas(info, width=280, height=14,
                                              bg="#1e293b", highlightthickness=0)
        self.obj_progress_canvas.pack(anchor=tk.W, pady=4)

        self.obj_status = tk.Label(info, text="Hold an object steady in the center box",
                                    font=("Segoe UI", 10),
                                    fg=self.FG_MUTED, bg=self.BG_DARK)
        self.obj_status.pack(anchor=tk.W, pady=8)

        ctrl = tk.Frame(info, bg=self.BG_FIELD, padx=10, pady=8)
        ctrl.pack(fill=tk.X, pady=4)

        self.obj_rec_btn = tk.Button(
            ctrl, text="🎤 RECORD", font=("Segoe UI", 10, "bold"),
            bg=self.ACCENT_RED, fg="white", state=tk.DISABLED,
            command=self._obj_toggle_record)
        self.obj_rec_btn.pack(side=tk.LEFT, padx=4)

        self.obj_play_btn = tk.Button(
            ctrl, text="▶ PLAY", font=("Segoe UI", 10, "bold"),
            bg="#475569", fg="white", state=tk.DISABLED,
            command=lambda: self.recorder.play_latest() or None)
        self.obj_play_btn.pack(side=tk.LEFT, padx=4)

        self.obj_save_btn = tk.Button(
            ctrl, text="💾 SAVE", font=("Segoe UI", 10, "bold"),
            bg=self.ACCENT_GREEN, fg="white", state=tk.DISABLED,
            command=self._obj_save)
        self.obj_save_btn.pack(side=tk.LEFT, padx=4)

        entry_row = tk.Frame(info, bg=self.BG_DARK)
        entry_row.pack(fill=tk.X, pady=4)
        tk.Label(entry_row, text="Santali Word:", font=("Segoe UI", 10, "bold"),
                 fg=self.ACCENT_CYAN, bg=self.BG_DARK).pack(side=tk.LEFT)
        self.obj_word_entry = tk.Entry(entry_row, width=24,
                                        bg=self.BG_FIELD, fg="white",
                                        font=("Segoe UI", 11),
                                        insertbackground="white")
        self.obj_word_entry.pack(side=tk.LEFT, padx=8)

        # Reset state
        self.sustained_class = None
        self.sustained_time = 0.0
        self.last_frame_time = time.time()
        self.object_triggered = False
        self._obj_detected_class = None

        # Start camera
        self.camera_running = True
        self._obj_camera_loop()

    def _obj_camera_loop(self):
        """Camera loop for object mode — runs MobileNetV3 and tracks sustained detection."""
        if not self.camera_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            box_size = 224
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2, y2 = x1 + box_size, y1 + box_size

            roi = frame[y1:y2, x1:x2]
            clean_label, conf, raw_label = self.classifier.classify_crop(roi)

            CONF_THRESHOLD = 0.45
            is_detected = conf >= CONF_THRESHOLD
            now = time.time()
            dt = now - self.last_frame_time
            self.last_frame_time = now

            # Sustained detection tracking
            if not self.object_triggered:
                if is_detected and clean_label == self.sustained_class:
                    self.sustained_time += dt
                elif is_detected:
                    self.sustained_class = clean_label
                    self.sustained_time = dt
                else:
                    self.sustained_class = None
                    self.sustained_time = 0.0

                if self.sustained_time >= self.sustained_threshold:
                    self._obj_trigger(clean_label)

            # Update UI
            box_color = (56, 189, 248) if is_detected else (100, 116, 139)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, "HOLD OBJECT HERE",
                        (x1 + 18, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, box_color, 1)

            if is_detected:
                display_name = clean_label.replace("_", " ").title()
                self.obj_detected_lbl.config(
                    text=f"Detected: {display_name}", fg=self.ACCENT_GREEN)
                self.obj_conf_lbl.config(
                    text=f"Confidence: {conf*100:.1f}%", fg=self.ACCENT_GREEN)
                self.eyes.state = "detect"
            else:
                self.obj_detected_lbl.config(
                    text="Detected: (scanning...)", fg=self.FG_MUTED)
                self.obj_conf_lbl.config(text="Confidence: —", fg=self.FG_MUTED)
                if self.eyes.state == "detect":
                    self.eyes.state = "normal"

            # Update sustained progress
            progress = min(self.sustained_time / self.sustained_threshold, 1.0)
            self.obj_sustained_lbl.config(
                text=f"Sustained: {self.sustained_time:.1f}s / {self.sustained_threshold:.1f}s")
            self.obj_progress_canvas.delete("all")
            bar_w = int(280 * progress)
            bar_color = self.ACCENT_GREEN if progress >= 1.0 else self.ACCENT_CYAN
            self.obj_progress_canvas.create_rectangle(
                0, 0, bar_w, 14, fill=bar_color, outline="")

            # Display camera frame in GUI
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (340, 240))
            img = ImageTk.PhotoImage(image=Image.fromarray(rgb_small))
            self.obj_cam_label.imgtk = img
            self.obj_cam_label.configure(image=img)

        self.root.after(25, self._obj_camera_loop)

    def _obj_trigger(self, clean_label):
        """Auto-triggered when object is detected for 3+ seconds."""
        self.object_triggered = True
        self._obj_detected_class = clean_label
        display_name = clean_label.replace("_", " ").title()

        self.obj_status.config(
            text=f"Object locked: {display_name} — Say the word!",
            fg=self.ACCENT_GREEN)
        self.obj_rec_btn.config(state=tk.NORMAL)

        # Pre-fill Santali suggestion if available
        self.obj_word_entry.delete(0, tk.END)
        if clean_label in SANTALI_OBJECT_MAP:
            self.obj_word_entry.insert(0, SANTALI_OBJECT_MAP[clean_label])

        # Show prompt on ST7789 & speak with robot voice
        def _display():
            self.eyes.pause()
            self.eyes.display_prompt_card(
                title=f"📷 {display_name}",
                bangla_text="এটাকে তোমার ভাষায় কী বলে?",
                english_text="What is this called in your language?",
                hint="🎤 Speak word now..."
            )
            # Robot voice speaks object prompt
            self.tts.speak(f"Say the name for this {display_name.lower()} in your language.")
        threading.Thread(target=_display, daemon=True).start()

    def _obj_toggle_record(self):
        if not self.recorder.is_recording:
            self.recorder.start_recording()
            self.eyes.state = "listening"
            self.obj_rec_btn.config(text="⏹ STOP", bg="#b91c1c")
            self.obj_status.config(text="🔴 Recording...", fg=self.ACCENT_RED)
        else:
            label = (self._obj_detected_class or "unknown").lower()
            path = self.recorder.stop_recording(mode_prefix=f"object_{label}")
            self.eyes.state = "normal"
            self.eyes.resume()
            self.obj_rec_btn.config(text="🎤 RECORD", bg=self.ACCENT_RED)
            self.obj_play_btn.config(state=tk.NORMAL)
            self.obj_save_btn.config(state=tk.NORMAL)
            if path:
                self.obj_status.config(
                    text="Recorded! Enter the word and save.",
                    fg=self.ACCENT_GREEN)

    def _obj_save(self):
        word = self.obj_word_entry.get().strip()
        if not word:
            messagebox.showwarning("Missing", "Enter the Santali word first.")
            return
        en_name = (self._obj_detected_class or "Unknown").replace("_", " ").title()
        self._save_to_db(word, en_name, "", "Object Mode", "Everyday Object")
        self.obj_status.config(text=f"Saved '{word}' to archive!", fg=self.ACCENT_GREEN)
        # Reset for next detection
        self.object_triggered = False
        self.sustained_class = None
        self.sustained_time = 0.0
        self.obj_rec_btn.config(state=tk.DISABLED)
        self.obj_play_btn.config(state=tk.DISABLED)
        self.obj_save_btn.config(state=tk.DISABLED)

    # ---------------------------------------------------------------
    # CONVERSATION MODE
    # ---------------------------------------------------------------
    def show_conversation_mode(self):
        self._clear_content()
        frame = tk.Frame(self._content_area, bg=self.BG_DARK)
        frame.pack(fill=tk.BOTH, expand=True)
        self._active_frame = frame

        hdr = tk.Frame(frame, bg=self.BG_CARD, padx=12, pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="💬  CONVERSATION MODE", font=("Segoe UI", 14, "bold"),
                 fg=self.ACCENT_AMBER, bg=self.BG_CARD).pack(side=tk.LEFT)
        self._make_exit_button(hdr).pack(side=tk.RIGHT)

        instr = tk.Frame(frame, bg=self.BG_INPUT, padx=16, pady=14)
        instr.pack(fill=tk.X, padx=8, pady=12)

        tk.Label(instr, text="The robot's display will show a prompt in Bangla.",
                 font=("Segoe UI", 11), fg=self.FG_PRIMARY,
                 bg=self.BG_INPUT).pack(anchor=tk.W)
        tk.Label(instr, text="Speak freely — recording will auto-stop after",
                 font=("Segoe UI", 10), fg=self.FG_MUTED,
                 bg=self.BG_INPUT).pack(anchor=tk.W)
        tk.Label(instr, text="5 seconds of silence.",
                 font=("Segoe UI", 10, "bold"), fg=self.ACCENT_AMBER,
                 bg=self.BG_INPUT).pack(anchor=tk.W)

        tk.Label(frame, text="Display prompt preview:",
                 font=("Segoe UI", 9), fg=self.FG_MUTED,
                 bg=self.BG_DARK).pack(anchor=tk.W, padx=8, pady=(12, 2))

        preview = tk.Label(frame,
                           text="তোমার গ্রামের সবচেয়ে মেলার গল্প বলো",
                           font=("Segoe UI", 16),
                           fg=self.ACCENT_AMBER, bg=self.BG_INPUT,
                           padx=16, pady=12)
        preview.pack(fill=tk.X, padx=8)

        ctrl = tk.Frame(frame, bg=self.BG_DARK, pady=12)
        ctrl.pack(fill=tk.X, padx=8)

        self.conv_rec_btn = tk.Button(
            ctrl, text="🔴 START RECORDING", font=("Segoe UI", 13, "bold"),
            bg=self.ACCENT_RED, fg="white", padx=20, pady=8,
            command=self._conv_start_recording)
        self.conv_rec_btn.pack(pady=6)

        self.conv_status = tk.Label(frame, text="Ready",
                                     font=("Segoe UI", 12),
                                     fg=self.FG_MUTED, bg=self.BG_DARK)
        self.conv_status.pack(pady=4)

        self.conv_silence_lbl = tk.Label(frame, text="",
                                          font=("Segoe UI", 10),
                                          fg=self.FG_MUTED, bg=self.BG_DARK)
        self.conv_silence_lbl.pack()

        self.conv_duration_lbl = tk.Label(frame, text="Duration: 0:00",
                                           font=("Segoe UI", 10),
                                           fg=self.FG_MUTED, bg=self.BG_DARK)
        self.conv_duration_lbl.pack()

        post_ctrl = tk.Frame(frame, bg=self.BG_FIELD, padx=10, pady=8)
        post_ctrl.pack(fill=tk.X, padx=8, pady=8)

        self.conv_play_btn = tk.Button(
            post_ctrl, text="▶ PLAY", font=("Segoe UI", 10, "bold"),
            bg="#475569", fg="white", state=tk.DISABLED,
            command=lambda: self.recorder.play_latest())
        self.conv_play_btn.pack(side=tk.LEFT, padx=4)

        self.conv_save_btn = tk.Button(
            post_ctrl, text="💾 SAVE TO ARCHIVE", font=("Segoe UI", 10, "bold"),
            bg=self.ACCENT_GREEN, fg="white", state=tk.DISABLED,
            command=self._conv_save)
        self.conv_save_btn.pack(side=tk.LEFT, padx=4)

        self._conv_start_time = None
        self._conv_update_id = None

    def _conv_start_recording(self):
        """Start conversation recording with auto-stop on silence."""
        def _display_and_record():
            self.eyes.pause()
            self.eyes.display_prompt_card(
                title="💬 STORY / গল্প",
                bangla_text="তোমার গ্রামের সবচেয়ে মেলার গল্প বলো",
                english_text="Tell the story of your village festival",
                hint="🔴 Recording... (stops on silence)"
            )
            # Robot voice speaks conversation prompt
            self.tts.speak("Please tell the story of your village festival.")

        threading.Thread(target=_display_and_record, daemon=True).start()

        self.recorder.start_recording(
            auto_stop=True,
            on_auto_stop=self._conv_auto_stopped
        )
        self.eyes.state = "listening"
        self._conv_start_time = time.time()

        self.conv_rec_btn.config(text="⏹ STOP MANUALLY", bg="#b91c1c",
                                  command=self._conv_stop_recording)
        self.conv_status.config(text="🔴 Recording — speak now...",
                                fg=self.ACCENT_RED)

        self._conv_update_ui()

    def _conv_update_ui(self):
        """Update conversation mode UI with elapsed time and silence progress."""
        if not self.recorder.is_recording:
            return

        elapsed = time.time() - self._conv_start_time
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self.conv_duration_lbl.config(text=f"Duration: {mins}:{secs:02d}")

        silence = self.recorder.get_silence_progress()
        self.conv_silence_lbl.config(
            text=f"Silence: {silence:.1f}s / {self.recorder.max_silence:.1f}s")

        self._conv_update_id = self.root.after(200, self._conv_update_ui)

    def _conv_auto_stopped(self):
        """Called when silence auto-stop triggers (from audio thread)."""
        self.root.after(0, self._conv_finalize)

    def _conv_stop_recording(self):
        """Manual stop."""
        self._conv_finalize()

    def _conv_finalize(self):
        """Finalize conversation recording."""
        if self._conv_update_id:
            self.root.after_cancel(self._conv_update_id)
            self._conv_update_id = None

        path = self.recorder.stop_recording(mode_prefix="conversation")
        self.eyes.state = "normal"
        self.eyes.resume()

        self.conv_rec_btn.config(text="🔴 START RECORDING", bg=self.ACCENT_RED,
                                  command=self._conv_start_recording)

        if path:
            elapsed = time.time() - (self._conv_start_time or time.time())
            self.conv_status.config(
                text=f"Recording saved ({elapsed:.0f}s)",
                fg=self.ACCENT_GREEN)
            self.conv_play_btn.config(state=tk.NORMAL)
            self.conv_save_btn.config(state=tk.NORMAL)
        else:
            self.conv_status.config(
                text="No audio captured.", fg=self.ACCENT_RED)

    def _conv_save(self):
        self._save_to_db(
            "Narrative Recording", "Conversation", "",
            "Conversation", "Oral Narrative"
        )
        self.conv_status.config(
            text="Saved to archive!", fg=self.ACCENT_GREEN)
        self.conv_save_btn.config(state=tk.DISABLED)

    # ---------------------------------------------------------------
    # DATABASE OPS
    # ---------------------------------------------------------------
    def _save_to_db(self, target, en, bn, mode, cat):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT INTO archive
                     (target_word, translation_en, translation_bn, category, mode, audio_file)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (target, en, bn, cat, mode, self.recorder.latest_path or ""))
        conn.commit()
        conn.close()
        self._update_db_count()

    # ---------------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------------
    def on_closing(self):
        self.camera_running = False
        self.recorder.is_recording = False
        self.cap.release()
        self.eyes.close()
        self.root.destroy()
        sys.exit(0)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ResonoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
