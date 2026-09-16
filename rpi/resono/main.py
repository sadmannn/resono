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
import random
import json
import sqlite3
import threading
import urllib.request
import tkinter as tk
from tkinter import messagebox, ttk
# Core image processing
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter

# Optional robot dependencies with graceful fallback for laptop simulation
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

# NCNN lightweight inference engine (optimised for ARM / Raspberry Pi)
try:
    import ncnn
except ImportError:
    ncnn = None

# Audio capture & playback
try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
except ImportError:
    sd = None
    wav_write = None

try:
    import pygame
    pygame.mixer.init()
except Exception:
    pygame = None

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
        if ncnn is None or cv2 is None or np is None:
            self.net = None
            print("[WARN] ncnn, cv2, or numpy not installed — classifier running in simulation mode.")
            return

        if not os.path.exists(NCNN_PARAM) or not os.path.exists(NCNN_BIN):
            print(f"[WARN] NCNN model files not found ({NCNN_PARAM}) — running in simulation mode.")
            self.net = None
            return

        self.net = ncnn.Net()
        self.net.opt.num_threads = 4
        self.net.load_param(NCNN_PARAM)
        self.net.load_model(NCNN_BIN)
        print(f"[AI] NCNN MobileNetV3-Large FP16 loaded from {SCRIPT_DIR}")

    @staticmethod
    def _softmax(x):
        """Numerically stable softmax."""
        if np is None:
            return [1.0]
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def classify_crop(self, crop_cv2):
        """Classify a BGR crop and return (clean_label, confidence, raw_label)."""
        if self.net is None or cv2 is None or np is None:
            return "WATER_BOTTLE", 0.92, "water_bottle"

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
# PROCEDURAL ANIMATION ENGINE (Bonfire & Water Bottle + Glass)
# ============================================================
class AnimationEngine:
    """Generates realistic procedural bonfire and water bottle + glass animations for the 320x170 ST7789."""

    WIDTH = 320
    HEIGHT = 170

    def __init__(self):
        # Persistent state for bonfire flying ember sparks
        self.bonfire_embers = []
        for _ in range(22):
            self.bonfire_embers.append({
                'x': 160 + random.uniform(-18, 18),
                'y': 136 - random.uniform(0, 85),
                'speed': random.uniform(1.2, 2.6),
                'size': random.uniform(1.0, 2.2),
                'life': random.uniform(0.1, 1.0),
                'sway_speed': random.uniform(3.5, 6.5),
                'phase': random.uniform(0, 6.28)
            })

        # Persistent state for water bubbles (in bottle and in glass)
        self.water_bubbles = []
        for _ in range(14):
            self.water_bubbles.append({
                'container': 'bottle',
                'x': 105 + random.uniform(-15, 15),
                'y': random.uniform(74, 134),
                'speed': random.uniform(0.5, 1.3),
                'r': random.uniform(0.8, 1.6),
                'seed': random.uniform(0, 6.28)
            })
        for _ in range(12):
            self.water_bubbles.append({
                'container': 'glass',
                'x': 215 + random.uniform(-15, 15),
                'y': random.uniform(80, 130),
                'speed': random.uniform(0.6, 1.4),
                'r': random.uniform(0.8, 1.8),
                'seed': random.uniform(0, 6.28)
            })

    def render_bonfire_frame(self, t):
        """Render a realistic small bonfire with burning crossed logs, glowing embers, and organic dancing flames."""
        w, h = self.WIDTH, self.HEIGHT
        canvas = Image.new('RGBA', (w, h), (7, 10, 18, 255))
        draw = ImageDraw.Draw(canvas)

        # 1. Ground terrain & Fire pit stones
        draw.rectangle([0, 142, w, h], fill=(16, 14, 13, 255))
        for sx, sy, sr in [(105, 148, 8), (128, 152, 10), (160, 154, 12), (192, 151, 10), (215, 147, 8), (145, 147, 7), (175, 148, 7)]:
            draw.ellipse([sx-sr, sy-sr//2, sx+sr, sy+sr//2], fill=(38, 34, 32, 255), outline=(24, 20, 18, 255))

        cx, base_y = 160, 138

        # 2. Ambient Fire Bloom (Pulsing warm atmospheric glow)
        pulse = 0.85 + 0.15 * math.sin(t * 8.0) + 0.05 * math.sin(t * 17.3)
        gw, gh = int(125 * pulse), int(95 * pulse)
        glow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse([cx - gw, base_y - gh - 10, cx + gw, base_y + 20], fill=(255, 80, 10, int(55 * pulse)))
        gdraw.ellipse([cx - int(gw*0.6), base_y - int(gh*0.7), cx + int(gw*0.6), base_y + 10], fill=(255, 160, 20, int(75 * pulse)))
        glow = glow.filter(ImageFilter.GaussianBlur(18))
        canvas = Image.alpha_composite(canvas, glow)

        # 3. Flame Tongues
        flame = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        fdraw = ImageDraw.Draw(flame)

        def get_flame_poly(height, width_base, sway_freq, sway_amp, t_off, tip_x_bias=0):
            tip_x = cx + tip_x_bias + math.sin(t * sway_freq + t_off) * sway_amp + math.sin(t * 11.7 + t_off*2) * (sway_amp * 0.4)
            tip_y = base_y - height + math.cos(t * 9.3 + t_off) * 6
            pts = []
            steps = 14
            for i in range(steps + 1):
                pct = i / steps
                y = base_y - pct * (base_y - tip_y)
                spread = (1.0 - pct) * (width_base * 0.5) * (1.0 + 0.15 * math.sin(y * 0.1 + t * 6))
                x = cx - spread + math.sin(y * 0.05 - t * 4) * (pct * sway_amp * 0.5)
                pts.append((x, y))
            for i in range(steps, -1, -1):
                pct = i / steps
                y = base_y - pct * (base_y - tip_y)
                spread = (1.0 - pct) * (width_base * 0.5) * (1.0 + 0.15 * math.cos(y * 0.1 - t * 5))
                x = cx + spread + math.sin(y * 0.05 - t * 4) * (pct * sway_amp * 0.5)
                pts.append((x, y))
            return pts

        # Multi-layer organic flame geometry
        fdraw.polygon(get_flame_poly(54, 30, 6.5, 10, 1.2, -12), fill=(245, 80, 10, 190))  # Left flame
        fdraw.polygon(get_flame_poly(58, 32, 7.2, 11, 3.5, 12), fill=(245, 85, 10, 190))   # Right flame
        fdraw.polygon(get_flame_poly(82, 48, 5.0, 8, 0.0), fill=(255, 90, 10, 220))        # Outer mantle
        fdraw.polygon(get_flame_poly(64, 36, 5.8, 6, 0.5), fill=(255, 175, 15, 240))       # Mid amber body
        fdraw.polygon(get_flame_poly(44, 24, 7.0, 4, 1.0), fill=(255, 235, 70, 250))       # Inner golden core
        fdraw.polygon(get_flame_poly(26, 14, 8.5, 2, 1.5), fill=(255, 255, 225, 255))      # White-hot heart

        flame = flame.filter(ImageFilter.GaussianBlur(1.2))
        canvas = Image.alpha_composite(canvas, flame)

        # 4. Bonfire Wooden Logs
        logs = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(logs)

        # Back log
        ldraw.polygon([(118, 147), (185, 131), (188, 137), (121, 153)], fill=(58, 36, 22, 255), outline=(32, 18, 10, 255))
        ldraw.ellipse([116, 146, 122, 154], fill=(78, 48, 28, 255), outline=(32, 18, 10, 255))

        # Crossed front log
        ldraw.polygon([(202, 147), (135, 131), (132, 137), (199, 153)], fill=(65, 40, 24, 255), outline=(32, 18, 10, 255))
        ldraw.ellipse([198, 146, 204, 154], fill=(82, 52, 30, 255), outline=(32, 18, 10, 255))

        # Horizontal bottom log
        ldraw.polygon([(128, 148), (192, 148), (190, 155), (130, 155)], fill=(48, 30, 18, 255), outline=(26, 14, 8, 255))

        # Glowing charcoal / embers bed in center
        for ex, ey, er in [(150, 141, 5), (160, 139, 6), (170, 142, 5), (155, 145, 4), (165, 144, 4)]:
            glow_val = int(200 + 55 * math.sin(t * 10 + ex))
            ldraw.ellipse([ex-er, ey-er//2, ex+er, ey+er//2], fill=(glow_val, int(glow_val*0.4), 10, 245))

        canvas = Image.alpha_composite(canvas, logs)

        # 5. Rising Sparks & Embers
        sparks = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(sparks)
        for p in self.bonfire_embers:
            p['y'] -= p['speed']
            p['x'] += math.sin(t * p['sway_speed'] + p['phase']) * 0.8
            p['life'] -= 0.018
            if p['life'] <= 0 or p['y'] < 15:
                p['x'] = cx + random.uniform(-18, 18)
                p['y'] = base_y - random.uniform(15, 30)
                p['speed'] = random.uniform(1.2, 2.6)
                p['size'] = random.uniform(1.0, 2.2)
                p['life'] = random.uniform(0.7, 1.0)
                p['sway_speed'] = random.uniform(3.5, 6.5)
                p['phase'] = random.uniform(0, 6.28)

            alpha = int(255 * max(0.0, p['life']))
            col = (255, int(150 + 105 * p['life']), int(30 + 160 * p['life']), alpha)
            sz = p['size']
            sdraw.ellipse([p['x']-sz, p['y']-sz, p['x']+sz, p['y']+sz], fill=col)

        canvas = Image.alpha_composite(canvas, sparks)
        return canvas.convert('RGB')

    def render_water_frame(self, t):
        """Render a realistic water bottle and glass of water side by side with animated waves and bubbles."""
        w, h = self.WIDTH, self.HEIGHT
        canvas = Image.new('RGBA', (w, h), (7, 12, 22, 255))
        draw = ImageDraw.Draw(canvas)

        # Soft background illumination
        glow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse([30, 20, 190, 150], fill=(2, 132, 199, 45))
        gdraw.ellipse([140, 30, 290, 155], fill=(14, 165, 233, 45))
        glow = glow.filter(ImageFilter.GaussianBlur(28))
        canvas = Image.alpha_composite(canvas, glow)

        # Floor / table surface
        draw.rectangle([0, 138, w, h], fill=(11, 18, 30, 255))
        draw.line([(0, 138), (w, 138)], fill=(28, 45, 70, 255), width=1)

        bx, by, bw = 105, 138, 21
        gx, gy_bot = 215, 138

        # Ground reflections
        refl = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        rdraw = ImageDraw.Draw(refl)
        rdraw.polygon([(bx-18, by+1), (bx+18, by+1), (bx+14, by+26), (bx-14, by+26)], fill=(14, 165, 233, 45))
        rdraw.polygon([(gx-20, by+1), (gx+20, by+1), (gx+16, by+24), (gx-16, by+24)], fill=(14, 165, 233, 45))
        refl = refl.filter(ImageFilter.GaussianBlur(5))
        canvas = Image.alpha_composite(canvas, refl)

        # ----------------- WATER LIQUID LAYER -----------------
        water_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        wdraw = ImageDraw.Draw(water_layer)

        # 1. Bottle Water Fill
        water_top = 70
        w_pts = [(bx - bw + 3, by - 2), (bx + bw - 3, by - 2), (bx + bw - 3, water_top)]
        for i in range(16, -1, -1):
            pct = i / 16
            wx = (bx - bw + 3) + pct * (bw * 2 - 6)
            wy = water_top + math.sin(t * 5.0 + pct * 6.28) * 1.8 + math.cos(t * 3.2 + pct * 3.14) * 0.7
            w_pts.append((wx, wy))
        w_pts.append((bx - bw + 3, water_top))
        wdraw.polygon(w_pts, fill=(14, 165, 233, 220))
        wdraw.line(w_pts[3:], fill=(186, 230, 253, 240), width=2)

        # 2. Glass Water Fill
        gtw, gbw, gy_rim = 26, 20, 60
        g_water_top = 76
        gw_pts = [(gx - gbw + 3, gy_bot - 8), (gx + gbw - 3, gy_bot - 8), (gx + int(gbw + (gtw-gbw)*0.75) - 2, g_water_top)]
        for i in range(16, -1, -1):
            pct = i / 16
            cur_w = (gbw + (gtw - gbw) * 0.75) - 3
            wx = (gx - cur_w) + pct * (cur_w * 2)
            wy = g_water_top + math.cos(t * 4.6 + pct * 6.28) * 1.7 + math.sin(t * 2.8 + pct * 3.14) * 0.6
            gw_pts.append((wx, wy))
        gw_pts.append((gx - int(gbw + (gtw-gbw)*0.75) + 2, g_water_top))
        wdraw.polygon(gw_pts, fill=(14, 165, 233, 215))
        wdraw.line(gw_pts[3:], fill=(224, 242, 254, 240), width=2)

        canvas = Image.alpha_composite(canvas, water_layer)

        # ----------------- GLASS / PLASTIC LAYER -----------------
        glass_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        gldraw = ImageDraw.Draw(glass_layer)

        # Bottle Outer Walls
        body_poly = [
            (bx - 8, 54), (bx - bw, 68),
            (bx - bw, 134), (bx - bw + 5, by),
            (bx + bw - 5, by), (bx + bw, 134),
            (bx + bw, 68), (bx + 8, 54)
        ]
        gldraw.polygon(body_poly, fill=(255, 255, 255, 18), outline=(186, 230, 253, 200))

        # Bottle Cap
        gldraw.rectangle([bx - 9, 36, bx + 9, 46], fill=(29, 78, 216, 255), outline=(96, 165, 250, 255))
        for rx in [-6, -2, 2, 6]:
            gldraw.line([(bx + rx, 37), (bx + rx, 45)], fill=(147, 197, 253, 220))
        gldraw.rectangle([bx - 10, 46, bx + 10, 50], fill=(30, 64, 175, 255))
        gldraw.rectangle([bx - 8, 50, bx + 8, 54], fill=(255, 255, 255, 30), outline=(186, 230, 253, 160))

        # Bottle Specular Gloss
        gldraw.line([(bx - bw + 4, 70), (bx - bw + 4, 132)], fill=(255, 255, 255, 210), width=2)
        gldraw.line([(bx - bw + 7, 72), (bx - bw + 7, 130)], fill=(255, 255, 255, 80), width=1)
        gldraw.line([(bx + bw - 4, 70), (bx + bw - 4, 132)], fill=(255, 255, 255, 100), width=1)

        # Glass Tumbler Walls & Solid Base
        gldraw.polygon([
            (gx - gbw + 2, gy_bot - 8), (gx + gbw - 2, gy_bot - 8),
            (gx + gbw, gy_bot), (gx - gbw, gy_bot)
        ], fill=(255, 255, 255, 80), outline=(186, 230, 253, 190))

        gldraw.line([(gx - gtw, gy_rim), (gx - gbw, gy_bot)], fill=(224, 242, 254, 210), width=2)
        gldraw.line([(gx + gtw, gy_rim), (gx + gbw, gy_bot)], fill=(224, 242, 254, 180), width=2)
        gldraw.line([(gx - gbw, gy_bot), (gx + gbw, gy_bot)], fill=(224, 242, 254, 210), width=2)
        gldraw.ellipse([gx - gtw, gy_rim - 4, gx + gtw, gy_rim + 4], outline=(224, 242, 254, 230), width=2)

        # Glass Specular Reflections
        gldraw.line([(gx - gtw + 5, gy_rim + 8), (gx - gbw + 4, gy_bot - 10)], fill=(255, 255, 255, 170), width=2)
        gldraw.line([(gx + gtw - 5, gy_rim + 8), (gx + gbw - 4, gy_bot - 10)], fill=(255, 255, 255, 90), width=1)

        # Floating Ice Cubes in Glass
        cube_dy = math.sin(t * 3.8) * 1.5
        c1x, c1y = gx - 11, g_water_top + 4 + cube_dy
        gldraw.rectangle([c1x, c1y, c1x + 14, c1y + 13], fill=(255, 255, 255, 120), outline=(255, 255, 255, 220))
        gldraw.line([(c1x + 2, c1y + 2), (c1x + 10, c1y + 2)], fill=(255, 255, 255, 250))
        gldraw.line([(c1x + 2, c1y + 2), (c1x + 2, c1y + 9)], fill=(255, 255, 255, 250))

        c2x, c2y = gx + 3, g_water_top + 7 - cube_dy * 0.7
        gldraw.rectangle([c2x, c2y, c2x + 13, c2y + 12], fill=(255, 255, 255, 100), outline=(255, 255, 255, 200))
        gldraw.line([(c2x + 2, c2y + 2), (c2x + 9, c2y + 2)], fill=(255, 255, 255, 240))

        canvas = Image.alpha_composite(canvas, glass_layer)

        # ----------------- BUBBLES LAYER -----------------
        bubble_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        bbdraw = ImageDraw.Draw(bubble_layer)
        for b in self.water_bubbles:
            b['y'] -= b['speed']
            b['x'] += math.sin(t * 6.0 + b['seed']) * 0.4
            top_lim = water_top + 2 if b['container'] == 'bottle' else g_water_top + 2
            bot_lim = by - 6 if b['container'] == 'bottle' else by - 12
            if b['y'] < top_lim:
                b['y'] = bot_lim - random.uniform(0, 10)
                if b['container'] == 'bottle':
                    b['x'] = bx + random.uniform(-bw + 7, bw - 7)
                else:
                    b['x'] = gx + random.uniform(-gbw + 6, gbw - 6)

            r = b['r']
            bbdraw.ellipse([b['x'] - r, b['y'] - r, b['x'] + r, b['y'] + r], fill=(255, 255, 255, 230), outline=(186, 230, 253, 255))

        canvas = Image.alpha_composite(canvas, bubble_layer)
        return canvas.convert('RGB')

    def generate_fire_animation(self, display_controller, duration=None, stop_event=None, frame_callback=None):
        """Run the realistic bonfire animation on loop on ST7789 and optional preview callback."""
        t_start = time.time()
        while True:
            if stop_event and stop_event.is_set():
                break
            t_elapsed = time.time() - t_start
            if duration is not None and t_elapsed >= duration:
                break

            frame_img = self.render_bonfire_frame(t_elapsed)
            display_controller.display_image(frame_img)
            if frame_callback:
                frame_callback(frame_img)

            time.sleep(0.04)

    def generate_water_frames(self, display_controller, duration=None, stop_event=None, frame_callback=None):
        """Run the realistic water bottle and glass animation on loop on ST7789 and optional preview callback."""
        t_start = time.time()
        while True:
            if stop_event and stop_event.is_set():
                break
            t_elapsed = time.time() - t_start
            if duration is not None and t_elapsed >= duration:
                break

            frame_img = self.render_water_frame(t_elapsed)
            display_controller.display_image(frame_img)
            if frame_callback:
                frame_callback(frame_img)

            time.sleep(0.04)


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

        if sd is None:
            print("[WARN] sounddevice not installed — recording in simulation mode.")
            return

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

        if self._auto_stop_enabled and np is not None:
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

        if sd is None or wav_write is None or np is None:
            path = os.path.join(AUDIO_DIR, f"{mode_prefix}_{int(time.time())}.wav")
            with open(path, "wb") as f:
                f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
            self.latest_path = path
            return path

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
        if pygame and self.latest_path and os.path.exists(self.latest_path):
            try:
                pygame.mixer.music.load(self.latest_path)
                pygame.mixer.music.play()
                return True
            except Exception:
                return False
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
        self.cap = cv2.VideoCapture(0) if cv2 is not None else None
        self.camera_running = False

        # Object mode tracking
        self.sustained_class = None
        self.sustained_time = 0.0
        self.sustained_threshold = 3.0  # seconds
        self.last_frame_time = time.time()
        self.object_triggered = False

        # Current frame reference (to prevent GC)
        self._current_frame = None

        # Archive showcase state
        self.tree = None
        self._archive_all_rows = []

        # Prompt animation state
        self._prompt_stop_anim = None

        # TTK Styling
        self._setup_ttk_styles()

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

        # Right side: Show Database button and Word Counter
        right_box = tk.Frame(head, bg=self.BG_CARD)
        right_box.pack(side=tk.RIGHT, pady=8)

        self.btn_db_show = tk.Button(
            right_box, text="📂 SHOW DATABASE",
            font=("Segoe UI", 9, "bold"),
            bg=self.ACCENT_BLUE, fg="white",
            activebackground=self.ACCENT_CYAN,
            relief=tk.FLAT, padx=10, pady=3,
            cursor="hand2",
            command=self.show_archive_mode
        )
        self.btn_db_show.pack(side=tk.LEFT, padx=(0, 14))

        self.db_counter_lbl = tk.Label(right_box, text="Words: 0",
                                        font=("Segoe UI", 10),
                                        fg=self.FG_MUTED, bg=self.BG_CARD)
        self.db_counter_lbl.pack(side=tk.LEFT)
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
        self._stop_prompt_animation()
        self._stop_archive_audio()
        if self._active_frame:
            self._active_frame.destroy()
            self._active_frame = None
        self.tree = None
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
        self._stop_prompt_animation()
        self._stop_archive_audio()
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
                 font=("Segoe UI", 16, "bold"),
                 fg=self.FG_PRIMARY, bg=self.BG_DARK).pack(pady=(14, 10))

        # Centered container for mode cards
        cards = tk.Frame(frame, bg=self.BG_DARK)
        cards.pack(expand=True, pady=(0, 10))

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

        self._make_mode_card(
            cards,
            icon="📂", title="ARCHIVE SHOWCASE",
            subtitle="Browse database & audio recordings",
            desc="Inspect preserved words, listen to recordings, verify database",
            color=self.ACCENT_BLUE,
            command=self.show_archive_mode
        )

    def _make_mode_card(self, parent, icon, title, subtitle, desc, color, command):
        """Create a rock-solid, jitter-free clickable mode selection card."""
        card = tk.Frame(parent, bg=self.BG_CARD, cursor="hand2",
                        padx=18, pady=8, bd=0, highlightthickness=2,
                        highlightbackground="#334155", highlightcolor=color,
                        width=520, height=86)
        card.pack_propagate(False)
        card.pack(fill=tk.X, pady=4)

        top = tk.Frame(card, bg=self.BG_CARD)
        top.pack(fill=tk.X)
        i_lbl = tk.Label(top, text=icon, font=("Segoe UI", 20),
                         bg=self.BG_CARD, fg=color)
        i_lbl.pack(side=tk.LEFT)
        t_lbl = tk.Label(top, text=f"  {title}", font=("Segoe UI", 13, "bold"),
                         bg=self.BG_CARD, fg=color)
        t_lbl.pack(side=tk.LEFT)

        s_lbl = tk.Label(card, text=subtitle, font=("Segoe UI", 9),
                         bg=self.BG_CARD, fg=self.FG_PRIMARY)
        s_lbl.pack(anchor=tk.W, padx=38)
        d_lbl = tk.Label(card, text=desc, font=("Segoe UI", 8),
                         bg=self.BG_CARD, fg=self.FG_MUTED)
        d_lbl.pack(anchor=tk.W, padx=38)

        all_widgets = [card, top, i_lbl, t_lbl, s_lbl, d_lbl]
        state = {"hovered": False, "timer": None}

        def _on_click(e):
            command()

        def _apply_hover():
            if not state["hovered"]:
                state["hovered"] = True
                card.config(highlightbackground=color, bg="#24334a")
                for w in all_widgets:
                    try:
                        w.config(bg="#24334a")
                    except Exception:
                        pass

        def _apply_unhover():
            state["hovered"] = False
            card.config(highlightbackground="#334155", bg=self.BG_CARD)
            for w in all_widgets:
                try:
                    w.config(bg=self.BG_CARD)
                except Exception:
                    pass

        def _on_enter(e):
            if state["timer"] is not None:
                try:
                    self.root.after_cancel(state["timer"])
                except Exception:
                    pass
                state["timer"] = None
            _apply_hover()

        def _on_leave(e):
            if state["timer"] is not None:
                try:
                    self.root.after_cancel(state["timer"])
                except Exception:
                    pass
            state["timer"] = self.root.after(40, _apply_unhover)

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

        # Content Row: Left animation selection buttons, Right 320x170 live display monitor
        mid_row = tk.Frame(frame, bg=self.BG_DARK)
        mid_row.pack(fill=tk.X, padx=8, pady=8)

        # Left: Animation trigger buttons
        left_ctrl = tk.Frame(mid_row, bg=self.BG_DARK)
        left_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))

        tk.Label(left_ctrl, text="Select concept animation for robot:",
                 font=("Segoe UI", 10, "bold"), fg=self.FG_PRIMARY,
                 bg=self.BG_DARK).pack(anchor=tk.W, pady=(0, 6))

        tk.Button(left_ctrl, text="🔥  BONFIRE (FIRE)", font=("Segoe UI", 12, "bold"),
                  bg="#991b1b", fg="white", activebackground="#dc2626",
                  width=24, height=2, relief=tk.FLAT, cursor="hand2",
                  command=lambda: self._prompt_play_animation("fire")
                  ).pack(pady=4)

        tk.Button(left_ctrl, text="💧  BOTTLE & GLASS (WATER)", font=("Segoe UI", 12, "bold"),
                  bg="#1e3a5f", fg="white", activebackground="#2563eb",
                  width=24, height=2, relief=tk.FLAT, cursor="hand2",
                  command=lambda: self._prompt_play_animation("water")
                  ).pack(pady=4)

        # Right: ST7789 Display Live Preview Monitor
        right_preview = tk.Frame(mid_row, bg=self.BG_INPUT, padx=4, pady=4, bd=1, relief=tk.SOLID)
        right_preview.pack(side=tk.RIGHT, padx=4)

        tk.Label(right_preview, text="ST7789 ROBOT DISPLAY PREVIEW (320×170)",
                 font=("Segoe UI", 8, "bold"), fg=self.FG_MUTED, bg=self.BG_INPUT).pack(pady=(0, 2))

        self.prompt_preview_lbl = tk.Label(right_preview, bg="#070a12", width=320, height=170)
        self.prompt_preview_lbl.pack()

        self.prompt_status = tk.Label(frame, text="Select Fire or Water above to start looping animation on robot",
                                       font=("Segoe UI", 10),
                                       fg=self.FG_MUTED, bg=self.BG_DARK)
        self.prompt_status.pack(pady=4)

        ctrl = tk.Frame(frame, bg=self.BG_FIELD, padx=12, pady=8)
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

        entry_frame = tk.Frame(frame, bg=self.BG_DARK, pady=6)
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
        """Play bonfire or water bottle + glass animation on ST7789 on loop, with live GUI preview."""
        self._stop_prompt_animation()
        self._prompt_current_anim = anim_type

        # Pre-fill Santali suggestions
        self.prompt_word_entry.delete(0, tk.END)
        if anim_type == "fire":
            self.prompt_word_entry.insert(0, "Sengel")
            self.prompt_status.config(
                text="🔥 Bonfire animation looping on robot display — Speak the word now!",
                fg=self.ACCENT_GREEN)
        else:
            self.prompt_word_entry.insert(0, "Dak'")
            self.prompt_status.config(
                text="💧 Water bottle & glass animation looping on robot display — Speak the word now!",
                fg=self.ACCENT_GREEN)

        self.prompt_rec_btn.config(state=tk.NORMAL)

        # Robot voice speaks question prompt
        self.tts.speak("What is this called in your language?")

        self._prompt_stop_anim = threading.Event()
        stop_event = self._prompt_stop_anim

        def _update_gui_preview(frame_img):
            def _apply():
                if hasattr(self, 'prompt_preview_lbl'):
                    try:
                        if self.prompt_preview_lbl.winfo_exists():
                            imgtk = ImageTk.PhotoImage(frame_img)
                            self.prompt_preview_lbl.imgtk = imgtk
                            self.prompt_preview_lbl.configure(image=imgtk)
                    except Exception:
                        pass
            try:
                self.root.after(0, _apply)
            except Exception:
                pass

        def _run():
            self.eyes.pause()
            try:
                if anim_type == "fire":
                    self.animations.generate_fire_animation(
                        self.eyes, stop_event=stop_event, frame_callback=_update_gui_preview
                    )
                else:
                    self.animations.generate_water_frames(
                        self.eyes, stop_event=stop_event, frame_callback=_update_gui_preview
                    )
            finally:
                if not self.recorder.is_recording:
                    self.eyes.resume()

        threading.Thread(target=_run, daemon=True).start()

    def _stop_prompt_animation(self):
        """Stop active bonfire or water animation loop."""
        if hasattr(self, '_prompt_stop_anim') and self._prompt_stop_anim is not None:
            self._prompt_stop_anim.set()
            self._prompt_stop_anim = None

    def _prompt_toggle_record(self):
        if not self.recorder.is_recording:
            self.recorder.start_recording()
            self.eyes.state = "listening"
            self.prompt_rec_btn.config(text="⏹ STOP", bg="#b91c1c")
            self.prompt_status.config(text="🔴 Recording audio...", fg=self.ACCENT_RED)
        else:
            path = self.recorder.stop_recording(
                mode_prefix=f"prompt_{self._prompt_current_anim or 'unknown'}")
            self._stop_prompt_animation()
            self.eyes.state = "normal"
            self.eyes.resume()
            self.prompt_rec_btn.config(text="🎤 RECORD", bg=self.ACCENT_RED)
            self.prompt_play_btn.config(state=tk.NORMAL)
            self.prompt_save_btn.config(state=tk.NORMAL)
            if path:
                self.prompt_status.config(
                    text="Recorded cleanly! Verify the Santali word and click SAVE.",
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
        self._stop_prompt_animation()
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

        if self.cap is None or cv2 is None:
            # Simulation mode placeholder frame
            sim_img = Image.new("RGB", (340, 240), (15, 23, 42))
            sim_draw = ImageDraw.Draw(sim_img)
            sim_draw.rectangle([40, 30, 300, 210], outline=(14, 165, 233), width=2)
            sim_draw.text((70, 110), "CAMERA SIMULATION MODE", fill=(148, 163, 184))
            imgtk = ImageTk.PhotoImage(sim_img)
            self._current_frame = imgtk
            if hasattr(self, 'cam_label') and self.cam_label.winfo_exists():
                self.cam_label.configure(image=imgtk)
            self.root.after(200, self._obj_camera_loop)
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
    # ARCHIVE SHOWCASE / DATABASE VIEW
    # ---------------------------------------------------------------
    def _setup_ttk_styles(self):
        """Configure ttk styles for Treeview and scrollbars to match dark theme."""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        style.configure("Treeview",
                        background=self.BG_INPUT,
                        foreground=self.FG_PRIMARY,
                        fieldbackground=self.BG_INPUT,
                        rowheight=26,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
                        background=self.BG_FIELD,
                        foreground=self.ACCENT_CYAN,
                        font=("Segoe UI", 9, "bold"),
                        relief=tk.FLAT)
        style.map("Treeview",
                  background=[("selected", self.ACCENT_BLUE)],
                  foreground=[("selected", "white")])
        style.map("Treeview.Heading",
                  background=[("active", "#475569")])

    def show_archive_mode(self):
        """Display the database archive showcase table with playback and search controls."""
        # Gracefully stop active background loops from other modes
        self.camera_running = False
        self.recorder.is_recording = False
        self.eyes.state = "normal"
        self.eyes.resume()
        self.object_triggered = False
        self.sustained_class = None
        self.sustained_time = 0.0

        self._clear_content()
        frame = tk.Frame(self._content_area, bg=self.BG_DARK)
        frame.pack(fill=tk.BOTH, expand=True)
        self._active_frame = frame

        # Mode Header
        hdr = tk.Frame(frame, bg=self.BG_CARD, padx=12, pady=8)
        hdr.pack(fill=tk.X)

        title_box = tk.Frame(hdr, bg=self.BG_CARD)
        title_box.pack(side=tk.LEFT)

        tk.Label(title_box, text="📂  DATABASE ARCHIVE SHOWCASE", font=("Segoe UI", 14, "bold"),
                 fg=self.ACCENT_CYAN, bg=self.BG_CARD).pack(side=tk.LEFT)
        tk.Label(title_box, text=" | SQLite: resono_archive.db", font=("Segoe UI", 9),
                 fg=self.FG_MUTED, bg=self.BG_CARD).pack(side=tk.LEFT, padx=6)

        self._make_exit_button(hdr).pack(side=tk.RIGHT)

        # Search & Filter bar
        filter_bar = tk.Frame(frame, bg=self.BG_INPUT, padx=10, pady=8)
        filter_bar.pack(fill=tk.X, padx=4, pady=(8, 4))

        tk.Label(filter_bar, text="🔍 Search:", font=("Segoe UI", 10, "bold"),
                 fg=self.FG_PRIMARY, bg=self.BG_INPUT).pack(side=tk.LEFT, padx=(0, 6))

        self.archive_search_var = tk.StringVar()
        self.archive_search_var.trace_add("write", lambda *args: self._filter_archive_tree())
        search_entry = tk.Entry(filter_bar, textvariable=self.archive_search_var,
                                width=26, bg=self.BG_FIELD, fg="white",
                                font=("Segoe UI", 10), insertbackground="white")
        search_entry.pack(side=tk.LEFT, padx=(0, 12))

        self.archive_count_lbl = tk.Label(filter_bar, text="Showing: 0 records",
                                          font=("Segoe UI", 10),
                                          fg=self.FG_MUTED, bg=self.BG_INPUT)
        self.archive_count_lbl.pack(side=tk.RIGHT)

        # Treeview frame
        tree_frame = tk.Frame(frame, bg=self.BG_DARK)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        columns = ("id", "target", "en", "bn", "mode", "category", "audio", "timestamp")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="ID")
        self.tree.column("id", width=36, minwidth=30, anchor=tk.CENTER)

        self.tree.heading("target", text="Santali Word")
        self.tree.column("target", width=125, minwidth=100)

        self.tree.heading("en", text="English")
        self.tree.column("en", width=120, minwidth=90)

        self.tree.heading("bn", text="Bengali")
        self.tree.column("bn", width=120, minwidth=90)

        self.tree.heading("mode", text="Mode")
        self.tree.column("mode", width=105, minwidth=80, anchor=tk.CENTER)

        self.tree.heading("category", text="Category")
        self.tree.column("category", width=110, minwidth=80, anchor=tk.CENTER)

        self.tree.heading("audio", text="Audio")
        self.tree.column("audio", width=65, minwidth=55, anchor=tk.CENTER)

        self.tree.heading("timestamp", text="Recorded At")
        self.tree.column("timestamp", width=130, minwidth=100, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click to play audio immediately
        self.tree.bind("<Double-1>", lambda e: self._play_selected_archive_audio())

        # Control Bar
        ctrl_bar = tk.Frame(frame, bg=self.BG_FIELD, padx=10, pady=8)
        ctrl_bar.pack(fill=tk.X, padx=4, pady=(4, 0))

        btn_play = tk.Button(ctrl_bar, text="▶ PLAY AUDIO", font=("Segoe UI", 10, "bold"),
                             bg=self.ACCENT_BLUE, fg="white", activebackground=self.ACCENT_CYAN,
                             relief=tk.FLAT, padx=10, pady=4,
                             command=self._play_selected_archive_audio)
        btn_play.pack(side=tk.LEFT, padx=(0, 6))

        btn_stop = tk.Button(ctrl_bar, text="⏹ STOP", font=("Segoe UI", 10, "bold"),
                             bg="#475569", fg="white", activebackground="#64748b",
                             relief=tk.FLAT, padx=8, pady=4,
                             command=self._stop_archive_audio)
        btn_stop.pack(side=tk.LEFT, padx=(0, 6))

        btn_refresh = tk.Button(ctrl_bar, text="🔄 REFRESH", font=("Segoe UI", 10, "bold"),
                                bg="#334155", fg="white", activebackground="#475569",
                                relief=tk.FLAT, padx=8, pady=4,
                                command=self._refresh_archive_tree)
        btn_refresh.pack(side=tk.LEFT, padx=(0, 6))

        btn_delete = tk.Button(ctrl_bar, text="🗑 DELETE", font=("Segoe UI", 10, "bold"),
                               bg="#991b1b", fg="white", activebackground=self.ACCENT_RED,
                               relief=tk.FLAT, padx=8, pady=4,
                               command=self._delete_selected_archive_entry)
        btn_delete.pack(side=tk.LEFT, padx=(0, 6))

        self.archive_status_lbl = tk.Label(ctrl_bar, text="Select an entry to play audio or view details.",
                                           font=("Segoe UI", 9),
                                           fg=self.FG_MUTED, bg=self.BG_FIELD)
        self.archive_status_lbl.pack(side=tk.RIGHT, padx=6)

        # Populate rows
        self._refresh_archive_tree()

    def _refresh_archive_tree(self):
        """Fetch all records from SQLite and populate the treeview."""
        if not hasattr(self, 'tree') or self.tree is None:
            return

        self._stop_archive_audio()

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT id, target_word, translation_en, translation_bn, mode, category, audio_file, timestamp 
                FROM archive 
                ORDER BY id DESC
            """)
            self._archive_all_rows = c.fetchall()
            conn.close()
        except Exception as e:
            print(f"[DB ERROR] {e}")
            self._archive_all_rows = []

        self._filter_archive_tree()
        self._update_db_count()

    def _filter_archive_tree(self):
        """Filter the displayed rows based on the search box query."""
        if not hasattr(self, 'tree') or self.tree is None:
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        query = self.archive_search_var.get().strip().lower() if hasattr(self, 'archive_search_var') else ""

        count = 0
        rows = getattr(self, '_archive_all_rows', [])
        for row in rows:
            entry_id, target, en, bn, mode, cat, audio_path, tstamp = row
            tstamp_clean = str(tstamp)[:19] if tstamp else "—"
            audio_indicator = "🔊 Yes" if (audio_path and os.path.exists(audio_path)) else "—"

            searchable = f"{entry_id} {target} {en} {bn} {mode} {cat}".lower()
            if not query or query in searchable:
                self.tree.insert("", tk.END, values=(
                    entry_id, target, en, bn, mode, cat, audio_indicator, tstamp_clean
                ))
                count += 1

        if hasattr(self, 'archive_count_lbl'):
            total = len(rows)
            if query:
                self.archive_count_lbl.config(text=f"Showing: {count} of {total} records")
            else:
                self.archive_count_lbl.config(text=f"Total: {total} records preserved")

    def _play_selected_archive_audio(self):
        """Play the recorded audio of the currently selected row."""
        if not hasattr(self, 'tree') or self.tree is None:
            return
        sel = self.tree.selection()
        if not sel:
            if hasattr(self, 'archive_status_lbl'):
                self.archive_status_lbl.config(text="⚠ Select an entry first.", fg=self.ACCENT_AMBER)
            return

        entry_id = self.tree.item(sel[0])['values'][0]
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT audio_file, target_word FROM archive WHERE id=?", (entry_id,))
            row = c.fetchone()
            conn.close()

            if row:
                audio_path, target_word = row[0], row[1]
                if audio_path and os.path.exists(audio_path):
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    if hasattr(self, 'archive_status_lbl'):
                        self.archive_status_lbl.config(
                            text=f"▶ Playing audio for '{target_word}'...",
                            fg=self.ACCENT_GREEN
                        )
                else:
                    if hasattr(self, 'archive_status_lbl'):
                        self.archive_status_lbl.config(
                            text=f"No audio file saved for '{target_word}'.",
                            fg=self.FG_MUTED
                        )
                    messagebox.showinfo("Audio", f"No audio sample available for '{target_word}'.")
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
            if hasattr(self, 'archive_status_lbl'):
                self.archive_status_lbl.config(text=f"Audio error: {e}", fg=self.ACCENT_RED)

    def _stop_archive_audio(self):
        """Stop any playing audio in archive showcase."""
        try:
            pygame.mixer.music.stop()
            if hasattr(self, 'archive_status_lbl'):
                self.archive_status_lbl.config(text="Audio playback stopped.", fg=self.FG_MUTED)
        except Exception:
            pass

    def _delete_selected_archive_entry(self):
        """Delete selected row from archive database with confirmation."""
        if not hasattr(self, 'tree') or self.tree is None:
            return
        sel = self.tree.selection()
        if not sel:
            if hasattr(self, 'archive_status_lbl'):
                self.archive_status_lbl.config(text="⚠ Select an entry to delete.", fg=self.ACCENT_AMBER)
            return

        values = self.tree.item(sel[0])['values']
        entry_id = values[0]
        target_word = values[1]

        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{target_word}' (ID #{entry_id}) from the archive?"):
            return

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM archive WHERE id=?", (entry_id,))
            conn.commit()
            conn.close()

            if hasattr(self, 'archive_status_lbl'):
                self.archive_status_lbl.config(text=f"Deleted entry #{entry_id} ('{target_word}').", fg=self.ACCENT_GREEN)
            self._refresh_archive_tree()
        except Exception as e:
            print(f"[DELETE ERROR] {e}")
            messagebox.showerror("Error", f"Failed to delete entry: {e}")

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
        if hasattr(self, 'tree') and self.tree is not None:
            try:
                if self.tree.winfo_exists():
                    self._refresh_archive_tree()
            except Exception:
                pass

    # ---------------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------------
    def on_closing(self):
        self._stop_prompt_animation()
        self._stop_archive_audio()
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
