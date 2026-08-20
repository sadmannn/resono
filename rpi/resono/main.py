import os
import sys
import time
import math
import json
import sqlite3
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

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
# Paths to the NCNN FP16 model files exported from training.
# All files live in the same directory as main.py.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NCNN_PARAM  = os.path.join(SCRIPT_DIR, "resono_mobilenetv3_large_fp16.ncnn.param")
NCNN_BIN    = os.path.join(SCRIPT_DIR, "resono_mobilenetv3_large_fp16.ncnn.bin")
LABELS_FILE = os.path.join(SCRIPT_DIR, "labels.json")

# Input / output blob names (from the PNNX-exported .param file).
# Verify by opening the .param in a text editor or Netron if inference fails.
NCNN_INPUT_BLOB  = "in0"
NCNN_OUTPUT_BLOB = "out0"

# Santali dictionary suggestions  (keys match the training class labels)
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
    
    Requires:
      model/resono_mobilenetv3_large_fp16.ncnn.param
      model/resono_mobilenetv3_large_fp16.ncnn.bin
      model/labels.json   (list like ["glasses","mug",…])
    """

    # ImageNet normalisation (same values used during training)
    _MEAN = [123.675, 116.28, 103.53]        # RGB pixel means (0-255 scale)
    _NORM = [1/58.395, 1/57.12, 1/57.375]    # 1 / std  (0-255 scale)
    _INPUT_SIZE = 224

    def __init__(self):
        # --- Load class labels ---------------------------------------------------
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "classes" in data:
                    # Rich metadata format: {"classes": ["glasses", ...], "class_to_idx": {...}, ...}
                    self.classes = list(data["classes"])
                elif isinstance(data, dict):
                    # Flat dict: {"glasses": 0, ...} or {"0": "glasses", ...}
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
            # Fallback: hard-coded order from training output
            print(f"[WARN] {LABELS_FILE} not found — using hard-coded class order.")
            self.classes = ["glasses", "mug", "pen", "phone", "spoon", "watch", "water_bottle"]

        self.num_classes = len(self.classes)
        print(f"[AI] Classes ({self.num_classes}): {self.classes}")

        # --- Load NCNN network ---------------------------------------------------
        if not os.path.exists(NCNN_PARAM) or not os.path.exists(NCNN_BIN):
            raise FileNotFoundError(
                f"NCNN model files not found.\n"
                f"  Expected param : {NCNN_PARAM}\n"
                f"  Expected bin   : {NCNN_BIN}\n"
                f"Copy the exported model files into the same folder as main.py."
            )

        self.net = ncnn.Net()
        # Use all available CPU cores on the Pi 5
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
        # Resize to model input size
        resized = cv2.resize(crop_cv2, (self._INPUT_SIZE, self._INPUT_SIZE))

        # Create ncnn Mat from BGR pixel data and resize
        mat_in = ncnn.Mat.from_pixels(
            resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self._INPUT_SIZE, self._INPUT_SIZE
        )

        # Apply ImageNet normalisation (mean subtraction + scaling)
        mat_in.substract_mean_normalize(self._MEAN, self._NORM)

        # Run forward pass
        ex = self.net.create_extractor()
        ex.input(NCNN_INPUT_BLOB, mat_in)
        ret, mat_out = ex.extract(NCNN_OUTPUT_BLOB)

        # Convert to numpy and apply softmax
        logits = np.array(mat_out).flatten()
        probs = self._softmax(logits)

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        raw_label = self.classes[idx]              # e.g. "water_bottle"
        clean_label = raw_label.upper()            # e.g. "WATER_BOTTLE"

        return clean_label, conf, raw_label

# ============================================================
# ST7789 EVE EYES DISPLAY ENGINE
# ============================================================
class EyesDisplayController:
    def __init__(self):
        self.width = 320
        self.height = 170
        self.x_offset = 0
        self.y_offset = 35
        self.dc_pin = 25
        self.rst_pin = 24
        self.bl_pin = 18
        self.state = "normal"
        self.running = True
        
        # FIX 1: Initialize active early to prevent AttributeError during init_display()
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

                # FIX 1: Set active to True BEFORE calling init_display so command() works
                self.active = True
                self.init_display()
            except Exception as e:
                print(f"[DISPLAY ERROR] {e}")
                self.active = False
        else:
            self.active = False

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
        if not self.active: return
        self.command(0x2A); self.data([(self.x_offset >> 8) & 0xFF, self.x_offset & 0xFF, ((self.x_offset + self.width - 1) >> 8) & 0xFF, (self.x_offset + self.width - 1) & 0xFF])
        self.command(0x2B); self.data([(self.y_offset >> 8) & 0xFF, self.y_offset & 0xFF, ((self.y_offset + self.height - 1) >> 8) & 0xFF, (self.y_offset + self.height - 1) & 0xFF])
        self.command(0x2C)
        
        pixels = image.load()
        buffer = bytearray()
        for y in range(image.height):
            for x in range(image.width):
                r, g, b = pixels[x, y]
                pixel = (((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3))
                buffer.append((pixel >> 8) & 0xFF)
                buffer.append(pixel & 0xFF)
        lgpio.gpio_write(self.chip, self.dc_pin, 1)
        for i in range(0, len(buffer), 4096):
            self.spi.writebytes(buffer[i:i + 4096])

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
            color = (56, 189, 248, 255) # Cyan
        elif self.state == "listening":
            color = (239, 68, 68, 255) # Red recording

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
        if not self.active: return
        while self.running:
            self.render_eyes(look=-0.8); time.sleep(0.4)
            self.render_eyes(look=0.0); time.sleep(0.5)
            self.render_eyes(look=0.0, blink=1.0); time.sleep(0.08)
            self.render_eyes(look=0.0, blink=0.0); time.sleep(0.7)
            self.render_eyes(look=0.8); time.sleep(0.4)
            self.render_eyes(look=0.0); time.sleep(0.5)

    def close(self):
        self.running = False
        if self.active:
            try:
                lgpio.gpio_write(self.chip, self.bl_pin, 0)
                self.spi.close()
                lgpio.gpiochip_close(self.chip)
            except: pass

# ============================================================
# MAIN RESONO GUI
# ============================================================
class ResonoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RESONO — Indigenous Language Robotic Archivist (WRO 2026)")
        self.root.geometry("1160x700")
        self.root.configure(bg="#0b1120")

        self.classifier = PretrainedClassifier()
        self.eyes = EyesDisplayController()
        threading.Thread(target=self.eyes.animation_loop, daemon=True).start()

        self.last_clean_class = "Background / Idle"
        self.last_confidence = 0.0

        self.is_recording = False
        self.audio_frames = []
        
        # FIX 2: Changed to 48000. Linux/Pi audio drivers natively prefer 48kHz over 44.1kHz
        self.sample_rate = 48000 
        self.latest_recorded_audio = None

        self.setup_ui()
        self.cap = cv2.VideoCapture(0)
        self.update_video_stream()

    def setup_ui(self):
        # 1. HEADER
        head = tk.Frame(self.root, bg="#1e293b", height=54, padx=16)
        head.pack(side=tk.TOP, fill=tk.X)

        tk.Label(head, text="RESONO ARCHIVIST", font=("Segoe UI", 15, "bold"), fg="#38bdf8", bg="#1e293b").pack(side=tk.LEFT, pady=8)
        tk.Label(head, text="PILOT: SANTALI (Northern Bangladesh) | GREETING: 'JOHAR'", 
                 font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#064e3b", padx=8, pady=3).pack(side=tk.LEFT, padx=16)

        self.db_counter_lbl = tk.Label(head, text="Words Preserved: 4", font=("Segoe UI", 11), fg="#94a3b8", bg="#1e293b")
        self.db_counter_lbl.pack(side=tk.RIGHT, pady=8)

        # 2. BODY LAYOUT
        body = tk.Frame(self.root, bg="#0b1120", padx=14, pady=14)
        body.pack(fill=tk.BOTH, expand=True)

        # LEFT: CAMERA & HUD
        left_box = tk.Frame(body, bg="#1e293b", width=420, bd=1, relief=tk.SOLID)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 12))

        tk.Label(left_box, text="LIVE CHEST CAMERA FEED", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b").pack(pady=6)
        self.cam_label = tk.Label(left_box, bg="#0f172a")
        self.cam_label.pack(padx=8, pady=2)

        hud = tk.Frame(left_box, bg="#0f172a", padx=10, pady=8)
        hud.pack(fill=tk.X, padx=12, pady=8)

        self.pred_lbl = tk.Label(hud, text="Class: Background / Idle", font=("Segoe UI", 12, "bold"), fg="#94a3b8", bg="#0f172a")
        self.pred_lbl.pack(anchor=tk.W)

        self.conf_lbl = tk.Label(hud, text="Confidence: 0%", font=("Segoe UI", 10), fg="#38bdf8", bg="#0f172a")
        self.conf_lbl.pack(anchor=tk.W)

        self.btn_trigger_obj = tk.Button(left_box, text="⚡ ELICIT DETECTED OBJECT", font=("Segoe UI", 11, "bold"),
                                         bg="#0284c7", fg="white", activebackground="#38bdf8", command=self.trigger_object_elicitation)
        self.btn_trigger_obj.pack(fill=tk.X, padx=14, pady=(0, 10))

        # RIGHT: NOTEBOOK TABS
        right_box = tk.Frame(body, bg="#1e293b")
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background="#1e293b", borderwidth=0)
        style.configure("TNotebook.Tab", background="#334155", foreground="#f8fafc", padding=[12, 7], font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#0284c7")], foreground=[("selected", "white")])

        self.nb = ttk.Notebook(right_box)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_p = tk.Frame(self.nb, bg="#1e293b", padx=14, pady=14)
        self.tab_o = tk.Frame(self.nb, bg="#1e293b", padx=14, pady=14)
        self.tab_c = tk.Frame(self.nb, bg="#1e293b", padx=14, pady=14)
        self.tab_d = tk.Frame(self.nb, bg="#1e293b", padx=14, pady=14)

        self.nb.add(self.tab_p, text=" 1. Prompt Mode ")
        self.nb.add(self.tab_o, text=" 2. Object Mode ")
        self.nb.add(self.tab_c, text=" 3. Conversation Mode ")
        self.nb.add(self.tab_d, text=" 4. Archive Showcase ")

        self.build_prompt_tab()
        self.build_object_tab()
        self.build_conversation_tab()
        self.build_archive_tab()

    def build_prompt_tab(self):
        tk.Label(self.tab_p, text="Structured Elicitation: Visual Prompt $\\rightarrow$ Record Pronunciation $\\rightarrow$ Bilingual Tag", 
                 font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b").pack(anchor=tk.W, pady=(0, 8))

        f = tk.Frame(self.tab_p, bg="#1e293b")
        f.pack(fill=tk.X)

        tk.Label(f, text="Target Concept:", fg="white", bg="#1e293b", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.p_concept = tk.Entry(f, width=28, bg="#334155", fg="white"); self.p_concept.insert(0, "Drinking Water"); self.p_concept.grid(row=0, column=1, padx=6, pady=3)

        tk.Label(f, text="Santali Target Word:", fg="#38bdf8", bg="#1e293b", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=3)
        self.p_santali = tk.Entry(f, width=28, bg="#334155", fg="#38bdf8"); self.p_santali.insert(0, "Dak'"); self.p_santali.grid(row=1, column=1, padx=6, pady=3)

        tk.Label(f, text="English Translation:", fg="white", bg="#1e293b", font=("Segoe UI", 9)).grid(row=2, column=0, sticky=tk.W, pady=3)
        self.p_en = tk.Entry(f, width=28, bg="#334155", fg="white"); self.p_en.insert(0, "Water"); self.p_en.grid(row=2, column=1, padx=6, pady=3)

        tk.Label(f, text="Bengali Translation:", fg="white", bg="#1e293b", font=("Segoe UI", 9)).grid(row=3, column=0, sticky=tk.W, pady=3)
        self.p_bn = tk.Entry(f, width=28, bg="#334155", fg="white"); self.p_bn.insert(0, "পানি / জল"); self.p_bn.grid(row=3, column=1, padx=6, pady=3)

        abox = tk.Frame(self.tab_p, bg="#334155", padx=8, pady=8)
        abox.pack(fill=tk.X, pady=10)

        self.btn_p_rec = tk.Button(abox, text="🎤 RECORD ELDER AUDIO", bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold"), command=self.toggle_record)
        self.btn_p_rec.pack(side=tk.LEFT, padx=4)

        tk.Button(abox, text="▶ VERIFY", bg="#475569", fg="white", font=("Segoe UI", 9, "bold"), command=self.play_latest_audio).pack(side=tk.LEFT, padx=4)

        tk.Button(self.tab_p, text="💾 CONFIRM & SAVE TO DATABASE", bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"),
                  command=lambda: self.save_word(self.p_santali.get(), self.p_en.get(), self.p_bn.get(), "Prompt Mode", "Basic Concept")).pack(fill=tk.X, pady=6)

    def build_object_tab(self):
        tk.Label(self.tab_o, text="Point real object in front of camera guide box $\\rightarrow$ auto-identifies category.", 
                 font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b").pack(anchor=tk.W, pady=(0, 6))

        self.obj_status_box = tk.Label(self.tab_o, text="Hold Bottle / Cup / Phone in Center Box", font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#0f172a", pady=10)
        self.obj_status_box.pack(fill=tk.X, pady=4)

        f = tk.Frame(self.tab_o, bg="#1e293b")
        f.pack(fill=tk.X, pady=6)

        tk.Label(f, text="Detected Object:", fg="white", bg="#1e293b", font=("Segoe UI", 9)).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.o_en = tk.Entry(f, width=28, bg="#334155", fg="white"); self.o_en.grid(row=0, column=1, padx=6, pady=3)

        tk.Label(f, text="Santali Term for Object:", fg="#38bdf8", bg="#1e293b", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=3)
        self.o_santali = tk.Entry(f, width=28, bg="#334155", fg="#38bdf8"); self.o_santali.grid(row=1, column=1, padx=6, pady=3)

        abox = tk.Frame(self.tab_o, bg="#334155", padx=8, pady=8)
        abox.pack(fill=tk.X, pady=8)

        self.btn_o_rec = tk.Button(abox, text="🎤 RECORD OBJECT NAME", bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold"), command=self.toggle_record)
        self.btn_o_rec.pack(side=tk.LEFT, padx=4)

        tk.Button(abox, text="▶ PLAYBACK", bg="#475569", fg="white", font=("Segoe UI", 9, "bold"), command=self.play_latest_audio).pack(side=tk.LEFT, padx=4)

        tk.Button(self.tab_o, text="💾 SAVE OBJECT WORD", bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"),
                  command=lambda: self.save_word(self.o_santali.get(), self.o_en.get(), "", "Object Mode", "Everyday Object")).pack(fill=tk.X, pady=6)

    def build_conversation_tab(self):
        tk.Label(self.tab_c, text="Narrative Capture: Long-form oral folklore & story recording with term extraction.", 
                 font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b").pack(anchor=tk.W, pady=(0, 6))

        cbox = tk.Frame(self.tab_c, bg="#334155", padx=8, pady=10)
        cbox.pack(fill=tk.X, pady=6)

        self.btn_c_rec = tk.Button(cbox, text="🔴 START NARRATIVE SESSION", bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold"), command=self.toggle_record)
        self.btn_c_rec.pack(side=tk.LEFT, padx=4)

        tk.Label(self.tab_c, text="Flag Word to Queue for Verification:", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#1e293b").pack(anchor=tk.W, pady=(10, 2))
        self.c_flag = tk.Entry(self.tab_c, bg="#334155", fg="white"); self.c_flag.pack(fill=tk.X, pady=2)

        tk.Button(self.tab_c, text="📌 QUEUE WORD INTO PROMPT MODE", bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"),
                  command=lambda: messagebox.showinfo("Queued", f"'{self.c_flag.get()}' queued for structured capture session.")).pack(fill=tk.X, pady=6)

    def build_archive_tab(self):
        cols = ("id", "target", "en", "bn", "mode", "category")
        self.tree = ttk.Treeview(self.tab_d, columns=cols, show="headings", height=8)
        self.tree.heading("id", text="ID"); self.tree.column("id", width=30, anchor=tk.CENTER)
        self.tree.heading("target", text="Santali"); self.tree.column("target", width=100)
        self.tree.heading("en", text="English"); self.tree.column("en", width=110)
        self.tree.heading("bn", text="Bengali"); self.tree.column("bn", width=110)
        self.tree.heading("mode", text="Mode"); self.tree.column("mode", width=85, anchor=tk.CENTER)
        self.tree.heading("category", text="Category"); self.tree.column("category", width=95, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)

        bar = tk.Frame(self.tab_d, bg="#1e293b")
        bar.pack(fill=tk.X, pady=4)
        tk.Button(bar, text="▶ PLAY SELECTED AUDIO", bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"), command=self.play_selected_db).pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text="🔄 REFRESH", bg="#475569", fg="white", font=("Segoe UI", 9, "bold"), command=self.refresh_db).pack(side=tk.LEFT, padx=3)
        self.refresh_db()

    # -------------------------------------------------------------
    # CAMERA STREAM & CLASSIFICATION LOOP
    # -------------------------------------------------------------
    def update_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            box_size = 224
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            roi = frame[y1:y2, x1:x2]
            clean_label, conf, raw_label = self.classifier.classify_crop(roi)

            self.last_clean_class = clean_label
            self.last_confidence = conf

            # Confidence threshold — treat low-confidence predictions as idle
            CONF_THRESHOLD = 0.45
            is_detected = conf >= CONF_THRESHOLD

            box_color = (56, 189, 248) if is_detected else (100, 116, 139)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, "HOLD OBJECT HERE", (x1 + 18, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            if is_detected:
                display_name = clean_label.replace("_", " ").title()
                self.pred_lbl.config(text=f"Object: {display_name}", fg="#10b981")
                self.conf_lbl.config(text=f"Confidence: {conf*100:.1f}% ({raw_label})", fg="#10b981")
                self.obj_status_box.config(text=f"Detected: {display_name} → Ready", fg="#10b981")
                self.eyes.state = "detect"
            else:
                self.pred_lbl.config(text="Object: Background / Idle", fg="#94a3b8")
                self.conf_lbl.config(text=f"Raw: {raw_label} ({conf*100:.0f}%)", fg="#94a3b8")
                if self.eyes.state == "detect":
                    self.eyes.state = "normal"

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (360, 250))
            img = ImageTk.PhotoImage(image=Image.fromarray(rgb_small))
            self.cam_label.imgtk = img
            self.cam_label.configure(image=img)

        self.root.after(25, self.update_video_stream)

    def trigger_object_elicitation(self):
        lbl = self.last_clean_class
        conf = self.last_confidence
        if conf < 0.45:
            messagebox.showinfo("RESONO Vision", "Hold an object (bottle, mug, phone, watch, pen, glasses, spoon) in the square.")
            return

        self.nb.select(self.tab_o)
        self.o_en.delete(0, tk.END)
        self.o_en.insert(0, lbl.replace("_", " ").title())

        self.o_santali.delete(0, tk.END)
        if lbl in SANTALI_OBJECT_MAP:
            self.o_santali.insert(0, SANTALI_OBJECT_MAP[lbl])

    # -------------------------------------------------------------
    # AUDIO CAPTURE
    # -------------------------------------------------------------
    def toggle_record(self):
        if not self.is_recording:
            self.is_recording = True
            self.audio_frames = []
            self.eyes.state = "listening"
            self.btn_p_rec.config(text="⏹ STOP", bg="#b91c1c")
            self.btn_o_rec.config(text="⏹ STOP", bg="#b91c1c")
            self.btn_c_rec.config(text="⏹ STOP", bg="#b91c1c")

            def rec():
                try:
                    with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                                        callback=lambda data, f, t, s: self.audio_frames.append(data.copy())):
                        while self.is_recording:
                            sd.sleep(100)
                except Exception as e:
                    print(f"[AUDIO ERROR] {e}")
                    self.is_recording = False
                    
            threading.Thread(target=rec, daemon=True).start()
        else:
            self.is_recording = False
            self.eyes.state = "normal"
            self.btn_p_rec.config(text="🎤 RECORD ELDER AUDIO", bg="#ef4444")
            self.btn_o_rec.config(text="🎤 RECORD OBJECT NAME", bg="#ef4444")
            self.btn_c_rec.config(text="🔴 START NARRATIVE SESSION", bg="#ef4444")

            if self.audio_frames:
                data = np.concatenate(self.audio_frames, axis=0)
                path = os.path.join(AUDIO_DIR, f"rec_{int(time.time())}.wav")
                wav_write(path, self.sample_rate, data)
                self.latest_recorded_audio = path
                messagebox.showinfo("Saved", "Audio captured cleanly!")

    def play_latest_audio(self):
        if self.latest_recorded_audio and os.path.exists(self.latest_recorded_audio):
            pygame.mixer.music.load(self.latest_recorded_audio)
            pygame.mixer.music.play()
        else:
            messagebox.showinfo("Audio", "Record a sample first.")

    # -------------------------------------------------------------
    # DATABASE OPS
    # -------------------------------------------------------------
    def save_word(self, target, en, bn, mode, cat):
        if not target:
            messagebox.showwarning("Missing", "Enter the native Santali word.")
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO archive (target_word, translation_en, translation_bn, category, mode, audio_file) VALUES (?, ?, ?, ?, ?, ?)",
                  (target, en, bn, cat, mode, self.latest_recorded_audio or ""))
        conn.commit()
        conn.close()
        messagebox.showinfo("Preserved", f"Saved '{target}' to local archive!")
        self.refresh_db()

    def refresh_db(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, target_word, translation_en, translation_bn, mode, category FROM archive ORDER BY id DESC")
        rows = c.fetchall()
        for r in rows: self.tree.insert("", tk.END, values=r)
        self.db_counter_lbl.config(text=f"Words Preserved: {len(rows)}")
        conn.close()

    def play_selected_db(self):
        sel = self.tree.selection()
        if not sel: return
        entry_id = self.tree.item(sel[0])['values'][0]
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT audio_file, target_word FROM archive WHERE id=?", (entry_id,))
        row = c.fetchone()
        conn.close()
        if row and row[0] and os.path.exists(row[0]):
            pygame.mixer.music.load(row[0]); pygame.mixer.music.play()
        else:
            messagebox.showinfo("Audio", f"Audio sample for '{row[1]}'")

    def on_closing(self):
        self.cap.release()
        self.eyes.close()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = ResonoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
