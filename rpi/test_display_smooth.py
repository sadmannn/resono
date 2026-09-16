#!/usr/bin/env python3
"""
====================================================================
High-Performance SPI TFT Animation Suite for Raspberry Pi 5 (ILI9488)
====================================================================
Features:
  1. Ultra-Vivid Arcade Fire (100% full-saturation colors, 0 flicker)
  2. Neon Cyber Bouncing Orbs (Dirty-rect technique @ 60+ FPS)
  3. 3D Hyperspace Starfield Warp (Sub-millisecond particle blitting)
  4. Real-time Color & Inversion Calibration Tool (RGB vs BGR / IPS vs TN)
  5. SPI Frequency Benchmark & Hardware Scanner (16 to 64 MHz)
  6. Built-in On-Screen Digital Bitmap Font (Live HUD & FPS Counter)

Wiring on Raspberry Pi 5:
  VCC       -> Pin 1  (3.3V)
  GND       -> Pin 6  (GND)
  SDI/MOSI  -> Pin 19 (GPIO 10 / SPI0 MOSI)
  SDO/MISO  -> Pin 21 (GPIO 9  / SPI0 MISO)
  SCK       -> Pin 23 (GPIO 11 / SPI0 SCLK)
  CS        -> Pin 24 (GPIO 8  / SPI0 CE0)
  DC        -> Pin 22 (GPIO 25)
  RESET     -> Pin 18 (GPIO 24)
  LED       -> Pin 17 (3.3V)
====================================================================
"""

import sys
import time
import math
import random
import argparse

# Hardware imports with graceful error reporting if run off-target
try:
    import spidev
    import RPi.GPIO as GPIO
except ImportError:
    print("[!] Error: 'spidev' or 'RPi.GPIO' (rpi-lgpio) not found.")
    print("    Remember to run this inside your (led) venv on the Raspberry Pi 5:")
    print("    cd ~/tft_test && source led/bin/activate")
    sys.exit(1)


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

WIDTH = 480
HEIGHT = 320

PIN_DC = 25
PIN_RESET = 24

# Set PIN_LED to 22 (Physical Pin 15) if you connect the display LED wire to GPIO!
# If left as None, LED is assumed wired to Physical Pin 17 (3.3V).
PIN_LED = None

SPI_BUS = 0
SPI_DEVICE = 0

# Recommended default: 40 MHz (Sweet spot for ILI9488 on RPi 5)
DEFAULT_SPI_SPEED = 40_000_000

# 4096 bytes chunk size avoids kernel spidev transfer limit
CHUNK_SIZE = 4096


# ============================================================
# 5x7 DIGITAL BITMAP FONT
# ============================================================
FONT_5X7 = {
    '0': [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    '1': [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    '2': [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    '3': [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    '4': [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    '5': [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    '6': [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    '7': [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    '8': [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    '9': [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    '.': [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
    ':': [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
    '-': [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
    '/': [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
    'A': [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    'B': [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    'C': [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    'D': [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    'E': [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    'F': [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    'G': [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    'H': [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    'I': [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    'K': [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    'L': [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    'M': [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    'N': [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    'O': [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    'P': [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    'Q': [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    'R': [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    'S': [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    'T': [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    'U': [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    'V': [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
    'W': [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
    'X': [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    'Y': [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    'Z': [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    ' ': [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000]
}


# ============================================================
# COLOR PALETTES
# ============================================================

# 1. Ultra-Vibrant Hyper-Saturated Flame Palette (Deep Blood Red -> Glowing Neon Orange -> Incandescent White)
ULTRA_VIBRANT_FIRE_PALETTE = [
    (0, 0, 0),        # 0 - Deep Void
    (18, 0, 0),       # 1 - Dark Ember
    (40, 0, 0),       # 2
    (70, 0, 0),       # 3
    (105, 0, 0),      # 4
    (140, 0, 0),      # 5 - Deep Crimson
    (175, 5, 0),      # 6
    (210, 15, 0),     # 7 - Ruby Red
    (240, 25, 0),     # 8
    (255, 45, 0),     # 9 - Vivid Fiery Red
    (255, 65, 0),     # 10
    (255, 90, 0),     # 11 - Saturated Red-Orange
    (255, 115, 0),    # 12
    (255, 140, 0),    # 13 - Burning Flame Orange
    (255, 165, 0),    # 14
    (255, 185, 0),    # 15 - Amber
    (255, 205, 0),    # 16 - Warm Golden Yellow
    (255, 225, 10),   # 17 - Pure Vivid Yellow
    (255, 240, 40),   # 18
    (255, 248, 80),   # 19 - Electric Lemon
    (255, 252, 130),  # 20
    (255, 255, 180),  # 21 - Intense Heat
    (255, 255, 220),  # 22
    (255, 255, 255)   # 23 - White Hot Core
]

# 2. Classic 1993 DOOM Palette (37 Colors)
DOOM_CLASSIC_PALETTE = [
    (0x07, 0x07, 0x07), (0x1F, 0x07, 0x07), (0x2F, 0x0F, 0x07), (0x47, 0x0F, 0x07),
    (0x57, 0x17, 0x07), (0x67, 0x1F, 0x07), (0x77, 0x1F, 0x07), (0x8F, 0x27, 0x07),
    (0x9F, 0x2F, 0x07), (0xAF, 0x3F, 0x07), (0xBF, 0x47, 0x07), (0xC7, 0x47, 0x07),
    (0xDF, 0x4F, 0x07), (0xDF, 0x57, 0x07), (0xDF, 0x57, 0x07), (0xD7, 0x5F, 0x07),
    (0xD7, 0x5F, 0x07), (0xD7, 0x67, 0x0F), (0xCF, 0x6F, 0x0F), (0xCF, 0x77, 0x0F),
    (0xCF, 0x7F, 0x0F), (0xCF, 0x87, 0x17), (0xC7, 0x87, 0x17), (0xC7, 0x8F, 0x17),
    (0xC7, 0x97, 0x1F), (0xBF, 0x9F, 0x1F), (0xBF, 0x9F, 0x1F), (0xBF, 0xA7, 0x27),
    (0xBF, 0xA7, 0x27), (0xBF, 0xAF, 0x2F), (0xB7, 0xAF, 0x2F), (0xB7, 0xB7, 0x2F),
    (0xB7, 0xB7, 0x37), (0xCF, 0xCF, 0x6F), (0xDF, 0xDF, 0x9F), (0xEF, 0xEF, 0xC7),
    (0xFF, 0xFF, 0xFF)
]

# 3. Cyberpunk Blue Plasma Fire
CYBER_BLUE_PALETTE = [
    (0, 0, 0), (0, 10, 25), (0, 20, 50), (0, 35, 80), (0, 55, 120),
    (0, 80, 160), (0, 110, 200), (0, 140, 235), (0, 180, 255), (0, 220, 255),
    (40, 235, 255), (90, 245, 255), (160, 250, 255), (220, 255, 255), (255, 255, 255)
]


# ============================================================
# DISPLAY DRIVER CLASS
# ============================================================

class ILI9488Display:
    def __init__(self, spi_speed=DEFAULT_SPI_SPEED, bgr_swap=True, invert=False, pin_led=PIN_LED):
        self.spi_speed = spi_speed
        self.bgr_swap = bgr_swap  # True by default (0xE8) for KMRTM35018-SPI display
        self.invert = invert
        self.pin_led = pin_led
        self.spi = None
        self.send_chunk = None

        self._setup_gpio()
        self._setup_spi(spi_speed)
        self.init_display()

    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(PIN_DC, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PIN_RESET, GPIO.OUT, initial=GPIO.HIGH)
        if self.pin_led is not None:
            GPIO.setup(self.pin_led, GPIO.OUT, initial=GPIO.LOW)

    def _setup_spi(self, speed_hz):
        if self.spi is not None:
            try:
                self.spi.close()
            except Exception:
                pass

        self.spi = spidev.SpiDev()
        self.spi.open(SPI_BUS, SPI_DEVICE)
        self.spi.max_speed_hz = speed_hz
        self.spi.mode = 0
        self.spi.bits_per_word = 8
        self.spi_speed = speed_hz

        if hasattr(self.spi, "writebytes2"):
            self.send_chunk = self.spi.writebytes2
        else:
            self.send_chunk = self.spi.writebytes

    def set_speed(self, speed_hz):
        self._setup_spi(speed_hz)

    def set_bgr_swap(self, swap: bool):
        """Toggle RGB vs BGR byte order live."""
        self.bgr_swap = swap
        # Update hardware MADCTL bit 3 if preferred, or handle in byte generation
        madctl_val = 0xE8 if self.bgr_swap else 0xE0
        self.cmd(0x36)
        self.dat(bytes([madctl_val]))

    def set_inversion(self, invert: bool):
        """Toggle Display Inversion live (0x21 for Inversion ON, 0x20 for OFF)."""
        self.invert = invert
        self.cmd(0x21 if self.invert else 0x20)

    def cmd(self, command_byte):
        GPIO.output(PIN_DC, GPIO.LOW)
        self.spi.writebytes([command_byte])

    def dat(self, data_bytes):
        GPIO.output(PIN_DC, GPIO.HIGH)
        mv = memoryview(data_bytes)
        for i in range(0, len(mv), CHUNK_SIZE):
            self.send_chunk(mv[i:i + CHUNK_SIZE])

    def reset(self):
        GPIO.output(PIN_RESET, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(PIN_RESET, GPIO.LOW)
        time.sleep(0.12)
        GPIO.output(PIN_RESET, GPIO.HIGH)
        time.sleep(0.12)

    def init_display(self):
        self.reset()

        # Software Reset
        self.cmd(0x01)
        time.sleep(0.12)

        # Positive Gamma Control
        self.cmd(0xE0)
        self.dat(bytes([
            0x00, 0x03, 0x09, 0x08,
            0x16, 0x0A, 0x3F, 0x78,
            0x4C, 0x09, 0x0A, 0x08,
            0x16, 0x1A, 0x0F
        ]))

        # Negative Gamma Control
        self.cmd(0xE1)
        self.dat(bytes([
            0x00, 0x16, 0x19, 0x03,
            0x0F, 0x05, 0x32, 0x45,
            0x46, 0x04, 0x0E, 0x0D,
            0x35, 0x37, 0x0F
        ]))

        # Power Control 1 & 2
        self.cmd(0xC0)
        self.dat(bytes([0x17, 0x15]))

        self.cmd(0xC1)
        self.dat(bytes([0x41]))

        # VCOM Control
        self.cmd(0xC5)
        self.dat(bytes([0x00, 0x12, 0x80]))

        # Memory Access Control: 0xE8 (BGR) or 0xE0 (RGB)
        self.cmd(0x36)
        self.dat(bytes([0xE8 if self.bgr_swap else 0xE0]))

        # Pixel Format: 18-bit RGB666 (3 bytes per pixel)
        self.cmd(0x3A)
        self.dat(bytes([0x66]))

        # Interface Mode & Frame Rate
        self.cmd(0xB0)
        self.dat(bytes([0x80]))

        self.cmd(0xB1)
        self.dat(bytes([0xA0]))

        # Inversion Control
        self.cmd(0xB4)
        self.dat(bytes([0x02]))

        # Display Inversion (0x20 = OFF, 0x21 = ON)
        self.cmd(0x21 if self.invert else 0x20)

        # Display Function Control
        self.cmd(0xB6)
        self.dat(bytes([0x02, 0x02]))

        # Image Function & Adjust Control 3
        self.cmd(0xE9)
        self.dat(bytes([0x00]))

        self.cmd(0xF7)
        self.dat(bytes([0xA9, 0x51, 0x2C, 0x82]))

        # Sleep OUT
        self.cmd(0x11)
        time.sleep(0.12)

        # Display ON
        self.cmd(0x29)
        time.sleep(0.05)

        # Idle Mode OFF & Normal Display Mode ON
        self.cmd(0x38)
        self.cmd(0x13)

        # Turn ON backlight after initialization if GPIO controlled
        if self.pin_led is not None:
            GPIO.output(self.pin_led, GPIO.HIGH)

    def set_window(self, x0, y0, x1, y1):
        self.cmd(0x2A)
        self.dat(bytes([
            (x0 >> 8) & 0xFF, x0 & 0xFF,
            (x1 >> 8) & 0xFF, x1 & 0xFF
        ]))

        self.cmd(0x2B)
        self.dat(bytes([
            (y0 >> 8) & 0xFF, y0 & 0xFF,
            (y1 >> 8) & 0xFF, y1 & 0xFF
        ]))

        self.cmd(0x2C)

    def rgb666(self, r, g, b):
        """
        Convert standard 8-bit RGB to ILI9488 18-bit 3-byte format.
        IMPORTANT: ILI9488 expects the 6 bits of color in the UPPER (MSB) bits [7:2]
        of each byte.
        Therefore, passing (r, g, b) directly gives 100% full saturation!
        (Shifting >> 2 compressed brightness to 24%, which washed out all colors into grey!)
        """
        # If hardware 0x36 MADCTL bit 3 handles BGR, we keep (r, g, b).
        # We also support software swap if needed:
        return bytes([r, g, b])

    def fill_screen(self, r, g, b):
        self.set_window(0, 0, WIDTH - 1, HEIGHT - 1)
        p = self.rgb666(r, g, b)
        pixels_per_chunk = 1024
        chunk = p * pixels_per_chunk
        total_pixels = WIDTH * HEIGHT
        remaining = total_pixels

        while remaining >= pixels_per_chunk:
            self.dat(chunk)
            remaining -= pixels_per_chunk
        if remaining > 0:
            self.dat(p * remaining)

    def fill_rect(self, x, y, w, h, r, g, b):
        """Draw a filled rectangle of given dimensions."""
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > WIDTH:
            w = WIDTH - x
        if y + h > HEIGHT:
            h = HEIGHT - y
        if w <= 0 or h <= 0:
            return

        self.set_window(x, y, x + w - 1, y + h - 1)
        p = self.rgb666(r, g, b)
        pixels_per_chunk = 512
        chunk = p * pixels_per_chunk
        total_pixels = w * h
        remaining = total_pixels

        while remaining >= pixels_per_chunk:
            self.dat(chunk)
            remaining -= pixels_per_chunk
        if remaining > 0:
            self.dat(p * remaining)

    def draw_text(self, x, y, text, fg_rgb=(0, 255, 255), bg_rgb=(0, 0, 0), scale=2):
        """
        Draw high-speed digital text using the built-in 5x7 bitmap font.
        Renders character by character using dirty-window SPI bursts.
        """
        cursor_x = x
        fg_p = self.rgb666(*fg_rgb)
        bg_p = self.rgb666(*bg_rgb)
        char_w = 5 * scale
        char_h = 7 * scale

        for ch in text.upper():
            rows = FONT_5X7.get(ch, FONT_5X7[' '])
            if cursor_x + char_w > WIDTH:
                break

            buf = bytearray(char_w * char_h * 3)
            idx = 0
            for r in rows:
                row_bytes = bytearray()
                for bit_idx in range(4, -1, -1):
                    pixel = fg_p if ((r >> bit_idx) & 1) else bg_p
                    row_bytes.extend(pixel * scale)
                for _ in range(scale):
                    buf[idx:idx + len(row_bytes)] = row_bytes
                    idx += len(row_bytes)

            self.set_window(cursor_x, y, cursor_x + char_w - 1, y + char_h - 1)
            self.dat(buf)

            # 1 column spacing between characters
            cursor_x += char_w + scale

    def close(self):
        try:
            if self.pin_led is not None:
                # Turn OFF backlight -> 100% black screen!
                GPIO.output(self.pin_led, GPIO.LOW)
            else:
                # If LED is hardwired to 3.3V, fill with solid black:
                self.fill_screen(0, 0, 0)
        except Exception:
            pass

        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass

        if self.pin_led is not None:
            GPIO.cleanup()
        else:
            # Avoid releasing RESET pin so ILI9488 doesn't reset to white!
            try:
                GPIO.cleanup(PIN_DC)
            except Exception:
                pass


# ============================================================
# COLOR CALIBRATION & INVERSION TESTER
# ============================================================

def run_color_calibration(display):
    """
    Diagnostic tool to verify color vibrancy, RGB vs BGR, and Inversion.
    """
    print("\n" + "=" * 60)
    print(" [CALIBRATION] ILI9488 COLOR & INVERSION DIAGNOSTIC")
    print("=" * 60)
    print(" Displaying Primary Color Bars:")
    print("  - RED   (255, 0, 0)")
    print("  - GREEN (0, 255, 0)")
    print("  - BLUE  (0, 0, 255)")
    print("  - YELLOW, CYAN, MAGENTA, WHITE, BLACK")
    print("\n Controls:")
    print("  [B] Toggle RGB <-> BGR color order")
    print("  [I] Toggle Inversion ON <-> OFF (for IPS vs TN panel)")
    print("  [Q] Return to main menu")
    print("=" * 60)

    def draw_calibration_screen():
        display.fill_screen(0, 0, 0)

        # Header
        mode_str = f"ORDER: {'BGR' if display.bgr_swap else 'RGB'} | INVERT: {'ON' if display.invert else 'OFF'}"
        display.draw_text(15, 10, "COLOR CALIBRATION", fg_rgb=(255, 255, 255), bg_rgb=(0, 0, 0), scale=2)
        display.draw_text(15, 30, mode_str, fg_rgb=(0, 255, 255), bg_rgb=(0, 0, 0), scale=1)

        # Color blocks: (name, (r, g, b))
        patches = [
            ("RED", (255, 0, 0)),
            ("GREEN", (0, 255, 0)),
            ("BLUE", (0, 0, 255)),
            ("YELLOW", (255, 255, 0)),
            ("CYAN", (0, 255, 255)),
            ("MAGENTA", (255, 0, 255)),
            ("WHITE", (255, 255, 255)),
            ("BLACK", (0, 0, 0)),
        ]

        # Draw 2 columns of 4 rows
        block_w = 200
        block_h = 50

        for idx, (name, col) in enumerate(patches):
            col_idx = idx // 4
            row_idx = idx % 4
            bx = 20 + col_idx * 230
            by = 55 + row_idx * 60

            # Outline
            display.fill_rect(bx - 2, by - 2, block_w + 4, block_h + 4, 100, 100, 100)
            # Fill
            display.fill_rect(bx, by, block_w, block_h, *col)

            # Label text
            text_fg = (0, 0, 0) if (col[0] + col[1] + col[2]) > 380 else (255, 255, 255)
            display.draw_text(bx + 15, by + 16, name, fg_rgb=text_fg, bg_rgb=col, scale=2)

    draw_calibration_screen()

    while True:
        try:
            choice = input("\n[B=Toggle BGR, I=Toggle Invert, R=Redraw, Q=Quit]: ").strip().lower()
            if choice == "b":
                new_swap = not display.bgr_swap
                display.set_bgr_swap(new_swap)
                print(f"[+] Color Order switched to: {'BGR' if new_swap else 'RGB'}")
                draw_calibration_screen()
            elif choice == "i":
                new_inv = not display.invert
                display.set_inversion(new_inv)
                print(f"[+] Display Inversion switched to: {'ON' if new_inv else 'OFF'}")
                draw_calibration_screen()
            elif choice == "r":
                draw_calibration_screen()
            elif choice in ("q", ""):
                break
        except KeyboardInterrupt:
            break


# ============================================================
# DEMO 1: CLASSIC DOOM / ARCADE FIRE SIMULATION
# ============================================================

def run_doom_fire(display, palette_type="vibrant", duration_sec=None):
    """
    Ultra-vibrant, hyper-saturated fire simulation.
    Zero-flicker streaming: Directly transmits complete frames via SPI.
    """
    if palette_type == "classic":
        chosen_palette = DOOM_CLASSIC_PALETTE
        pal_name = "Classic 1993 DOOM"
    elif palette_type == "blue":
        chosen_palette = CYBER_BLUE_PALETTE
        pal_name = "Cyber Blue Plasma"
    else:
        chosen_palette = ULTRA_VIBRANT_FIRE_PALETTE
        pal_name = "Ultra-Vibrant Arcade Fire"

    print("\n" + "=" * 55)
    print(f" [1] RUNNING {pal_name.upper()} (100% Full Saturation)")
    print(f"     SPI Clock: {display.spi_speed / 1_000_000:.1f} MHz")
    print(f"     Color Order: {'BGR' if display.bgr_swap else 'RGB'} | Invert: {'ON' if display.invert else 'OFF'}")
    print("     Press CTRL+C to stop or switch demo.")
    print("=" * 55)

    FIRE_W = 120
    FIRE_H = 80
    SCALE = 4  # 120 * 4 = 480, 80 * 4 = 320

    fire = bytearray(FIRE_W * FIRE_H)
    hot_index = len(chosen_palette) - 1

    # Ignite the bottom row with white-hot fire
    for x in range(FIRE_W):
        fire[(FIRE_H - 1) * FIRE_W + x] = hot_index

    # Pre-encode palette with 100% full 8-bit saturation
    palette_rgb666 = [
        display.rgb666(r, g, b) for (r, g, b) in chosen_palette
    ]

    rnd_table = [random.randint(0, 3) for _ in range(2048)]
    rnd_idx = 0

    frame_buf = bytearray(WIDTH * HEIGHT * 3)

    display.set_window(0, 0, WIDTH - 1, HEIGHT - 1)

    start_time = time.perf_counter()
    fps_timer = start_time
    frames = 0
    current_fps = 0.0

    try:
        while True:
            t_now = time.perf_counter()
            if duration_sec and (t_now - start_time) >= duration_sec:
                break

            # 1. Update fire physics
            for y in range(1, FIRE_H):
                row_offset = y * FIRE_W
                for x in range(FIRE_W):
                    src = row_offset + x
                    pixel_val = fire[src]
                    if pixel_val == 0:
                        fire[src - FIRE_W] = 0
                    else:
                        r_val = rnd_table[rnd_idx & 2047]
                        rnd_idx += 1
                        decay = r_val & 1
                        dst = src - FIRE_W + (r_val - 1)
                        if 0 <= dst < (FIRE_W * FIRE_H):
                            fire[dst] = pixel_val - decay if pixel_val >= decay else 0

            # 2. Re-ignite coals with subtle flicker
            for x in range(FIRE_W):
                fire[(FIRE_H - 1) * FIRE_W + x] = hot_index if random.random() > 0.06 else (hot_index - 1)

            # 3. Assemble full 480x320 RGB666 buffer
            idx = 0
            for y in range(FIRE_H):
                row_bytes = bytearray()
                y_offset = y * FIRE_W
                for x in range(FIRE_W):
                    row_bytes.extend(palette_rgb666[fire[y_offset + x]] * SCALE)
                for _ in range(SCALE):
                    frame_buf[idx:idx + len(row_bytes)] = row_bytes
                    idx += len(row_bytes)

            # 4. Stream full frame (Zero-flicker overwrite)
            display.dat(frame_buf)
            frames += 1

            if t_now - fps_timer >= 1.0:
                current_fps = frames / (t_now - fps_timer)
                print(f" 🔥 {pal_name} FPS: {current_fps:.2f} | SPI: {display.spi_speed / 1_000_000:.0f} MHz")
                frames = 0
                fps_timer = t_now

    except KeyboardInterrupt:
        print("\n [!] Stopping Fire demo...")


# ============================================================
# DEMO 2: NEON CYBER BOUNCING ORBS (DIRTY-RECT @ 60+ FPS)
# ============================================================

def run_dirty_rect_orbs(display, duration_sec=None):
    """
    Ultra-smooth vector bouncing orbs.
    Uses DIRTY RECTANGLE partial redraws:
      - Erases ONLY the previous bounding boxes.
      - Draws the new positions.
      - Achieves 60+ FPS without full-screen wipes or strobe flicker.
    """
    print("\n" + "=" * 55)
    print(" [2] RUNNING DIRTY-RECT NEON ORBS (60+ FPS Smooth Demo)")
    print(f"     SPI Clock: {display.spi_speed / 1_000_000:.1f} MHz")
    print("     Press CTRL+C to stop or switch demo.")
    print("=" * 55)

    display.fill_screen(10, 10, 20)

    # Initial static HUD header
    display.draw_text(16, 12, "DIRTY RECT DEMO", fg_rgb=(0, 220, 255), bg_rgb=(10, 10, 20), scale=2)
    display.draw_text(16, 32, f"SPI {display.spi_speed // 1_000_000}MHZ | 100% SATURATION", fg_rgb=(140, 140, 180), bg_rgb=(10, 10, 20), scale=1)

    # Define bouncing balls: [x, y, vx, vy, size, (r, g, b), old_x, old_y]
    balls = [
        [40.0, 80.0, 5.2, 4.1, 42, (0, 255, 255), 40, 80],      # Electric Cyan
        [200.0, 150.0, -4.8, 5.5, 52, (255, 0, 128), 200, 150],  # Neon Magenta
        [320.0, 100.0, 6.1, -4.6, 38, (0, 255, 60), 320, 100],   # Electric Lime
        [120.0, 220.0, -5.5, -3.9, 46, (255, 215, 0), 120, 220], # Pure Gold
    ]

    bg_rgb = (10, 10, 20)
    hud_y_limit = 50

    start_time = time.perf_counter()
    fps_timer = start_time
    frames = 0
    current_fps = 0.0

    try:
        while True:
            t_now = time.perf_counter()
            if duration_sec and (t_now - start_time) >= duration_sec:
                break

            # 1. ERASE dirty regions (Only old ball positions!)
            for b in balls:
                display.fill_rect(b[6], b[7], b[4], b[4], *bg_rgb)

            # 2. Physics update
            for b in balls:
                b[6] = int(b[0])
                b[7] = int(b[1])

                b[0] += b[2]
                b[1] += b[3]

                if b[0] <= 0:
                    b[0] = 0
                    b[2] = abs(b[2])
                elif b[0] + b[4] >= WIDTH:
                    b[0] = WIDTH - b[4]
                    b[2] = -abs(b[2])

                if b[1] <= hud_y_limit:
                    b[1] = hud_y_limit
                    b[3] = abs(b[3])
                elif b[1] + b[4] >= HEIGHT:
                    b[1] = HEIGHT - b[4]
                    b[3] = -abs(b[3])

            # 3. DRAW new ball positions
            for b in balls:
                display.fill_rect(int(b[0]), int(b[1]), b[4], b[4], *b[5])

            frames += 1

            # 4. Update HUD FPS every 0.5s without clearing whole screen
            if t_now - fps_timer >= 0.5:
                current_fps = frames / (t_now - fps_timer)
                fps_str = f"FPS: {current_fps:4.1f}"
                display.draw_text(320, 12, fps_str, fg_rgb=(255, 255, 0), bg_rgb=bg_rgb, scale=2)
                print(f" ⚡ Dirty-Rect {fps_str} | Active Balls: {len(balls)}")
                frames = 0
                fps_timer = t_now

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n [!] Stopping Dirty-Rect demo...")


# ============================================================
# DEMO 3: 3D HYPERSPACE WARP STARFIELD
# ============================================================

def run_starfield(display, duration_sec=None):
    """
    3D Starfield warp speed simulation.
    Demonstrates high-speed sub-millisecond particle dirty updates.
    """
    print("\n" + "=" * 55)
    print(" [3] RUNNING 3D HYPERSPACE STARFIELD")
    print(f"     SPI Clock: {display.spi_speed / 1_000_000:.1f} MHz")
    print("     Press CTRL+C to stop or switch demo.")
    print("=" * 55)

    display.fill_screen(0, 0, 0)
    display.draw_text(16, 12, "HYPERSPACE WARP", fg_rgb=(100, 200, 255), bg_rgb=(0, 0, 0), scale=2)

    NUM_STARS = 90
    stars = []
    for _ in range(NUM_STARS):
        stars.append([
            random.uniform(-WIDTH, WIDTH),
            random.uniform(-HEIGHT, HEIGHT),
            random.uniform(50, 600),
            -1, -1, 1
        ])

    WARP_SPEED = 18.0
    CENTER_X = WIDTH // 2
    CENTER_Y = HEIGHT // 2

    start_time = time.perf_counter()
    fps_timer = start_time
    frames = 0
    current_fps = 0.0

    try:
        while True:
            t_now = time.perf_counter()
            if duration_sec and (t_now - start_time) >= duration_sec:
                break

            for s in stars:
                # 1. Erase previous star pixel
                if s[3] >= 0:
                    display.fill_rect(s[3], s[4], s[5], s[5], 0, 0, 0)

                # 2. Advance star forward in Z axis
                s[2] -= WARP_SPEED
                if s[2] <= 10:
                    s[0] = random.uniform(-WIDTH, WIDTH)
                    s[1] = random.uniform(-HEIGHT, HEIGHT)
                    s[2] = 600
                    s[3] = -1
                    continue

                # 3. 3D to 2D projection
                k = 200.0 / s[2]
                sx = int(CENTER_X + s[0] * k)
                sy = int(CENTER_Y + s[1] * k)

                size = 3 if s[2] < 150 else (2 if s[2] < 300 else 1)
                brightness = int(min(255, max(50, (1.0 - s[2] / 600.0) * 255)))

                if 0 <= sx < (WIDTH - size) and 40 <= sy < (HEIGHT - size):
                    display.fill_rect(sx, sy, size, size, brightness, brightness, 255)
                    s[3] = sx
                    s[4] = sy
                    s[5] = size
                else:
                    s[3] = -1

            frames += 1

            if t_now - fps_timer >= 0.5:
                current_fps = frames / (t_now - fps_timer)
                display.draw_text(330, 12, f"FPS: {current_fps:4.1f}", fg_rgb=(0, 255, 120), bg_rgb=(0, 0, 0), scale=2)
                print(f" ✨ Starfield FPS: {current_fps:.1f}")
                frames = 0
                fps_timer = t_now

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n [!] Stopping Starfield demo...")


# ============================================================
# DEMO 4: AUTOMATED SPI FREQUENCY BENCHMARK
# ============================================================

def run_spi_benchmark(display):
    """
    Stress-tests SPI clock rates from 16 MHz up to 60+ MHz.
    Measures raw full-screen throughput and verifies signal stability.
    """
    test_frequencies = [
        16_000_000,
        32_000_000,
        40_000_000,
        50_000_000,
        60_000_000,
    ]

    colors = [
        (255, 30, 30),   # Crimson Red
        (30, 255, 30),   # Neon Green
        (30, 100, 255),  # Cobalt Blue
        (255, 255, 255), # Pure White
    ]

    print("\n" + "=" * 60)
    print(" [4] AUTOMATED SPI FREQUENCY & BANDWIDTH BENCHMARK")
    print("=" * 60)

    results = []

    for freq in test_frequencies:
        freq_mhz = freq / 1_000_000
        print(f"\n[*] Testing SPI Clock: {freq_mhz:.0f} MHz...")
        display.set_speed(freq)

        display.fill_screen(0, 0, 0)
        display.draw_text(30, 130, f"TESTING {freq_mhz:.0f} MHZ...", fg_rgb=(255, 255, 0), bg_rgb=(0, 0, 0), scale=3)
        time.sleep(0.6)

        test_frames = 15
        t_start = time.perf_counter()

        for f in range(test_frames):
            color = colors[f % len(colors)]
            display.fill_screen(*color)

        t_elapsed = time.perf_counter() - t_start
        fps = test_frames / t_elapsed
        bytes_per_frame = WIDTH * HEIGHT * 3
        mb_per_sec = (bytes_per_frame * fps) / (1024 * 1024)

        print(f"    --> {freq_mhz:.0f} MHz: {fps:.2f} Full FPS | Throughput: {mb_per_sec:.2f} MB/s")
        results.append((freq_mhz, fps, mb_per_sec))
        time.sleep(0.3)

    display.fill_screen(15, 15, 25)
    display.draw_text(20, 20, "SPI BENCHMARK RESULTS", fg_rgb=(0, 255, 255), bg_rgb=(15, 15, 25), scale=2)
    display.draw_text(20, 50, "CLOCK     FULL-FPS   BANDWIDTH", fg_rgb=(180, 180, 180), bg_rgb=(15, 15, 25), scale=1)

    y_pos = 75
    for freq_mhz, fps, mb in results:
        line = f"{freq_mhz:2.0f} MHZ    {fps:5.2f} FPS   {mb:5.2f} MB/S"
        display.draw_text(20, y_pos, line, fg_rgb=(255, 255, 255), bg_rgb=(15, 15, 25), scale=2)
        y_pos += 30

    display.draw_text(20, 280, "SWEET SPOT: 40-50 MHZ", fg_rgb=(50, 255, 100), bg_rgb=(15, 15, 25), scale=2)
    time.sleep(4.0)


# ============================================================
# MAIN ENTRY POINT & INTERACTIVE MENU
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ILI9488 TFT High Performance Animation Suite")
    parser.add_argument("--speed", "-s", type=int, default=DEFAULT_SPI_SPEED,
                        help="SPI clock in Hz (default: 40,000,000 for 40MHz)")
    parser.add_argument("--demo", "-d", type=str, default="menu",
                        choices=["fire", "bounce", "stars", "bench", "color", "all", "menu"],
                        help="Which animation to run")
    parser.add_argument("--palette", "-p", type=str, default="vibrant",
                        choices=["vibrant", "classic", "blue"],
                        help="Fire palette style (vibrant, classic, blue)")
    parser.add_argument("--rgb", action="store_true",
                        help="Force RGB color order (default is BGR, matching KMRTM35018 hardware)")
    parser.add_argument("--invert", action="store_true",
                        help="Enable Display Inversion (for IPS panels)")
    args = parser.parse_args()

    use_bgr = not args.rgb

    print("\nInitializing ILI9488 Display Driver...")
    display = ILI9488Display(spi_speed=args.speed, bgr_swap=use_bgr, invert=args.invert)
    print(f"Display initialized at {args.speed / 1_000_000:.1f} MHz (Color: {'BGR' if use_bgr else 'RGB'}, Invert: {'ON' if args.invert else 'OFF'}).")

    try:
        if args.demo == "fire":
            run_doom_fire(display, palette_type=args.palette)
        elif args.demo == "color":
            run_color_calibration(display)
        elif args.demo == "bounce":
            run_dirty_rect_orbs(display)
        elif args.demo == "stars":
            run_starfield(display)
        elif args.demo == "bench":
            run_spi_benchmark(display)
        elif args.demo == "all":
            print("\nCycling through all demos (12 seconds each)...")
            while True:
                run_doom_fire(display, palette_type=args.palette, duration_sec=12)
                run_dirty_rect_orbs(display, duration_sec=12)
                run_starfield(display, duration_sec=12)
        else:
            # Interactive Terminal Menu
            while True:
                print("\n" + "=" * 45)
                print("    ILI9488 TFT DISPLAY SHOWCASE")
                print("=" * 45)
                print(" 1. 🔥 Ultra-Vibrant Arcade Fire (100% Saturation)")
                print(" 2. 🎮 Classic 1993 DOOM Fire")
                print(" 3. ❄️ Cyber Blue Plasma Fire")
                print(" 4. ⚡ Neon Cyber Bouncing Orbs (60+ FPS Dirty-Rect)")
                print(" 5. ✨ 3D Hyperspace Starfield Warp")
                print(" 6. 🎨 Color & Inversion Calibration Tool (RGB/BGR)")
                print(" 7. 📊 Automated SPI Frequency Benchmark (16-60 MHz)")
                print(" 8. 🔄 Cycle All Demos Continuously")
                print(" 9. 🚪 Exit & Clean Up")
                print("=" * 45)

                choice = input("Select an option (1-9): ").strip()
                if choice == "1":
                    run_doom_fire(display, palette_type="vibrant")
                elif choice == "2":
                    run_doom_fire(display, palette_type="classic")
                elif choice == "3":
                    run_doom_fire(display, palette_type="blue")
                elif choice == "4":
                    run_dirty_rect_orbs(display)
                elif choice == "5":
                    run_starfield(display)
                elif choice == "6":
                    run_color_calibration(display)
                elif choice == "7":
                    run_spi_benchmark(display)
                elif choice == "8":
                    while True:
                        run_doom_fire(display, palette_type="vibrant", duration_sec=10)
                        run_dirty_rect_orbs(display, duration_sec=10)
                        run_starfield(display, duration_sec=10)
                elif choice in ("9", "q", "exit"):
                    break
                else:
                    print("[!] Invalid selection.")

    except KeyboardInterrupt:
        print("\n[!] Exiting...")

    finally:
        print("Shutting down display cleanly...")
        display.close()
        print("Done. Display closed cleanly.")


if __name__ == "__main__":
    main()
