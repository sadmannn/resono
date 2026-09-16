/*
 * ====================================================================
 * RESONO — ESP32 Autonomous & Serial Robot Eyes Controller (WRO 2026)
 * ====================================================================
 * Hardware:
 *   - ESP32 Development Board (ESP32-WROOM-32 / NodeMCU)
 *   - 1.9" IPS TFT Display (ST7789V, 320x170, 16-bit RGB565)
 *
 * --------------------------------------------------------------------
 * WIRING DIAGRAM: ESP32 <-> 1.9" ST7789 TFT
 * --------------------------------------------------------------------
 *  ST7789 Pin    ESP32 Pin     Notes
 *  -------------------------------------------------------------
 *  VCC           3V3 (or 5V)   Power (3.3V Recommended)
 *  GND           GND           Ground
 *  SCL / SCK     GPIO 18       Hardware VSPI Clock (SCK)
 *  SDA / MOSI    GPIO 23       Hardware VSPI Data  (MOSI)
 *  RES / RST     GPIO 4        TFT Hardware Reset
 *  DC / RS       GPIO 2        Data / Command Select
 *  CS            GPIO 5        Chip Select (Tie to GND if no pin)
 *  BLK / LED     GPIO 15       Backlight (Or connect to 3.3V)
 *
 * --------------------------------------------------------------------
 * WIRING DIAGRAM: ESP32 <-> RASPBERRY PI 5 (UART COMMUNICATION)
 * --------------------------------------------------------------------
 *  Raspberry Pi 5 Pin          ESP32 Pin         Safety Notes
 *  -------------------------------------------------------------
 *  Pin 8  (GPIO 14 / TXD0) ->  GPIO 16 (RX2) or GPIO 3 (RX0)  [3.3V SAFE]
 *  Pin 10 (GPIO 15 / RXD0) <-  GPIO 17 (TX2) or GPIO 1 (TX0)  [3.3V SAFE]
 *  Pin 6/9/14/20 (GND)     <-> GND               [COMMON GROUND]
 *
 *  SAFETY NOTE: Both Raspberry Pi 5 and ESP32 GPIO pins use 3.3V logic.
 *  They can be connected directly without any level shifter!
 * ====================================================================
 */

#include <SPI.h>

// ====================================================================
// PIN ASSIGNMENTS (Configurable for your wiring)
// ====================================================================
#define TFT_SCLK 18
#define TFT_MOSI 23
#define TFT_MISO 19  // Not connected, but required by VSPI
#define TFT_CS   5
#define TFT_DC   2
#define TFT_RST  4
#define TFT_BL   15  // Set to -1 if hardwired to 3.3V

// Serial pins connected to Raspberry Pi 5
#define PI_UART_RX 16 // ESP32 RX2 <- RPi Pin 8 (GPIO 14 / TXD)
#define PI_UART_TX 17 // ESP32 TX2 -> RPi Pin 10 (GPIO 15 / RXD)

// --------------------------------------------------------------------
// DISPLAY SPECS (ST7789 320x170)
// --------------------------------------------------------------------
#define SCREEN_W 320
#define SCREEN_H 170

// Hardware window offsets for ST7789 320x170 controller
#define OFFSET_X 0
#define OFFSET_Y 35

// 16-bit RGB565 Color Definitions
#define COLOR_BLACK   0x0000
#define COLOR_WHITE   0xFFFF
#define COLOR_CYAN    0x3DF7 // (56, 189, 248) - Detect State
#define COLOR_RED     0xF228 // (239, 68, 68)  - Listening State
#define COLOR_GREEN   0x25E7 // (34, 197, 94)  - Happy State
#define COLOR_AMBER   0xFDE0 // (245, 158, 11) - Conversation State

// ====================================================================
// EMOTION / OPERATING STATES
// ====================================================================
enum RobotState {
  STATE_NORMAL,    // Pure white eyes, casual idle glances
  STATE_DETECT,    // Cyan alert eyes, dilated, tracking object
  STATE_LISTEN,    // Vivid red eyes, attentive / pulse
  STATE_HAPPY      // Arched happy eyes
};

RobotState currentState = STATE_NORMAL;
uint16_t eyeColor = COLOR_WHITE;

// Animation Variables
float currentLookX = 0.0f;
float currentLookY = 0.0f;
float targetLookX = 0.0f;
float targetLookY = 0.0f;

float currentBlink = 0.0f; // 0.0 = open, 1.0 = closed
bool isBlinking = false;
unsigned long blinkStartTime = 0;
unsigned long blinkDuration = 140; // ms
unsigned long nextBlinkTime = 0;

unsigned long nextGazeChangeTime = 0;
unsigned long lastFrameTime = 0;

// Pulse effect for listening state
float pulseAngle = 0.0f;

// Bounding box tracking for fast dirty-rect updating (avoids full screen redraws)
int prevLBox[4] = {0, 0, 0, 0}; // x, y, w, h
int prevRBox[4] = {0, 0, 0, 0};

// Hardware SPI settings (40MHz rock solid on ESP32 VSPI)
SPISettings spiSettings(40000000, MSBFIRST, SPI_MODE0);

// ====================================================================
// LOW-LEVEL ST7789 HARDWARE DRIVER
// ====================================================================

inline void writeCommand(uint8_t cmd) {
  digitalWrite(TFT_DC, LOW);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);
  SPI.transfer(cmd);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);
}

inline void writeData8(uint8_t data) {
  digitalWrite(TFT_DC, HIGH);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);
  SPI.transfer(data);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);
}

inline void writeData16(uint16_t data) {
  digitalWrite(TFT_DC, HIGH);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);
  SPI.transfer16(data);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);
}

void setWindow(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
  x0 += OFFSET_X;
  x1 += OFFSET_X;
  y0 += OFFSET_Y;
  y1 += OFFSET_Y;

  writeCommand(0x2A); // CASET
  digitalWrite(TFT_DC, HIGH);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);
  SPI.transfer(x0 >> 8);
  SPI.transfer(x0 & 0xFF);
  SPI.transfer(x1 >> 8);
  SPI.transfer(x1 & 0xFF);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);

  writeCommand(0x2B); // RASET
  digitalWrite(TFT_DC, HIGH);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);
  SPI.transfer(y0 >> 8);
  SPI.transfer(y0 & 0xFF);
  SPI.transfer(y1 >> 8);
  SPI.transfer(y1 & 0xFF);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);

  writeCommand(0x2C); // RAMWR
}

void fillScreen(uint16_t color) {
  setWindow(0, 0, SCREEN_W - 1, SCREEN_H - 1);
  digitalWrite(TFT_DC, HIGH);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);
  
  uint32_t totalPixels = SCREEN_W * SCREEN_H;
  for (uint32_t i = 0; i < totalPixels; i++) {
    SPI.transfer16(color);
  }
  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);
}

void fillRect(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t color) {
  if (x >= SCREEN_W || y >= SCREEN_H || w <= 0 || h <= 0) return;
  if (x < 0) { w += x; x = 0; }
  if (y < 0) { h += y; y = 0; }
  if (x + w > SCREEN_W) w = SCREEN_W - x;
  if (y + h > SCREEN_H) h = SCREEN_H - y;

  setWindow(x, y, x + w - 1, y + h - 1);
  digitalWrite(TFT_DC, HIGH);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);
  uint32_t total = (uint32_t)w * h;
  for (uint32_t i = 0; i < total; i++) {
    SPI.transfer16(color);
  }
  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);
}

void initST7789() {
  pinMode(TFT_DC, OUTPUT);
  if (TFT_CS >= 0) {
    pinMode(TFT_CS, OUTPUT);
    digitalWrite(TFT_CS, HIGH);
  }
  if (TFT_RST >= 0) {
    pinMode(TFT_RST, OUTPUT);
    digitalWrite(TFT_RST, HIGH);
    delay(50);
    digitalWrite(TFT_RST, LOW);
    delay(50);
    digitalWrite(TFT_RST, HIGH);
    delay(120);
  }
  if (TFT_BL >= 0) {
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH);
  }

  writeCommand(0x01); // Software Reset
  delay(150);

  writeCommand(0x11); // Sleep Out
  delay(120);

  writeCommand(0x3A); // Interface Pixel Format: 16-bit RGB565
  writeData8(0x55);

  writeCommand(0x36); // Memory Data Access Control (Landscape)
  writeData8(0x60);   // Standard 320x170 orientation (Matches Resono robot mounting)

  writeCommand(0x21); // Display Inversion ON (Required for ST7789 IPS vivid colors)
  writeCommand(0x13); // Normal Display Mode ON
  writeCommand(0x29); // Display ON
  delay(50);

  fillScreen(COLOR_BLACK);
}

// ====================================================================
// PROCEDURAL EYE RENDERING ENGINE
// ====================================================================

// Renders one slanted eye into a local buffer and pushes directly to SPI
void drawSingleEye(int cx, int cy, int baseW, int baseH, float slantAngleDeg, float blinkFactor, uint16_t color, int boxOut[4]) {
  // Squish height based on blink
  int h = (int)(4 + (baseH - 4) * (1.0f - blinkFactor));
  int w = baseW;

  // Compute rotation angle in radians
  float rad = slantAngleDeg * 3.14159265f / 180.0f;
  float cosA = cos(rad);
  float sinA = sin(rad);

  float rx = w / 2.0f;
  float ry = h / 2.0f;

  // Compute bounding box
  float maxDim = sqrt(rx * rx + ry * ry) + 4;
  int bbX = max(0, (int)(cx - maxDim));
  int bbY = max(0, (int)(cy - maxDim));
  int bbW = min(SCREEN_W - bbX, (int)(maxDim * 2));
  int bbH = min(SCREEN_H - bbY, (int)(maxDim * 2));

  boxOut[0] = bbX;
  boxOut[1] = bbY;
  boxOut[2] = bbW;
  boxOut[3] = bbH;

  if (bbW <= 0 || bbH <= 0) return;

  setWindow(bbX, bbY, bbX + bbW - 1, bbY + bbH - 1);
  digitalWrite(TFT_DC, HIGH);
  if (TFT_CS >= 0) digitalWrite(TFT_CS, LOW);

  // Scanline rendering of rotated ellipse
  for (int y = bbY; y < bbY + bbH; y++) {
    float dy = y - cy;
    for (int x = bbX; x < bbX + bbW; x++) {
      float dx = x - cx;

      // Rotate point back: (x * cos - y * sin, x * sin + y * cos)
      float u = dx * cosA + dy * sinA;
      float v = -dx * sinA + dy * cosA;

      float dist = (u * u) / (rx * rx) + (v * v) / (ry * ry);

      if (dist <= 1.0f) {
        // Pixel is inside eye
        if (currentState == STATE_HAPPY && v < 0) {
          // In happy mode, cut the top off to create a crescent upward curve smile!
          SPI.transfer16(COLOR_BLACK);
        } else {
          SPI.transfer16(color);
        }
      } else {
        SPI.transfer16(COLOR_BLACK);
      }
    }
  }

  if (TFT_CS >= 0) digitalWrite(TFT_CS, HIGH);
}

void renderEyes() {
  int baseW = 68;
  int baseH = 92;

  if (currentState == STATE_DETECT) {
    baseW = 76;
    baseH = 98;
    eyeColor = COLOR_CYAN;
  } else if (currentState == STATE_LISTEN) {
    // Subtle rhythmic heartbeat pulse
    float p = 1.0f + 0.08f * sin(pulseAngle);
    baseW = (int)(72 * p);
    baseH = (int)(94 * p);
    eyeColor = COLOR_RED;
  } else if (currentState == STATE_HAPPY) {
    baseW = 74;
    baseH = 80;
    eyeColor = COLOR_GREEN;
  } else {
    baseW = 68;
    baseH = 92;
    eyeColor = COLOR_WHITE;
  }

  int cy = SCREEN_H / 2 + (int)(currentLookY * 12);
  int lx = 105 + (int)(currentLookX * 18);
  int rx = 215 + (int)(currentLookX * 18);

  int newLBox[4];
  int newRBox[4];

  // Draw left eye (-14 degree slant) and right eye (+14 degree slant)
  drawSingleEye(lx, cy, baseW, baseH, -14.0f, currentBlink, eyeColor, newLBox);
  drawSingleEye(rx, cy, baseW, baseH,  14.0f, currentBlink, eyeColor, newRBox);

  // Clear any leftover black pixels outside new bounding box if gaze shifted
  if (prevLBox[2] > 0) {
    if (prevLBox[0] < newLBox[0]) fillRect(prevLBox[0], prevLBox[1], newLBox[0] - prevLBox[0], prevLBox[3], COLOR_BLACK);
    if (prevLBox[0] + prevLBox[2] > newLBox[0] + newLBox[2]) {
      int ox = newLBox[0] + newLBox[2];
      fillRect(ox, prevLBox[1], (prevLBox[0] + prevLBox[2]) - ox, prevLBox[3], COLOR_BLACK);
    }
  }
  if (prevRBox[2] > 0) {
    if (prevRBox[0] < newRBox[0]) fillRect(prevRBox[0], prevRBox[1], newRBox[0] - prevRBox[0], prevRBox[3], COLOR_BLACK);
    if (prevRBox[0] + prevRBox[2] > newRBox[0] + newRBox[2]) {
      int ox = newRBox[0] + newRBox[2];
      fillRect(ox, prevRBox[1], (prevRBox[0] + prevRBox[2]) - ox, prevRBox[3], COLOR_BLACK);
    }
  }

  for (int i = 0; i < 4; i++) {
    prevLBox[i] = newLBox[i];
    prevRBox[i] = newRBox[i];
  }
}

// ====================================================================
// SERIAL PROTOCOL PARSER (Commands from Raspberry Pi 5)
// ====================================================================
void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  Serial.print("[RESONO EYE RX] ");
  Serial.println(cmd);

  if (cmd == "STATE:NORMAL" || cmd == "NORMAL") {
    currentState = STATE_NORMAL;
  } else if (cmd == "STATE:DETECT" || cmd == "DETECT") {
    currentState = STATE_DETECT;
  } else if (cmd == "STATE:LISTEN" || cmd == "LISTEN" || cmd == "RECORD") {
    currentState = STATE_LISTEN;
  } else if (cmd == "STATE:HAPPY" || cmd == "HAPPY") {
    currentState = STATE_HAPPY;
  } else if (cmd == "BLINK") {
    isBlinking = true;
    blinkStartTime = millis();
  } else if (cmd.startsWith("LOOK:")) {
    float val = cmd.substring(5).toFloat();
    targetLookX = constrain(val, -1.0f, 1.0f);
  }
}

void checkSerialInputs() {
  // Check primary Serial (USB)
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    processCommand(line);
  }
  // Check Serial2 (Hardware pins connected to Pi TX/RX)
  while (Serial2.available()) {
    String line = Serial2.readStringUntil('\n');
    processCommand(line);
  }
}

// ====================================================================
// AUTONOMOUS BEHAVIOR & ANIMATION LOOP
// ====================================================================
void updateAnimations() {
  unsigned long now = millis();

  // 1. Autonomous Blinking
  if (!isBlinking && now >= nextBlinkTime) {
    isBlinking = true;
    blinkStartTime = now;
    blinkDuration = (currentState == STATE_DETECT) ? 90 : random(110, 160);
    nextBlinkTime = now + random(2500, 5500); // Blink every 2.5 to 5.5s
  }

  if (isBlinking) {
    unsigned long elapsed = now - blinkStartTime;
    if (elapsed >= blinkDuration) {
      isBlinking = false;
      currentBlink = 0.0f;
    } else {
      float half = blinkDuration / 2.0f;
      if (elapsed < half) {
        currentBlink = (float)elapsed / half;
      } else {
        currentBlink = 1.0f - ((float)(elapsed - half) / half);
      }
    }
  }

  // 2. Autonomous Idle Wandering Gaze (When in normal state and no explicit look locked)
  if (currentState == STATE_NORMAL && now >= nextGazeChangeTime) {
    int r = random(0, 5);
    if (r == 0)      { targetLookX = -0.7f; targetLookY = 0.0f; }
    else if (r == 1) { targetLookX =  0.7f; targetLookY = 0.0f; }
    else if (r == 2) { targetLookX =  0.0f; targetLookY = -0.4f; }
    else             { targetLookX =  0.0f; targetLookY = 0.0f; } // Center mostly
    nextGazeChangeTime = now + random(1400, 3600);
  }

  // Smooth interpolation toward target look position
  currentLookX += (targetLookX - currentLookX) * 0.22f;
  currentLookY += (targetLookY - currentLookY) * 0.22f;

  // Pulse effect update
  pulseAngle += 0.12f;
  if (pulseAngle > 6.283f) pulseAngle -= 6.283f;
}

// ====================================================================
// ARDUINO MAIN ENTRY POINTS
// ====================================================================
void setup() {
  // Initialize standard Serial (USB monitoring)
  Serial.begin(115200);
  delay(100);
  Serial.println("\n[RESONO] ESP32 Robot Eyes Controller Initializing...");

  // Initialize Serial2 for direct Raspberry Pi 5 UART (Pins 16 RX, 17 TX)
  Serial2.begin(115200, SERIAL_8N1, PI_UART_RX, PI_UART_TX);
  Serial.printf("[RESONO] UART Bridge Active on RX2 (GPIO %d) / TX2 (GPIO %d)\n", PI_UART_RX, PI_UART_TX);

  // Initialize Hardware SPI for ST7789
  SPI.begin(TFT_SCLK, TFT_MISO, TFT_MOSI, TFT_CS);
  SPI.beginTransaction(spiSettings);

  // Initialize ST7789 Display
  initST7789();
  Serial.println("[RESONO] ST7789 1.9\" TFT Ready (320x170)");

  // Smooth Startup Eye Wake-up Sequence
  for (float b = 1.0f; b >= 0.0f; b -= 0.1f) {
    currentBlink = b;
    renderEyes();
    delay(25);
  }

  nextBlinkTime = millis() + 2000;
  nextGazeChangeTime = millis() + 1500;
}

void loop() {
  // 1. Process commands from Raspberry Pi 5
  checkSerialInputs();

  // 2. Update procedural animation state
  updateAnimations();

  // 3. Render eyes at locked 40-50 FPS
  renderEyes();

  delay(20);
}
