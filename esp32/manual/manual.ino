#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// =================================================================
// 1. NETWORK CONFIGURATION
// =================================================================
const char* WIFI_SSID     = "OX";
const char* WIFI_PASSWORD = "sadman02610";

// =================================================================
// 2. UPDATED PIN ASSIGNMENTS (GPIO 14 -> GPIO 4)
// =================================================================
// Front Left (FL)
#define FL_IN1 25
#define FL_IN2 26

// Front Right (FR) - CHANGED IN2 TO GPIO 4
#define FR_IN1 27
#define FR_IN2 4 

// Rear Left (RL)
#define RL_IN1 18
#define RL_IN2 19

// Rear Right (RR)
#define RR_IN1 21
#define RR_IN2 22

// Servo Pins
#define SERVO_LEFT_PIN  23
#define SERVO_RIGHT_PIN 32

// Servo Angles
#define ARM_LEFT_DOWN   10
#define ARM_LEFT_UP     110
#define ARM_RIGHT_DOWN  170
#define ARM_RIGHT_UP    70

Servo leftServo;
Servo rightServo;
WebServer server(80);

int currentSpeed = 200;
unsigned long lastHeartbeatTime = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 600;

// Non-blocking Speed Ramping
int targetFL = 0, targetFR = 0, targetRL = 0, targetRR = 0;
int actualFL = 0, actualFR = 0, actualRL = 0, actualRR = 0;

unsigned long lastRampUpdate = 0;
const unsigned long RAMP_INTERVAL_MS = 8;
const int RAMP_STEP_UP   = 35;
const int RAMP_STEP_DOWN = 22;

// Embedded HTML
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>RESONO — Controller</title>
  <style>
    * { box-sizing: border-box; -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; touch-action: none; }
    body { background: #0b1120; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 16px; display: flex; flex-direction: column; align-items: center; }
    .header { text-align: center; margin-bottom: 12px; }
    .header h1 { margin: 0; font-size: 22px; color: #38bdf8; letter-spacing: 2px; }
    .header p { margin: 2px 0 0 0; font-size: 11px; color: #94a3b8; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 999px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-size: 11px; font-weight: 600; margin-top: 4px; }
    .card { background: #1e293b; border-radius: 16px; padding: 14px; width: 100%; max-width: 360px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); margin-bottom: 14px; border: 1px solid #334155; }
    .section-title { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 10px; font-weight: 700; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    button.btn { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 12px; width: 100%; height: 58px; font-size: 20px; font-weight: 600; display: flex; justify-content: center; align-items: center; cursor: pointer; transition: background 0.05s ease, transform 0.05s ease; }
    button.btn:active, button.btn.active { background: #38bdf8 !important; color: #0b1120 !important; border-color: #38bdf8 !important; transform: scale(0.96); }
    button.btn-stop { background: #ef4444 !important; border-color: #dc2626 !important; color: white !important; font-size: 13px; }
    .arm-grid { display: grid; grid-template-columns: 1fr; gap: 8px; }
    .btn-arm-up { background: #0284c7 !important; border-color: #0ea5e9 !important; height: 48px !important; font-size: 14px !important; }
    .btn-arm-down { background: #475569 !important; border-color: #64748b !important; height: 40px !important; font-size: 13px !important; }
    .slider-container { margin-top: 12px; }
    .slider-header { display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
    input[type=range] { width: 100%; height: 6px; background: #475569; border-radius: 4px; outline: none; -webkit-appearance: none; }
    input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; border-radius: 50%; background: #38bdf8; cursor: pointer; }
  </style>
</head>
<body>
  <div class="header">
    <h1>RESONO</h1>
    <p>Odyssey X • WRO 2026</p>
    <div class="badge">ESP32 Clean Boot</div>
  </div>
  <div class="card">
    <div class="section-title">4WD Mecanum Base</div>
    <div class="grid">
      <button class="btn" id="btnTL">↺</button>
      <button class="btn" id="btnF">▲</button>
      <button class="btn" id="btnTR">↻</button>
      <button class="btn" id="btnSL">◀</button>
      <button class="btn btn-stop" id="btnStop">STOP</button>
      <button class="btn" id="btnSR">▶</button>
      <div></div>
      <button class="btn" id="btnB">▼</button>
      <div></div>
    </div>
    <div class="slider-container">
      <div class="slider-header">
        <span>Motor Speed PWM (0 - 255)</span>
        <span id="speedVal">200</span>
      </div>
      <input type="range" min="0" max="255" value="200" id="speedSlider" oninput="updateSpeedDisplay(this.value)" onchange="sendSpeed(this.value)">
    </div>
  </div>
  <div class="card">
    <div class="section-title">Robotic Arms & Cue</div>
    <div class="arm-grid">
      <button class="btn btn-arm-up" onclick="sendArm('up')">🙌 RAISE BOTH ARMS</button>
      <button class="btn btn-arm-down" onclick="sendArm('down')">👇 Lower Both Arms</button>
    </div>
  </div>
  <script>
    let activeCmd = null;
    let heartbeatInterval = null;
    function sendCommand(cmd) { fetch('/move?cmd=' + cmd).catch(() => {}); }
    function startHold(cmd, element) {
      if (activeCmd === cmd) return;
      activeCmd = cmd;
      if (element) element.classList.add('active');
      sendCommand(cmd);
      if (heartbeatInterval) clearInterval(heartbeatInterval);
      heartbeatInterval = setInterval(() => { if (activeCmd) sendCommand(activeCmd); }, 120);
    }
    function stopHold(element) {
      if (!activeCmd) return;
      activeCmd = null;
      if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; }
      if (element) element.classList.remove('active');
      sendCommand('STOP');
    }
    function bindButton(id, cmd) {
      const el = document.getElementById(id);
      if (!el) return;
      const onStart = (e) => { e.preventDefault(); startHold(cmd, el); };
      const onEnd   = (e) => { e.preventDefault(); stopHold(el); };
      el.addEventListener('mousedown', onStart);
      el.addEventListener('mouseup', onEnd);
      el.addEventListener('mouseleave', onEnd);
      el.addEventListener('touchstart', onStart, { passive: false });
      el.addEventListener('touchend', onEnd, { passive: false });
      el.addEventListener('touchcancel', onEnd, { passive: false });
    }
    bindButton('btnF', 'F'); bindButton('btnB', 'B'); bindButton('btnSL', 'SL');
    bindButton('btnSR', 'SR'); bindButton('btnTL', 'TL'); bindButton('btnTR', 'TR');
    document.getElementById('btnStop').addEventListener('click', () => { stopHold(null); });
    function sendArm(action) { fetch('/arms?state=' + action).catch(() => {}); }
    function updateSpeedDisplay(val) { document.getElementById('speedVal').innerText = val; }
    function sendSpeed(val) { fetch('/speed?val=' + val).catch(() => {}); }
    window.addEventListener('blur', () => { stopHold(null); });
  </script>
</body>
</html>
)rawliteral";

void applyMotorPWM(int in1, int in2, int speed) {
  if (speed > 0) {
    analogWrite(in1, speed);
    analogWrite(in2, 0);
  } else if (speed < 0) {
    analogWrite(in1, 0);
    analogWrite(in2, -speed);
  } else {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
  }
}

int calculateRamp(int current, int target) {
  if (current == target) return current;
  int step = (abs(target) < abs(current)) ? RAMP_STEP_DOWN : RAMP_STEP_UP;
  if (current < target) {
    current += step;
    if (current > target) current = target;
  } else {
    current -= step;
    if (current < target) current = target;
  }
  return current;
}

void updateMotorRamping() {
  actualFL = calculateRamp(actualFL, targetFL);
  actualFR = calculateRamp(actualFR, targetFR);
  actualRL = calculateRamp(actualRL, targetRL);
  actualRR = calculateRamp(actualRR, targetRR);

  applyMotorPWM(FL_IN1, FL_IN2, actualFL);
  applyMotorPWM(FR_IN1, FR_IN2, actualFR);
  applyMotorPWM(RL_IN1, RL_IN2, actualRL);
  applyMotorPWM(RR_IN1, RR_IN2, actualRR);
}

void setTargetSpeeds(int fl, int fr, int rl, int rr) {
  targetFL = fl; targetFR = fr; targetRL = rl; targetRR = rr;
  lastHeartbeatTime = millis();
}

void stopMotors() { setTargetSpeeds(0, 0, 0, 0); }

void handleRoot() { server.send(200, "text/html", INDEX_HTML); }

void handleMove() {
  if (!server.hasArg("cmd")) { server.send(400, "text/plain", "Missing cmd"); return; }
  String cmd = server.arg("cmd");
  int spd = currentSpeed;

  if (cmd == "F")       setTargetSpeeds(spd, spd, spd, spd);
  else if (cmd == "B")  setTargetSpeeds(-spd, -spd, -spd, -spd);
  else if (cmd == "SL") setTargetSpeeds(-spd, spd, spd, -spd);
  else if (cmd == "SR") setTargetSpeeds(spd, -spd, -spd, spd);
  else if (cmd == "TL") setTargetSpeeds(-spd, spd, -spd, spd);
  else if (cmd == "TR") setTargetSpeeds(spd, -spd, spd, -spd);
  else                  stopMotors();

  server.send(200, "text/plain", "OK");
}

void handleArms() {
  if (!server.hasArg("state")) { server.send(400, "text/plain", "Missing state"); return; }
  String state = server.arg("state");
  if (state == "up") {
    leftServo.write(ARM_LEFT_UP);
    rightServo.write(ARM_RIGHT_UP);
  } else if (state == "down") {
    leftServo.write(ARM_LEFT_DOWN);
    rightServo.write(ARM_RIGHT_DOWN);
  }
  server.send(200, "text/plain", "OK");
}

void handleSpeed() {
  if (server.hasArg("val")) {
    currentSpeed = constrain(server.arg("val").toInt(), 0, 255);
  }
  server.send(200, "text/plain", "OK");
}

void setup() {
  // 1. INSTANT MOTOR SHUTOFF (Executed in microseconds before anything else)
  pinMode(FL_IN1, OUTPUT); pinMode(FL_IN2, OUTPUT);
  pinMode(FR_IN1, OUTPUT); pinMode(FR_IN2, OUTPUT);
  pinMode(RL_IN1, OUTPUT); pinMode(RL_IN2, OUTPUT);
  pinMode(RR_IN1, OUTPUT); pinMode(RR_IN2, OUTPUT);

  digitalWrite(FL_IN1, LOW); digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, LOW); digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, LOW); digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, LOW); digitalWrite(RR_IN2, LOW);

  Serial.begin(115200);

  // 2. Servos
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  leftServo.setPeriodHertz(50);
  rightServo.setPeriodHertz(50);
  leftServo.attach(SERVO_LEFT_PIN, 500, 2400);
  rightServo.attach(SERVO_RIGHT_PIN, 500, 2400);
  leftServo.write(ARM_LEFT_DOWN);
  rightServo.write(ARM_RIGHT_DOWN);

  // 3. Wi-Fi
  Serial.printf("Connecting to Wi-Fi Hotspot: %s ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\n[SUCCESS] Connected!");
  Serial.print(">>> Web Controller URL: http://");
  Serial.println(WiFi.localIP());

  server.on("/", HTTP_GET, handleRoot);
  server.on("/move", HTTP_GET, handleMove);
  server.on("/arms", HTTP_GET, handleArms);
  server.on("/speed", HTTP_GET, handleSpeed);
  server.begin();
}

void loop() {
  server.handleClient();

  if (millis() - lastRampUpdate >= RAMP_INTERVAL_MS) {
    lastRampUpdate = millis();
    updateMotorRamping();
  }

  if (millis() - lastHeartbeatTime > WATCHDOG_TIMEOUT_MS) {
    targetFL = 0; targetFR = 0; targetRL = 0; targetRR = 0;
  }
}