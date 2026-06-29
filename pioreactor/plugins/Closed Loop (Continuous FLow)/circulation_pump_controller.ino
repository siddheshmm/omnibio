/*
  circulation_pump_controller.ino

  Drives 3 pump MOSFETs to run/stop the closed-loop circulation:
    pump1: pio01 vial      -> pio02 vial
    pump2: pio02 vial      -> external beaker
    pump3: external beaker -> pio01 vial

  Subscribes to the same MQTT broker the Pioreactors use, and turns all
  3 pumps on or off together based on the retained message on
  ARDUINO_TOPIC. v1 is simple full-on/full-off; swap the analogWrite
  levels later for matched duty-cycle rate control (the pins are
  already PWM-capable, so no rewiring needed).

  Board:   Arduino Uno R4 WiFi
  Library: ArduinoMqttClient (install via Library Manager)
*/

#include <WiFiS3.h>
#include <ArduinoMqttClient.h>

const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const char* MQTT_BROKER = "10.102.100.34";
const int   MQTT_PORT   = 1883;
const char* MQTT_USER   = "pioreactor";
const char* MQTT_PASS   = "raspberry";
const char* MQTT_TOPIC  = "pioreactor/circulation_pump/run";

// PWM-capable pins on the Uno R4 WiFi
const int PUMP1_PIN = 5; // pio01 -> pio02
const int PUMP2_PIN = 6; // pio02 -> beaker
const int PUMP3_PIN = 9; // beaker -> pio01

// Validated via bench test: 65-70 runs smoothly without stalling.
// All 3 share one level for now; per-pump rate-matching comes later
// once each is calibrated (mL/sec) at this exact level.
const int PUMP_LEVEL = 67;

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

void setPumps(bool on) {
  int level = on ? PUMP_LEVEL : 0;
  analogWrite(PUMP1_PIN, level);
  analogWrite(PUMP2_PIN, level);
  analogWrite(PUMP3_PIN, level);
}

void onMqttMessage(int messageSize) {
  String payload;
  while (mqttClient.available()) payload += (char)mqttClient.read();
  setPumps(payload == "1");
}

void connectWiFi() {
  while (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    delay(2000);
  }
}

void connectMqtt() {
  mqttClient.setUsernamePassword(MQTT_USER, MQTT_PASS);
  while (!mqttClient.connect(MQTT_BROKER, MQTT_PORT)) {
    delay(1000);
  }
  mqttClient.onMessage(onMqttMessage);
  mqttClient.subscribe(MQTT_TOPIC);
}

void setup() {
  pinMode(PUMP1_PIN, OUTPUT);
  pinMode(PUMP2_PIN, OUTPUT);
  pinMode(PUMP3_PIN, OUTPUT);
  setPumps(false); // start safe: pumps off

  connectWiFi();
  connectMqtt();
}

void loop() {
  if (!mqttClient.connected()) {
    connectMqtt();
  }
  mqttClient.poll();
}
