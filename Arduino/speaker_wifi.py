#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

#define ANALOG_INPUT_PIN A10  // Analog input pin
#define PWM_OUTPUT_PIN A9     // PWM output for speaker

// WiFi credentials
const char* ssid = "Bob2_2.4GHz";
const char* password = "theaustralianfederation";

// WebSocket server details
const char* websocket_server = "192.168.1.38";
const int websocket_port = 8765;

// LED pins configuration
const int NUM_PINS = 50;
const int LED_PINS[NUM_PINS] = {2, 32, 33, 25, 26, 27, 14, 12, 13, 15, 4, 5};

// Frequency mapping constants
const int MIN_POWER = 0;
const int MAX_POWER = 20000;
const float MIN_FREQ = 7.0;   // 7 Hz (theta-alpha boundary)
const float MAX_FREQ = 40.0;  // 40 Hz (gamma)

// Analog frequency mapping
const int MIN_ANALOG_FREQ = 100;
const int MAX_ANALOG_FREQ = 60000;

WebSocketsClient webSocket;
StaticJsonDocument<200> doc;

void setup() {
  Serial.begin(115200);
  Serial.println("\nESP32 Starting up...");

  // Initialize LED pins
  for (int i = 0; i < NUM_PINS; i++) {
    pinMode(LED_PINS[i], OUTPUT);
  }

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.printf("Connecting to %s ", ssid);

  while (WiFi.status() != WL_CONNECTED) {
    blinkLEDs(100);
    Serial.print(".");
  }

  Serial.println("\nConnected to WiFi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Connect to WebSocket
  Serial.println("Connecting to WebSocket server...");
  webSocket.begin(websocket_server, websocket_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop() {
  webSocket.loop();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected!");
    blinkLEDs(50);
    return;
  }

  // Read the analog input and generate frequency
  int analogValue = analogRead(ANALOG_INPUT_PIN);
  int analogFreq = map(analogValue, 3, 1024, MIN_ANALOG_FREQ, MAX_ANALOG_FREQ);
  analogFreq = constrain(analogFreq, MIN_ANALOG_FREQ, MAX_ANALOG_FREQ);

  Serial.printf("Analog Value: %d -> Analog Frequency: %d Hz\n", analogValue, analogFreq);

  // Output the mapped frequency as audio
  tone(PWM_OUTPUT_PIN, analogFreq);
  delay(100);
}

float mapPowerToFrequency(int power) {
  return MIN_FREQ + (MAX_FREQ - MIN_FREQ) * (power / (float)MAX_POWER);
}

void handlePowerValue(int power) {
  power = constrain(power, MIN_POWER, MAX_POWER);

  float freq = mapPowerToFrequency(power);
  Serial.printf("Power: %d, Mapped Frequency: %.1f Hz\n", power, freq);

  // Output mapped frequency as audio
  tone(PWM_OUTPUT_PIN, (int)freq);
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("Disconnected from WebSocket server");
      blinkLEDs(500);
      break;
      
    case WStype_CONNECTED:
      Serial.println("Connected to WebSocket server");
      blinkLEDs(200);
      break;
      
    case WStype_TEXT:
      DeserializationError error = deserializeJson(doc, payload);
      if (error) {
        Serial.print("deserializeJson() failed: ");
        Serial.println(error.c_str());
        return;
      }
      if (doc.containsKey("delta_power")) {
        int power = doc["delta_power"];
        handlePowerValue(power);
      }
      break;
  }
}

void blinkLEDs(int delayMs) {
  for (int i = 0; i < NUM_PINS; i++) {
    digitalWrite(LED_PINS[i], HIGH);
  }
  delay(delayMs);
  for (int i = 0; i < NUM_PINS; i++) {
    digitalWrite(LED_PINS[i], LOW);
  }
  delay(delayMs);
}
