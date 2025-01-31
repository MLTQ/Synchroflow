#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h>  // Required for 16 MHz Adafruit Trinket
#endif

#define PIN_NEO_PIXEL 32  // Arduino pin that connects to NeoPixel
#define NUM_PIXELS 12    // The number of LEDs (pixels) on NeoPixel


// WiFi credentials
const char* ssid = "Bob2_2.4GHz";
const char* password = "theaustralianfederation";

// WebSocket server details
const char* websocket_server = "192.168.1.38";
const int websocket_port = 8765;

// LED pins configuration
const int NUM_PINS = 50;  // Number of LED pins
const int LED_PINS[NUM_PINS] = {2, 32, 33, 25, 26, 27, 14, 12, 13, 15, 4, 5};  // All LED pins

// Frequency mapping constants
const int MIN_POWER = 0;
const int MAX_POWER = 20000;  // Updated to match brain_serve_delta.py
const float MIN_FREQ = 7.0;   // 7 Hz (theta-alpha boundary)
const float MAX_FREQ = 40.0;  // 40 Hz (gamma)

Adafruit_NeoPixel NeoPixel(NUM_PIXELS, PIN_NEO_PIXEL, NEO_GRB + NEO_KHZ800);

// WebSocket client
WebSocketsClient webSocket;

// JSON document for parsing
StaticJsonDocument<200> doc;

void setup() {
  // Initialize all LED pins
  for (int i = 0; i < NUM_PINS; i++) {
    pinMode(LED_PINS[i], OUTPUT);
  }

  NeoPixel.begin();
  
  Serial.begin(115200);
  Serial.println("\nESP32 Starting up...");

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.printf("Connecting to %s ", ssid);
  
  // Blink all LEDs while connecting to WiFi
  while (WiFi.status() != WL_CONNECTED) {
    for (int i = 0; i < NUM_PINS; i++) {
      digitalWrite(LED_PINS[i], HIGH);
    }
    delay(100);
    for (int i = 0; i < NUM_PINS; i++) {
      digitalWrite(LED_PINS[i], LOW);
    }
    delay(100);
    Serial.print(".");
  }

  // WiFi connected - all LEDs solid for 3 seconds
  Serial.println("\nConnected to WiFi!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  for (int i = 0; i < NUM_PINS; i++) {
    digitalWrite(LED_PINS[i], HIGH);
  }
  delay(3000);

  // Initialize WebSocket connection
  Serial.println("Connecting to WebSocket server...");
  for (int i = 0; i < NUM_PINS; i++) {
    digitalWrite(LED_PINS[i], LOW);
  }
  webSocket.begin(websocket_server, websocket_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop() {
  webSocket.loop();
  
  if (WiFi.status() != WL_CONNECTED) {
    for (int i = 0; i < NUM_PINS; i++) {
      digitalWrite(LED_PINS[i], HIGH);
    }
    delay(50);
    for (int i = 0; i < NUM_PINS; i++) {
      digitalWrite(LED_PINS[i], LOW);
    }
    delay(50);
    Serial.println("WiFi disconnected!");
    return;
  }
  
}

float mapPowerToFrequency(int power) {
  // Map power (0-20000) to frequency (7-40 Hz)
  return MIN_FREQ + (MAX_FREQ - MIN_FREQ) * (power / (float)MAX_POWER);
}

void handlePowerValue(int power) {
  // Constrain power to valid range
  power = constrain(power, MIN_POWER, MAX_POWER);
  
  // Calculate flash frequency and period
  float freq = mapPowerToFrequency(power);
  int period_ms = (int)(1000.0 / freq);  // Convert frequency to period in milliseconds
  int half_period = period_ms / 2;        // Half period for equal on/off time
  
  Serial.printf("Power: %d, Frequency: %.1f Hz, Period: %d ms\n", power, freq, period_ms);
  
  // Flash all LEDs at calculated frequency
  /*for (int i = 0; i < NUM_PINS; i++) {
    digitalWrite(LED_PINS[i], HIGH);
  }
  delay(half_period);
  for (int i = 0; i < NUM_PINS; i++) {
    digitalWrite(LED_PINS[i], LOW);
  }
  delay(half_period);
  */
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    NeoPixel.setPixelColor(pixel, NeoPixel.Color(2, 0, 0));
  }
  NeoPixel.show();
  delay(half_period);

  NeoPixel.clear();
  NeoPixel.show();
  delay(half_period);
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("Disconnected from WebSocket server");
      // Slow alternating blink when disconnected
      for (int i = 0; i < NUM_PINS; i++) {
        digitalWrite(LED_PINS[i], HIGH);
      }
      delay(500);
      for (int i = 0; i < NUM_PINS; i++) {
        digitalWrite(LED_PINS[i], LOW);
      }
      delay(500);
      break;
      
    case WStype_CONNECTED:
      Serial.println("Connected to WebSocket server");
      // Double blink to indicate connection
      for(int j = 0; j < 2; j++) {
        for (int i = 0; i < NUM_PINS; i++) {
          digitalWrite(LED_PINS[i], HIGH);
        }
        delay(200);
        for (int i = 0; i < NUM_PINS; i++) {
          digitalWrite(LED_PINS[i], LOW);
        }
        delay(200);
      }
      for (int i = 0; i < NUM_PINS; i++) {
        digitalWrite(LED_PINS[i], HIGH);
      }
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
