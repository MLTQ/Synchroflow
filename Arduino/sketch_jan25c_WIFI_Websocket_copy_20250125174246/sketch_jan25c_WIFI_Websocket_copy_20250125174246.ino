#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "Bob2_2.4GHz";
const char* password = "theaustralianfederation";

// WebSocket server details
const char* websocket_server = "192.168.1.38";  // Replace with your computer's IP
const int websocket_port = 8765;

// LED pin
const int LED_PIN = 2;

WebSocketsClient webSocket;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  Serial.println("\nESP32 Starting up...");

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.printf("Connecting to %s ", ssid);
  
  // Blink while connecting to WiFi
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
    Serial.print(".");
  }

  // WiFi connected - solid LED for 3 seconds
  Serial.println("\nConnected to WiFi!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  digitalWrite(LED_PIN, HIGH);
  delay(3000);  // Stay solid for 3 seconds to confirm WiFi connection

  // Initialize WebSocket connection
  Serial.println("Connecting to WebSocket server...");
  digitalWrite(LED_PIN, LOW);  // LED off while attempting WebSocket connection
  webSocket.begin(websocket_server, websocket_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop() {
  webSocket.loop();
  
  // If WiFi disconnects, blink rapidly
  if (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);
    delay(50);
    Serial.println("WiFi disconnected!");
  }
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("Disconnected from WebSocket server");
      // Slow blink when disconnected from WebSocket
      digitalWrite(LED_PIN, HIGH);
      delay(500);
      digitalWrite(LED_PIN, LOW);
      delay(500);
      break;
      
    case WStype_CONNECTED:
      Serial.println("Connected to WebSocket server");
      // Double blink to indicate WebSocket connection
      for(int i = 0; i < 2; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(200);
        digitalWrite(LED_PIN, LOW);
        delay(200);
      }
      digitalWrite(LED_PIN, HIGH);  // Then stay solid
      break;
      
    case WStype_TEXT:
      // Quick blink for each data packet
      digitalWrite(LED_PIN, LOW);
      delay(50);
      digitalWrite(LED_PIN, HIGH);
      
      // Print first part of received data
      Serial.print("Data received: ");
      if (length > 100) {
        Serial.write(payload, 100);
        Serial.println("...");
      } else {
        Serial.write(payload, length);
        Serial.println();
      }
      break;
  }
}