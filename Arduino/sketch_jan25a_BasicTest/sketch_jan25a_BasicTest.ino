void setup() {
  pinMode(2, OUTPUT);  // Built-in LED on most ESP32 boards
}

void loop() {
  digitalWrite(2, HIGH);
  delay(1000);
  digitalWrite(2, LOW);
  delay(1000);
}