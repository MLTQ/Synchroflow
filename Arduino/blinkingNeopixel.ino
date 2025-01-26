/*
 * Created by ArduinoGetStarted.com
 *
 * This example code is in the public domain
 *
 * Tutorial page: https://arduinogetstarted.com/tutorials/arduino-neopixel-led-strip
 */

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h>  // Required for 16 MHz Adafruit Trinket
#endif

#define PIN_NEO_PIXEL 32  // Arduino pin that connects to NeoPixel
#define NUM_PIXELS 12    // The number of LEDs (pixels) on NeoPixel

#define DELAY_INTERVAL 250  // 250ms pause between each pixel

Adafruit_NeoPixel NeoPixel(NUM_PIXELS, PIN_NEO_PIXEL, NEO_GRB + NEO_KHZ800);

int potPin = A16;
int brightness = 0;
int freq = 7;

void setup() {
  NeoPixel.begin();
  //Serial.begin(9600);
  pinMode(potPin, INPUT);
}

void loop() {
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    //setBrightnessFromPot();
    NeoPixel.setPixelColor(pixel, NeoPixel.Color(2, 0, 0));
  }
  NeoPixel.show();
  delay(500/freq);

  NeoPixel.clear();
  NeoPixel.show();
  delay(500/freq);
}
