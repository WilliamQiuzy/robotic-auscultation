// V8B Hall displacement reader.
//
// Wiring:
//   Hall VCC -> 3V3 or 5V, depending on the Hall sensor
//   Hall GND -> GND
//   Hall OUT -> A0
//
// Serial output matches force_feedback_monitor.py:
//   {"t_ms":1234,"disp_mm":8.000,"raw":2048}

#include <math.h>

#if defined(ARDUINO_ARCH_RP2040)
const int ADC_BITS = 12;
const int ADC_MAX = 4095;
#else
const int ADC_BITS = 10;
const int ADC_MAX = 1023;
#endif

const int HALL_PIN = A0;
const unsigned long SAMPLE_PERIOD_MS = 10;  // 100 Hz.

// Replace these after calibration.
// ADC_ZERO is the no-contact installed reading.
// ADC_AT_8MM is the reading after mechanically compressing the carriage 8.0 mm.
float ADC_ZERO = 2048.0;
float ADC_AT_8MM = 2600.0;
const float CAL_TRAVEL_MM = 8.0;

unsigned long last_sample_ms = 0;

float read_adc_average() {
  const int n = 16;
  long sum = 0;
  for (int i = 0; i < n; i += 1) {
    sum += analogRead(HALL_PIN);
    delayMicroseconds(200);
  }
  return (float)sum / (float)n;
}

float adc_to_displacement_mm(float raw) {
  float delta = ADC_AT_8MM - ADC_ZERO;
  if (fabs(delta) < 1.0) {
    return 0.0;
  }
  float disp = (raw - ADC_ZERO) * CAL_TRAVEL_MM / delta;
  if (disp < 0.0) {
    disp = 0.0;
  }
  return disp;
}

void setup() {
#if defined(ARDUINO_ARCH_RP2040)
  analogReadResolution(ADC_BITS);
#endif
  Serial.begin(115200);
  while (!Serial && millis() < 2500) {
    delay(10);
  }
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample_ms < SAMPLE_PERIOD_MS) {
    return;
  }
  last_sample_ms = now;

  float raw = read_adc_average();
  float disp_mm = adc_to_displacement_mm(raw);

  Serial.print("{\"t_ms\":");
  Serial.print(now);
  Serial.print(",\"disp_mm\":");
  Serial.print(disp_mm, 3);
  Serial.print(",\"raw\":");
  Serial.print(raw, 1);
  Serial.println("}");
}
