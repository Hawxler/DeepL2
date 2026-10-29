#include <Arduino.h>
#include <WiFi.h>
#include <Network.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ---------- Wi-Fi 설정 ----------
const char* WIFI_SSID = "iptime2.4G";
const char* WIFI_PASSWORD = "12345678";

// ---------- 측정 설정 ----------
constexpr uint16_t PORT = 5000;
constexpr int SAMPLE_COUNT = 100;
constexpr uint32_t SAMPLE_PERIOD_US = 20000;  // 20ms = 50Hz

Adafruit_MPU6050 mpu;
NetworkServer server(PORT);

struct Sample {
  uint32_t t_us;
  float ax, ay, az;
  float gx, gy, gz;
};

Sample samples[SAMPLE_COUNT];

void captureGesture() {
  uint32_t start = micros();

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    uint32_t target = start + i * SAMPLE_PERIOD_US;

    // micros()의 wrap-around를 고려한 비교
    while ((int32_t)(micros() - target) < 0) {
      delayMicroseconds(100);
    }

    sensors_event_t a, g, temp;

    samples[i].t_us = micros() - start;
    mpu.getEvent(&a, &g, &temp);

    samples[i].ax = a.acceleration.x;
    samples[i].ay = a.acceleration.y;
    samples[i].az = a.acceleration.z;

    samples[i].gx = g.gyro.x;
    samples[i].gy = g.gyro.y;
    samples[i].gz = g.gyro.z;
  }

  // 마지막 측정은 약 1.98초, 전체 창은 2초
  while ((uint32_t)(micros() - start) <
         SAMPLE_COUNT * SAMPLE_PERIOD_US) {
    delay(1);
  }
}

void sendGesture(NetworkClient& client) {
  client.println("BEGIN,100");

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    const Sample& s = samples[i];

    client.printf(
      "%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
      (unsigned long)s.t_us,
      s.ax, s.ay, s.az,
      s.gx, s.gy, s.gz
    );
  }

  client.println("END");
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(21, 22);  // SDA, SCL
  Wire.setClock(400000);

  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("MPU6050 not found. Check wiring.");
    while (true) {
      delay(1000);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  WiFi.setSleep(false);
  server.begin();

  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("TCP port: ");
  Serial.println(PORT);
}

void loop() {
  NetworkClient client = server.accept();

  if (!client) {
    delay(10);
    return;
  }

  client.setTimeout(3000);
  client.setNoDelay(true);

  Serial.println("PC connected.");

  while (client.connected()) {
    if (!client.available()) {
      delay(1);
      continue;
    }

    String command = client.readStringUntil('\n');
    command.trim();

    if (command == "GO") {
      Serial.println("Recording...");
      captureGesture();
      sendGesture(client);
      Serial.println("Sent.");
    } else {
      client.println("ERROR,unknown_command");
    }
  }

  client.stop();
  Serial.println("PC disconnected.");
}