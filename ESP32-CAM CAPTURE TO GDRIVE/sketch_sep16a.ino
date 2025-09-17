#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "Base64.h"
#include "esp_camera.h"

// CAMERA_MODEL_AI_THINKER GPIO
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Google Apps Script Deployment ID and folder
String myDeploymentID = "AKfycbwA2ttsfrACwfiy7yEMUZ-yRxHPkbt3-ahIfhqnW_iUkhAzBBnoMfpS2hubOTSPQjf8-g";
String myMainFolderName = "ESP32-Cam";

// Timer
unsigned long previousMillis = 0;
const int Interval = 20000; // 20 seconds

WiFiClientSecure client;

// ------------------------- Camera Initialization -------------------------
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size = FRAMESIZE_SXGA; // 1280x1024
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    delay(1000);
    ESP.restart();
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_framesize(s, psramFound() ? FRAMESIZE_SXGA : FRAMESIZE_VGA);
  s->set_quality(s, 12);
  s->set_contrast(s, 0);
  s->set_brightness(s, 0);
  s->set_saturation(s, 0);
  s->set_gainceiling(s, GAINCEILING_2X);
  s->set_whitebal(s, 1);
  s->set_exposure_ctrl(s, 1);
  s->set_awb_gain(s, 1);

  Serial.println("Camera initialized successfully.");
}

// ------------------------- WiFi SmartConfig -------------------------
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  Serial.println("Waiting for WiFi credentials via SmartConfig...");

  WiFi.beginSmartConfig();
  while(!WiFi.smartConfigDone()){
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nSmartConfig received.");

  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print("*");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

// ------------------------- Test Google Script Connection -------------------------
void Test_Con() {
  const char* host = "script.google.com";
  Serial.println("Testing connection to Google Script...");
  client.setInsecure();
  if(client.connect(host, 443)){
    Serial.println("Connection successful.");
    client.stop();
  } else {
    Serial.println("Connection failed.");
    client.stop();
  }
}

// ------------------------- Send Photo to Google Drive -------------------------
void SendCapturedPhotos() {
  const char* host = "script.google.com";
  Serial.println("-----------");
  Serial.println("Connecting to Google Drive...");

  client.setInsecure();

  if(client.connect(host, 443)){
    Serial.println("Connection successful.");

    camera_fb_t * fb = esp_camera_fb_get();
    if(!fb){
      Serial.println("Camera capture failed. Restarting...");
      delay(1000);
      ESP.restart();
      return;
    }
    Serial.println("Photo captured: " + String(fb->len) + " bytes");

    String url = "/macros/s/" + myDeploymentID + "/exec?folder=" + myMainFolderName;
    client.println("POST " + url + " HTTP/1.1");
    client.println("Host: " + String(host));
    client.println("Transfer-Encoding: chunked");
    client.println();

    int fbLen = fb->len;
    char *input = (char *)fb->buf;
    int chunkSize = 3 * 1000;
    int chunkBase64Size = base64_enc_len(chunkSize);
    char output[chunkBase64Size + 1];

    for(int i = 0; i < fbLen; i += chunkSize){
      int l = base64_encode(output, input, min(fbLen-i, chunkSize));
      client.print(l, HEX);
      client.print("\r\n");
      client.print(output);
      client.print("\r\n");
      input += chunkSize;
      delay(100);
      Serial.print(".");
    }
    client.print("0\r\n\r\n");
    esp_camera_fb_return(fb);

    long int StartTime = millis();
    while(!client.available()){
      Serial.print(".");
      delay(100);
      if((StartTime + 10000) < millis()){
        Serial.println("\nNo response.");
        break;
      }
    }
    Serial.println();
    while(client.available()) Serial.print(char(client.read()));

  } else {
    Serial.println("Connection failed.");
  }

  Serial.println("-----------");
  client.stop();
}

// ------------------------- Setup -------------------------
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32-CAM starting up...");

  connectWiFi();
  initCamera();
  Test_Con();

  Serial.println("ESP32-CAM captures and sends photos every 20 seconds.");
  previousMillis = millis() - Interval;
}

// ------------------------- Loop -------------------------
void loop() {
  unsigned long currentMillis = millis();
  if(currentMillis - previousMillis >= Interval){
    previousMillis = currentMillis;
    SendCapturedPhotos();
  }
}
