# ESP32-CAM Dynamic WiFi Configuration

This sketch allows you to use an ESP32-CAM to take photos and upload them to Google Drive without hardcoding WiFi credentials.

## Features

- Dynamic WiFi configuration through a web portal
- Automatically connects to the last saved WiFi network
- Manual configuration mode trigger via a hardware button
- Uploads photos to Google Drive automatically
- LED status indicators for different operations

## Hardware Requirements

- ESP32-CAM module
- FTDI programmer or USB-to-Serial adapter for uploading
- Push button connected to GPIO 33 and GND (for manual configuration trigger)
- 5V power supply

## Installation

1. Install the required libraries in Arduino IDE:
   - WiFiManager by tzapu (install via Library Manager)
   
2. Upload the sketch to your ESP32-CAM using an FTDI programmer or USB-to-Serial adapter

## WiFi Configuration

When the ESP32-CAM first boots up or if it can't connect to a previously saved WiFi network, it will create its own WiFi access point:

1. Connect to the WiFi network named "ESP32-CAM-Config-XXXXXX" (where XXXXXX is a unique identifier)
2. Password: "password"
3. A configuration portal will automatically open (or navigate to 192.168.4.1 in your browser)
4. Enter your WiFi credentials and save
5. The ESP32-CAM will connect to your WiFi network and begin normal operation

## Manual Configuration Trigger

You can force the ESP32-CAM into configuration mode anytime:

1. Press the button connected to GPIO 33 when the device is starting up
2. OR hold the button for more than 5 seconds during normal operation to reset WiFi settings

## LED Status Indicators

- 3 quick flashes at startup: Device is booting up
- 5 quick flashes: Entered WiFi configuration mode
- 3 long flashes: Successfully connected to WiFi
- 10 quick flashes: WiFi settings reset
- 1 flash followed by continuous operation: Photo capture and upload in progress

## Troubleshooting

If the ESP32-CAM is not connecting to your WiFi:
1. Check if your WiFi network is operational
2. Force configuration mode by pressing the button on GPIO 33
3. Make sure you enter the correct WiFi credentials
4. Try positioning the ESP32-CAM closer to your WiFi router

## Modification

You can customize the code:
- Change the access point name and password in the defines at the top of the sketch
- Adjust the configuration portal timeout (default is 120 seconds)
- Modify the GPIO pin used for the configuration button
- Change the photo capture interval (default is 20 seconds)