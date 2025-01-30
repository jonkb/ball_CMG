/* esp32_CMGBall_bt.ino
Jon Black, 2024-11-04

Code to run on the ESP32 for the CMGBall

Summary: The ESP32 sets itself up as a Bluetooth BLE server, exposing sensor 
  measurements to the client as well as a writable characteristic for motor
  speeds. All the controls calculations should be done on the client computer.

References:
  L298N: https://brightspotcdn.byu.edu/cd/87/bbf866d84c06a0c52fa995396f30/l298n-motor-driver-quick-start-v6.pdf
  ESP32 PWM: https://randomnerdtutorials.com/esp32-dc-motor-l298n-motor-driver-control-speed-direction/
  ChatGPT conversation for BLE code
*/

// For BLE
#include <BLEDevice.h>
#include <BLEServer.h>
// For the MMA accel sensor
#include <Adafruit_MMA8451.h>
#include <Adafruit_Sensor.h>
// For writing to ESP32 pins
#include <Arduino.h>

// UUIDs for the BLE service and characteristics
#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define ACCEL_CHARACTERISTIC_UUID "abcdef01-1234-5678-1234-56789abcdef0"
#define SPEED_CHARACTERISTIC_UUID   "abcdef02-1234-5678-1234-56789abcdef0"
// Pins for L298N inputs
#define L298N_I1 26 // Motor A (gyro) fwd
#define L298N_I2 25 // Motor A (gyro) bkwd
#define L298N_I3 33 // Motor B (geared) fwd
#define L298N_I4 32 // Motor B (geared) bkwd
// Corresponding PWM channels
#define PWM_I1 0
#define PWM_I2 1
#define PWM_I3 2
#define PWM_I4 3
// Scaling factor for accel float (m/s^2) --> int16_t
#define ACCEL_SCALE 1000

/* Constants */
// PWM settings
const int pwm_frequency = 1000; // 1kHz
const int pwm_resolution = 8; // 8-bit resolution (0-255)
const int pwm_max = 255;

/* Variables */
// Is the BLE server running
bool live = false;
// Most recent accel reading (XYZ)
float accel[3] = {0.0, 0.0, 0.0};
// Current motor commands, between -1 and 1
float speeds[2] = {0.0, 0.0};
// Variables used in converting speeds to PWM values
int s1i;
int s2i;
/* Objects */
// Class for sensor
Adafruit_MMA8451 mma = Adafruit_MMA8451();
// BLE charasteristics for accel readings and motor speeds
BLECharacteristic accelCharacteristic(ACCEL_CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_NOTIFY);
BLECharacteristic speedCharacteristic(SPEED_CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_WRITE);

/* Function declarations */
void setup_accel();
void setup_motors();
void read_accel();
void update_speeds();


void setup() {
  Serial.begin(115200);
  delay(1000);
  
  setup_accel();
  setup_motors();
  
  // Initialize BLE server
  start_BLE();
  Serial.println("ESP32 BLE Server is running...");
}

void loop() {
  if (live) {
    read_accel();
    send_accel(accel);
  }
  update_speeds();
  delay(100); // For testing -- probably remove in production
}


class SpeedCallback : public BLECharacteristicCallbacks {
  // This callback function is triggered when the speed characteristic is written
  void onWrite(BLECharacteristic *pCharacteristic) {
    uint8_t *data = pCharacteristic->getData();
    size_t data_len = pCharacteristic->getLength();

    if (data_len == 8){
      // Store those bytes in speeds
      memcpy(speeds, data, 8);
      Serial.print("New speeds: ");
      Serial.print(speeds[0]);
      Serial.print(", ");
      Serial.println(speeds[1]);
    } else {
      Serial.print("Invalid server response.");
      // Print byte array
      Serial.print(" Bytes: [");
      for (int i=0; i<data_len; i++){
        Serial.print(data[i]);
        Serial.print(" ");
      } 
      Serial.println("]");
    }




    /* Convert to String, then back
    String value = pCharacteristic->getValue();
    Serial.print("Message String: ");
    Serial.println(value);

    // Store the response in the speeds array
    if (value.length() == 8) {
      // Create a byte array to hold the data (one longer because of null terminator)
      byte buf[9];
      // Convert the String to a byte array
      value.getBytes(buf, 9);
      Serial.print("Bytes: [");
      for (int i=0; i < 8; i++){
        Serial.print(buf[i]);
        Serial.print(" ");
      } 
      Serial.println("]");
      // Store those bytes in speeds
      memcpy(speeds, buf, 8);
      Serial.print("New speeds: ");
      Serial.print(speeds[0]);
      Serial.print(", ");
      Serial.println(speeds[1]);
    } else {
      Serial.print("Invalid server response: ");
      Serial.println(value);
    }
    */


    // OLD version using Strings
    // // Store the response in the given speeds array
    // //  Assumes that the response is formated as follows: "speed1,speed2"
    // try {
    //   int ix_comma = value.indexOf(',');
    //   speeds[0] = std::stod(value.substring(0, ix_comma).c_str());
    //   speeds[1] = std::stod(value.substring(ix_comma+1).c_str());
    //   Serial.print("Speeds updated to: ");
    //   Serial.println(value);
    // } catch(...) {
    //   Serial.print("Invalid server response: ");
    //   Serial.println(value);
    // }

  }
};

void start_BLE() {
  /** Set up and start Bluetooth server
  */

  // Set up Server & Service
  BLEDevice::init("ESP32 Sensor");
  BLEServer *pServer = BLEDevice::createServer();
  BLEService *pService = pServer->createService(SERVICE_UUID);
  
  // Add characteristics
  pService->addCharacteristic(&accelCharacteristic);
  pService->addCharacteristic(&speedCharacteristic);

  // Attach the callback to the rateCharacteristic
  speedCharacteristic.setCallbacks(new SpeedCallback());
  
  // Start the service
  pService->start();
  
  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();

  live = true;
}

void setup_accel() {
  /* Set up the MMA8451
  */
  if (! mma.begin()) {
    Serial.println("Couldn't connect to MMA8451");
    while (1);
  }
  Serial.println("Successfully connected to MMA8451");
  mma.setRange(MMA8451_RANGE_2_G);
  // Serial.print("Range = "); Serial.print(2 << mma.getRange());
  // Serial.println("G");
}

void setup_motors() {
  /* Set up the pins for the L298N motor driver
  */
  pinMode(L298N_I1, OUTPUT);
  pinMode(L298N_I2, OUTPUT);
  pinMode(L298N_I3, OUTPUT);
  pinMode(L298N_I4, OUTPUT);
  // PWM setup
  ledcAttachChannel(L298N_I1, pwm_frequency, pwm_resolution, PWM_I1);
  ledcAttachChannel(L298N_I2, pwm_frequency, pwm_resolution, PWM_I2);
  ledcAttachChannel(L298N_I3, pwm_frequency, pwm_resolution, PWM_I3);
  ledcAttachChannel(L298N_I4, pwm_frequency, pwm_resolution, PWM_I4);
  // PWM setup -- OLD
  //  See https://docs.espressif.com/projects/arduino-esp32/en/latest/migration_guides/2.x_to_3.0.html#ledc
  // ledcSetup(PWM_I1, pwm_frequency, pwm_resolution);
  // ledcAttachPin(L298N_I1, PWM_I1);
  // ledcSetup(PWM_I2, pwm_frequency, pwm_resolution);
  // ledcAttachPin(L298N_I2, PWM_I2);
  // ledcSetup(PWM_I3, pwm_frequency, pwm_resolution);
  // ledcAttachPin(L298N_I3, PWM_I3);
  // ledcSetup(PWM_I4, pwm_frequency, pwm_resolution);
  // ledcAttachPin(L298N_I4, PWM_I4);
}

void read_accel() {
  // Read data from the accelerometer and save to accel[]

  // Get a new sensor event
  sensors_event_t event; 
  mma.getEvent(&event);

  // Save the results (acceleration is measured in m/s^2)
  accel[0] = event.acceleration.x;
  accel[1] = event.acceleration.y;
  accel[2] = event.acceleration.z;
}

void update_speeds() {
  /* Update the speeds of the two motors.
  Pulls speeds from the speeds[] array
  */
  
  // Set motor speeds (PWM duty cycles)
  if (speeds[0] >= 0) { // fwd
    // Convert the speed to int, to be sent as PWM duty cycle.
    s1i = (int) ( speeds[0] * pwm_max );
    ledcWriteChannel(PWM_I1, s1i);
    ledcWriteChannel(PWM_I2, 0);
  }
  else { // rev
    s1i = (int) ( -speeds[0] * pwm_max );
    ledcWriteChannel(PWM_I1, 0);
    ledcWriteChannel(PWM_I2, s1i);
  }
  if (speeds[1] >= 0) { // fwd
    s2i = (int) ( speeds[1] * pwm_max );
    ledcWriteChannel(PWM_I3, s2i);
    ledcWriteChannel(PWM_I4, 0);
  }
  else { // rev
    s2i = (int) ( -speeds[1] * pwm_max );
    ledcWriteChannel(PWM_I3, 0);
    ledcWriteChannel(PWM_I4, s2i);
  }
}

void send_accel(float accel[3]) {
  /** send_accel(float accel[3])
  Send a message to the client with the given acceleration array accel[3]
  */

  // Convert the float array to a byte array
  // Calculate the size of the byte array
  size_t msg_size = 3*sizeof(accel[0]);
  // Create a byte array to hold accel
  uint8_t *byteArray = new uint8_t[msg_size];
  // Copy the float data to the byte array
  memcpy(byteArray, accel, msg_size);

  // Set the value of the characteristic to the byte array
  accelCharacteristic.setValue(byteArray, msg_size);
  accelCharacteristic.notify();  // Notify connected clients

  // Clean up
  delete[] byteArray;  // Free allocated memory
}

