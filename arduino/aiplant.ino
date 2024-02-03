#include <chrono>

#include <ArduinoBLE.h>
#include <Arduino_LPS22HB.h>
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"


#define TFLITE_SCHEMA_VERSION (3)

using namespace std;
using namespace std::chrono;

/**
======================== Moisture Simulation ========================
**/
// Indicate the start point for a constant decay
system_clock::time_point lastWateringTime;
// Average value taken from the training dataset
double initialMoisture = 1200.0;
// Arbitrary decay rate
double decayRate = 15;

/**
  * Simulate the current moisture as a linearly decresing value
  * @param currentTime the current system time
  * @return the current simulated value of moisture in ml
*/
double simulateMoisture(const system_clock::time_point& currentTime) {
    // Calculate the time difference in seconds

    auto timeDifference = duration_cast<seconds>(currentTime - lastWateringTime).count();

    // Simulate moisture decay using a simple linear decay model
    double currentMoisture = initialMoisture - (decayRate * timeDifference);

    if (currentMoisture < 0.0) {
      lastWateringTime = system_clock::now();
    }

    // Ensure the moisture does not go below zero
    currentMoisture = max(0.0, currentMoisture);

    return currentMoisture;
}

/**
======================== Low-energy Bluetooth Global Variables ========================
**/
String TEMPERATURE;
String MOISTURE;

// This values will determine the first 8 hexadecimal digits, used to recognize the characteristics by the Server
const char* MOISTURE_UUID = "1000";
const char* TEMPERATURE_UUID = "2000";
const char* RECEIVER_UUID = "3000";

BLEStringCharacteristic BLE_MOISTURE(MOISTURE_UUID, BLERead | BLENotify, 32);
BLEStringCharacteristic BLE_TEMPERATURE(TEMPERATURE_UUID, BLERead | BLENotify, 32);

// 512-byte is the max transmission size for sending buffers with response
BLECharacteristic BLE_MODEL(RECEIVER_UUID, BLEWrite | BLENotify, 512);

/**
======================== TFLite Model Global Variables ========================
**/

// Behavioral boolean values
boolean shouldLoadModel = false; // controls the loading of the model
bool modelReceived = false;  // controls the end of the model streaming
int numberOfChunks = 0;  // controls the model reset
bool shouldWater = false; // wheter to release the pump

// Usual TFLite initialization variables
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
// Input has 2 elements: (temperature, moisture)
// Output has 2 element: (stay, water)
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Actual Model implementation
unsigned char* tfModel;

// Streaming buffer
const int MAX_MODEL_SIZE = 5000; // 10,000 bytes
int8_t buffer[MAX_MODEL_SIZE];
int modelSize = 0;

/**
======================== Arduino Setup ========================
**/
void setup() {
  // Initialize BLE Service
  BLEService customService("AIPlant");

  // Initalizing the BARO sensor
  BARO.begin();
  if (!BARO.begin()) {
    Serial.println("Failed to initialize pressure sensor!");
    while (1);
  }

  // Initialize the led sensor, simulating the action of the water pump
  pinMode(LED_BUILTIN, OUTPUT);

  // Initial Watering
  lastWateringTime = system_clock::now();

  // Initialize Serial
  Serial.begin(9600);
  while (!Serial);

  // Start BLE
  if (!BLE.begin())
  {
      Serial.println("BLE failed to Initiate");
      delay(500);
      while (1);
  }

  // Setting BLE Name
  BLE.setDeviceName("AIPlant");
  BLE.setLocalName("AIPlant");

  // Setting BLE Service Advertisment
  BLE.setAdvertisedService(customService);

  // Adding characteristics to BLE Service Advertisment
  customService.addCharacteristic(BLE_MOISTURE);
  customService.addCharacteristic(BLE_TEMPERATURE);
  customService.addCharacteristic(BLE_MODEL);

  // Adding the service to the BLE stack
  BLE.addService(customService);

  // Start advertising
  BLE.advertise();
  Serial.println("Bluetooth device is now active, waiting for connections...");
}

/**
======================== Loop Definition ========================
**/

void loop() {

  BLEDevice aiPlant = BLE.central();
  if (aiPlant) {
      Serial.print("Connected to aiPlant: ");
      Serial.println(aiPlant.address());

      // Wait for Server connection
      while (aiPlant.connected()) {
          // Read values from sensors
          readValues();
          // Write the sensor values into the BLE Characteristic -> Server will receive them
          BLE_MOISTURE.writeValue(MOISTURE);
          BLE_TEMPERATURE.writeValue(TEMPERATURE);
          // Conditionally load the inference model
          loadModel();

          // If server started sending the inference model
          if(BLE_MODEL.written()) {
            Serial.println("!! Received Model chunk !!");
            // Reset the modelReceived control -> Disable inference
            modelReceived = false;

            // Store the received chunk and its length
            const uint8_t* receivedChunk = BLE_MODEL.value();
            size_t chunkLength = BLE_MODEL.valueLength();
            storeChunk(receivedChunk, chunkLength);
          }

          shouldWater = doInference();

          printLoopInfo();
          if (shouldWater) {
            doWaterPlan();
          }

          delay(500);
      }
  }

  // The Server is not connected, wait in semi-idle
  Serial.print("Disconnected from central: ");
  Serial.println(aiPlant.address());

  // wait 5 second to print again
  delay(5000);
}


/**
======================== Logical Utility Functions Definition ========================
**/


/**
  Read the temperature and moisture values from the sensor.
  Store them to be sent to the Server
*/
void readValues() {
  system_clock::time_point currentTime = system_clock::now();
  MOISTURE = String(simulateMoisture(currentTime));
  BARO.readPressure();
  TEMPERATURE = String(BARO.readTemperature());
}

/**
  Load the model and the needed dependencies (e.g. the resolver).
  This function is controlled by the `shouldLoadModel` boolean flag.
*/
void loadModel() {

  if (!shouldLoadModel) {
    return;
  }

  printReceivedModel();

  // Load the model
  shouldLoadModel = false;

  model = tflite::GetModel(tfModel);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error while loading the model: Unexpected tflite schema version. Exiting...");
    exit(1);
  }

  // This pulls in all the operation implementations we need.
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  modelReceived = true;

  Serial.println("Model Loaded!");
}

/**
  Run inference on the current (moisture, temperature) values.

  @return whether we should water the plan
*/
bool doInference() {
  if (!modelReceived) {
    return false;
  }
  float temp_val = TEMPERATURE.toFloat();
  float moisture_val = MOISTURE.toFloat();

  input->data.f[0] = temp_val;
  input->data.f[1] = moisture_val;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.print("Error while running inference, ignoring loop...");
    return false;
  }

  // Softmax layer
  float stay = output->data.f[0];
  float water = output->data.f[1];

  printInferenceDetails(temp_val, moisture_val, stay, water);

  // The output is a sigmoid function, so we need to check if the value is greater than 0.5
  return water > stay;
}

/**
  Store the received chunk into the buffer.
  If the received chunk equals `END__OF__MODEL__SEQUENCE`, then we assume the trasmission
  ends and we set it as successful.

  @param chunk the received characteristic chunk
  @param chunkLength the length of the chunk
*/
void storeChunk(const uint8_t* chunk, size_t chunkLength) {

  // if it's the first cycle, we reset the stored model
  if (!numberOfChunks) {
    modelSize = 0;
  }

  // end trasmissions variables
  const char* endSequence = "END__OF__MODEL__SEQUENCE";
  size_t endSequenceLength = strlen(endSequence);

  // Last chunk logic
  if (chunkLength >= endSequenceLength && memcmp(chunk, endSequence, endSequenceLength) == 0) {
        shouldLoadModel = true;
        Serial.println("Last chunk received, proceeding to load model.");
        tfModel = new unsigned char[modelSize];

        // Copy the buffer content into the tfModel pointer
        memcpy(tfModel, buffer, modelSize);

        numberOfChunks = 0;
        return;
  }

  if (modelSize + chunkLength < MAX_MODEL_SIZE) {
    memcpy(&buffer[modelSize], chunk, chunkLength);
    modelSize += chunkLength;
    numberOfChunks += 1;

    printChunkInfo();

  } else {
    Serial.println("Model size exceeds the limit. Increase MAX_MODEL_SIZE.");
    exit(1);
  }
}

/**
  This function simulates the watering of the plant (via the LED)
*/
void doWaterPlan() {
  Serial.println("Watering the plant for 5 seconds!");
  digitalWrite(LED_BUILTIN, RISING);
  delay(5000);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.println("Plant Watered!");
  lastWateringTime = system_clock::now();
}

/**
======================== Print Functions Definition ========================
**/

void printLoopInfo() {
  char *outputBuff = new char[200];

  sprintf(outputBuff, "Current loop information:\n    Temperature: %s\n    Moisture: %s\n    Should Water the Plant: %s\n", TEMPERATURE.c_str(), MOISTURE.c_str(), shouldWater ? "true" : "false");
  Serial.println(outputBuff);

  delete[] outputBuff;
}

void printReceivedModel() {
  char *outputBuff = new char[50];
  sprintf(outputBuff, "Received Model of size: %d\n", modelSize);
  Serial.println(outputBuff);
  for (int i = 0; i < modelSize; ++i) {
    Serial.print(tfModel[i], HEX);
    Serial.print(" ");
  }
  Serial.println();

  delete[] outputBuff;
}

void printInferenceDetails(float inputTemp, float inputMoisture, float stay, float water) {
  char *outputBuff = new char[400];

  sprintf(outputBuff, "\nInference details (temp, moisture) -> (stay, water)\n    (%.2f, %.2f) -> (%.2f, %.2f)\n", inputTemp, inputMoisture, stay, water);
  Serial.println(outputBuff);

  delete[] outputBuff;
}

void printChunkInfo() {
  char *outputBuff = new char[200];

  sprintf(outputBuff, "Buffer Info:\n    Chunk number: %d\n    Buffer Location: %p\n    Total Buffer Size (bytes): %d\n", numberOfChunks, buffer, modelSize);
  Serial.println(outputBuff);

  delete[] outputBuff;
}

