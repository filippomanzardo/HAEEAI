// Description: Main file for the AI Plant project

// Include TinyDatabase to store data in the persistent memory
#include <TinyDatabase.h>
// For datetime_utc column
#include <time.h>

// Moisture sensor pin
const int moisture_sensor  = A0;
// Temperature sensor pin
const int temperature_sensor = A1;
// MemoryManager object to access data base. It stores in a CSV file with SQL-like syntax
const MemoryManager mem;

// Table name
const char* table_name = "aiplant_data";
// datetime_utc column
const char* datetime_utc = "datetime_utc";
// moisture column
const char* moisture = "moisture";
// temperature column
const char* temperature = "temperature";


void setup() {

  // Set data rate to 9600 bps
  Serial.begin(9600);

  pinMode(moisture_sensor, INPUT);
  // Create array of columns
  Column myCols[] = {{datetime_utc, "INT"}, {moisture, "INT"}, {temperature, "INT"}};

  // Create table if it is not created yet and provide capacity, number of columns and table name
  int isTableCreated = mem.CREATE_TABLE(table_name, 15, 2, myCols);

  // 3- Verify if table is created
  // by compare the returned value to a status code
  if (isTbaleCreated != STATUS_TABLE_CREATED) {
    Serial.println("There was an error while creating the `aiplant` table, exiting...");
    exit(1);
  }

  // Display all informations about the meta data of the database
  mem.printMetaData();
}



void loop() {
  // Simply read the value from the sensor and store it in the persistent memory
  acquireAndStoreData();
}


// * Function: acquireAndStoreData
// * Description: Acquires data from the sensors and stores it in the persistent memory
// * Parameters: None
// * Returns: None
void acquireAndStoreData() {

      // Read the value from the moisture sensor
      int moisture_value = analogRead(moisture_sensor);

      // Read the value from the temperature sensor
      int temperature_value = analogRead(temperature_sensor);

      // Get current time in UTC integer format
      int current_time = time(NULL);

      // Store the values in the persistent memory
      mem.TO(table_name).INSERT(datetime_utc, &current_time).INSERT(moisture, &moisture_value).INSERT(temperature, &temperature_value).DONE();

      // Print the values to the serial monitor
      Serial.print("Moisture: ");
      Serial.println(moisture_value);
      Serial.print("Temperature: ");
      Serial.println(temperature_value);

      // Wait 5 seconds
      delay(5000);
}
