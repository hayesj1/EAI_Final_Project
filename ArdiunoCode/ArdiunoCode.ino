#include "CmdMessenger.h"
#include "string.h"
// Pins
const int ENA = 8; //points motor PWM
const int IN1 = 23; //points motor pin 1
const int IN2 = 24; //points motor pin 2

// Define switch point states
enum {
  UNKNOWN,
  MAIN,
  DIVERGING,
};

// Variables
byte switchPointState = UNKNOWN;

// Initialize CmdMessenger
// These should match PyCmdMessenger instance!
const int BAUD_RATE = 9600;
CmdMessenger c = CmdMessenger(Serial,',',';','/');

// Define available CmdMessenger commands
enum {
    oncomingPassenger,
    oncomingFreight,
    switchStatus,
    error,
};

/* Create callback functions to deal with incoming messages */
void onOncomingPassenger(void) { // Ensure the switch point is set to MAIN
  if (switchPointState != MAIN) {
    switchLeft();
    switchPointState = MAIN;
  }
}
void onOncomingFreight(void) { // Ensure the switch point is set to DIVERGING
  if (switchPointState != DIVERGING) {
    switchRight();
    switchPointState = DIVERGING;
  }
}
void onSwitchStatus(void) { // Return the switch point's state
  String res;
  switch (switchPointState) {
    case MAIN:
      res = "MAIN";
      break;
    case DIVERGING:
      res = "DIVERGING";
      break;
    default:
      res = "UNKNOWN";
      break;
  }
  
  c.sendCmd(switchStatus, res);
}
void on_unknown_command(void) { // Unknown Command got received 
  c.sendCmd(error, "Command without callback.");
}

/* Attach callbacks for CmdMessenger commands */
void attach_callbacks(void) { 
  c.attach(oncomingPassenger,onOncomingPassenger);
  c.attach(oncomingFreight,onOncomingFreight);
  c.attach(switchStatus, onSwitchStatus);
  c.attach(on_unknown_command);
}

/* switch to the left track */
void switchLeft(void) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  delay(1000);
  digitalWrite(IN1, LOW);
}
/* switch to the right track */
void switchRight(void) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  delay(1000);
  digitalWrite(IN2, LOW);
}

/* for reference */
/*
void Track_Switcher() {
  LDR2Value = analogRead(LDR2);  //reads the red LDR's value
    if (LDR2Value > lightSensitivity) { //if the sensor is triggered
      LDR2state = 1;   //save the sensor as ON
    }
  else   {
    LDR2state = 0;  //otherwise, sensor is OFF
  }

  if (LDR2state != previousLDR2state && previousLDR2state == 0) {  //if sensor just switched from OFF to ON
    LightSensorCounter++;    //add 1 to the light counter

    if (LightSensorCounter % 2 == 0) {     //if the light counter is an EVEN number
      digitalWrite(IN1, HIGH); // switch left
      digitalWrite(IN2, LOW);
      delay(200);
      digitalWrite(IN1, LOW);
    }
    else {                                //if the light counter is an ODD number
      digitalWrite(IN1, LOW); // switch right
      digitalWrite(IN2, HIGH);
      delay(200);
      digitalWrite(IN2, LOW);
    }

  }
  previousLDR2state = LDR2state;   //save light sensor state
}
*/
void setup() {
  // InitCmdMessenger
  Serial.begin(BAUD_RATE);
  attach_callbacks();

  // Init Pins
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  //Set switch point motor speed
  analogWrite(ENA, 255);

  switchLeft();
}

void loop() {
  c.feedinSerialData();
}
