## EAI_Final_Project
Final project for the Embedded AI Course.

This is a proof-of-concept project to re-route trains in real time.
This could be used to route freight trains into sidings so passenger trains can pass them,
without weeks of planning beforehand.

## Code highlights

#### Installation & Execution Instructions
1. Demo code is run from <code>ArduinoInterface.py</code>,
this requires an ardiuno and a compatabile program for it.
The Serial port might need to be changed as well.
See <code>ArduinoCode/ArduinoCode.ino</code> for an example.
2. Model training and tweaking is done in <code>KerasAI.py</code>
Run this script to download the data from google images and train/test a new model on it.

A pretrained model and some test images are provided for convience.

Required Libraries (Python-Side):
+ TensorFlow
+ Keras
+ Pillow
+ PyCmdMessenger
+ Numpy

Required Libraries (Arduino-Side):
+ CmdMessenger

#### Arduino Setup
##### Requirements
###### Required Components
+ 1 x Arduino (preferably a Mega)
+ 1 x L298N motor controller
+ 1 x 12V DC Power supply
+ Some male-to-female and male-to-male wires

###### Optional Components(used for our demo)
+ 1 x Lego Powerfunctions Medium Motor
+ 1 x Lego Power Functions Extension Wire
+ 1 x Lego Power Functions-eanbled Train Set (optional)
+ 1 x Lego Switch Track (If running a train, you'll want 2)
+ Some Lego Track (If running a train you'll want enough for a full loop)
+ Some 0.35mm (28 American Wire Guage) wire (optionlity depends on how you choose to connect the Power Functions to arduino)
+ Access to various Lego bricks and Technic supplies (see bricklink: <https://www.bricklink.com/v2/main.page>)

###### Tools (to setup the demo)
+ Wire Cutters/Strippers
+ Very small screwdrivers (preferably jeweler's or similar size)
+ Needle-nose Pliers (preferably 90-degree angled)

##### Setup
Follow these tutorials, they're a little outdated, so your mileage my vary:
+ Arduino Setup: <https://www.youtube.com/watch?v=gFDe3nqLHl4&t=195s>
+ Lego Gearbox Setup: <https://youtu.be/h-5FmGfYzRs>