import os
from time import sleep

import numpy as np
import PyCmdMessenger
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from serial import SerialException

from KerasAI import load, seed, normalizeImg

_PROJ_NAME = "Cargo Classifier For Real Time Train Routing"
_NAME = "Jacob Hayes"
_COLLABORATORS = "Josh Beaulieu"

class ArduinoInterface:
	def __init__(self):
		self._serial = "/dev/cu.usbmodem1411" # You might need to configure this. Use the Arduino IDE to find out which serial port Arduino is on

		# Initialize an ArduinoBoard instance.  This is where you specify baud rate and
		# serial timeout.  If you are using a non ATmega328 board, you might also need
		# to set the data sizes (bytes for integers, longs, floats, and doubles).
		self._arduino = PyCmdMessenger.ArduinoBoard(self._serial, baud_rate=9600)

		# List of commands and their associated argument formats. These must be in the
		# same order as in the sketch.
		self._oncomingPassenger = "oncomingPassenger"
		self._oncomingFreight = "oncomingFreight"
		self._switchStatus = "switchStatus"
		self._error = "error"

		self._commands = [[self._oncomingPassenger, ""],
		            [self._oncomingFreight, ""],
		            [self._switchStatus, "s"],
		            [self._error, "s"]]

		# Initialize the messenger
		self.com = PyCmdMessenger.CmdMessenger(self._arduino, self._commands)

	def receive(self):
		return self.com.receive()
	def sendError(self):
		self.com.send(self._error)
	def sendSwitchStatus(self):
		self.com.send(self._switchStatus)
	def sendOncomingPassenger(self):
		self.com.send(self._oncomingPassenger)
	def sendOncomingFreight(self):
		self.com.send(self._oncomingFreight)

if __name__ == '__main__':
	path = "data/trains/"
	imgDir = "real"

	try:
		ardInt = ArduinoInterface()
	except SerialException as se:
		print(se)
		ardInt = None
	
	model = load("model")
	dataGen = ImageDataGenerator(preprocessing_function=normalizeImg)

	# Send
	ardInt and ardInt.sendSwitchStatus()
	msg = ardInt and ardInt.receive()
	print(msg or "No Arduino Connection")

	ran = False
	while True:
		imgs  = [ img for img in os.listdir(os.path.abspath('./'+path+imgDir+"/test")) ]

		if len(imgs) > 0:
			data = np.empty(shape=(len(imgs), 112, 112, 3))
			i = 0
			for img in imgs:
				tmp = load_img(path+imgDir+"/test/"+img, target_size=(112,112))
				#tmp.thumbnail((112,112))
				x = img_to_array(tmp)
				data[i] = x
				i += 1
				
			genTmp = dataGen.flow(x=data , y=None, shuffle=False, seed=seed)
			probs = model.predict_generator(genTmp, verbose=0)
			if probs[:, 1].sum() >= probs[:, 0].sum():
				ardInt and ardInt.sendOncomingPassenger()
				print("Switch set for Passenger Train")
			else:
				ardInt and ardInt.sendOncomingFreight()
				print("Switch set for Freigh Train")

			ardInt and ardInt.sendSwitchStatus()
			msg = ardInt and ardInt.receive()
			print(msg or "No Arduino Connection")

			for i in range(len(imgs)):
				print(imgs[i])
				print("  F  |  P  ")
				print("%3.2f | %3.2f" % (probs[i, 0]*100, probs[i, 1]*100))
				os.remove(os.path.abspath('./'+path+imgDir+"/test/"+imgs[i]))
			
			probs = None
		
		#print("Cycle Complete") # debug code
		sleep(1.0)