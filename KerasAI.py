# coding=utf-8
import os, warnings, sys
from time import sleep
from typing import List, Dict, Tuple

# to suppress tensorflow warnings on 'speed up'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore", message="This ImageDataGenerator specifies", category=UserWarning, lineno=959)
#sys.path.append(os.environ['CUDA_PATH']) #use if Tensorflow is having issues finding CUDA

import numpy as np
import keras as k
from keras.applications.xception import Xception
from keras.callbacks import History
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

seed = 42
path = "data/trains/"
subsets = ["train", "valid", "test"]
classes = ["passenger", "freight"]
keywords = ["'passenger' 'train' -freight -toy -book -derail -thomas -lego", "'freight' 'train' -passenger -toy -book -derail -thomas -lego"]
limit = 1000

adam = Adam(lr=0.0001)

def loadImageData() -> None:
	import re
	from google_images_download import google_images_download as gid
	def cleanImgData() -> None:
		for subset in subsets:
			for clazz in classes:
				files = os.listdir(path=f'./{path}{subset}/{clazz}/')
				for file in files:
					if os.path.isfile(f'{path}{subset}/{clazz}/'+file):
						os.remove(f'{path}{subset}/{clazz}/'+file)
	def getImgData() -> List[Dict[str, str]]:
		response = gid.googleimagesdownload()
		absoluteImgPaths = []
		arguments = { "format":"jpg", "type":"photo", "limit":limit, "output_directory":path, "chromedriver":"./chromedriver/chromedriver.exe" }
		for subset in subsets[0:1]:
			for kw, clazz in zip(keywords, classes):
				arguments["keywords"] = kw
				arguments["image_directory"] = f'{subset}/{clazz}/'
				arguments["prefix"] = clazz
				absoluteImgPaths.append(response.download(arguments))
		return absoluteImgPaths
	def splitImgData() -> None:
		for i, subset in enumerate(subsets):
			if i == 0: continue
			for clazz in classes:
				files = os.listdir(path=f'./{path}{subsets[i-1]}/{clazz}/')
				nFiles = len(files)
				n = (nFiles // 2)
				for file in files[n:]:
					os.rename(os.path.abspath(f'./{path}{subsets[i-1]}/{clazz}/'+file), os.path.abspath(f'./{path}{subset}/{clazz}/'+file))
	def preprocessImgData() -> None:
		for subset in subsets:
			for clazz in classes:
				files = os.listdir(path=f'./{path}{subset}/{clazz}/')
				for i, file in enumerate(files):
					newFile = file
					if file.find("\.jpeg") >= 0:
						newFile = file.replace("\.jpeg", "\.jpg")
					elif file.find("\.png") >= 0:
						newFile = file.replace("\.png", "\.jpg")
					
					if newFile.find(".jpg") < 0:
						print("No extension!", newFile)
					elif re.match(re.compile(f'{clazz}'+'\\d+\.jpg$'), newFile):
						continue
					else:
						newFile = newFile.replace(" ", "")
						# newFile = newFile.replace(" ", "_")
						newFile = newFile.replace(newFile, f'{clazz}{i+1}.jpg')
						if len(newFile) > newFile.find(".jpg")+4:
							newFile = newFile.replace(newFile[ (newFile.find(".jpg")+4): ], "")
						#print(file, "|", newFile);
						os.rename(os.path.abspath(f'./{path}{subset}/{clazz}/'+file), os.path.abspath(f'./{path}{subset}/{clazz}/'+newFile))

	cleanImgData()
	getImgData()
	splitImgData()
	preprocessImgData()

def normalizeImg(img:np.ndarray) -> np.ndarray:
	return (img - 128) / 128

def getData() -> Tuple[ DirectoryIterator, ... ]:
	for subset in subsets:
		files = os.listdir(path+"transforms/"+subset)
		for file in files:
				os.remove(path+"transforms/"+subset+"/"+file)
		#os.removedirs(path+"transforms/"+subset)
	
	ret = []
	paramsList = [
						{
							#"rescale": 1./255,
							"zoom_range": 0.5,
							"horizontal_flip": True,
							"vertical_flip": True,
						},
						{
							#"rescale": 1./255,
						},
						{
							#"rescale": 1./255,
						}
					 ]
	for subset, params in zip(subsets, paramsList):
		dataGen = ImageDataGenerator(**params)
		gen = dataGen.flow_from_directory(path+subset, target_size=(112,112), class_mode="categorical", batch_size=32, shuffle=True, seed=seed, save_to_dir=path+"transforms/"+subset, save_format="jpg")
		ret.append(gen)

	return tuple(ret)

def makeModel(genTrn:DirectoryIterator, genVld:DirectoryIterator) -> k.Model:
	base = Xception(include_top=False, pooling="avg", input_shape=(112,112,3))
	last = base.output
	last = Dense(128, activation='relu')(last)
	last = Dense(128, activation='relu')(last)
	predictions = Dense(2, activation='softmax')(last)
	
	m = k.Model(inputs=base.input, outputs=predictions)
	for layer in base.layers:
		layer.trainable = False
	
	m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	hist = m.fit_generator(genTrn, epochs=16, validation_data=genVld, shuffle=True, use_multiprocessing=False, workers=1, verbose=0).history
	print(f"Training Accuracy:\t{hist['acc'][-5:]}")
	print(f"Training Loss:\t\t{hist['loss'][-5:]}")
	print(f"Validation Accuracy:\t{hist['val_acc'][-5:]}")
	print(f"Validation Loss:\t{hist['val_loss'][-5:]}")
	#for i, layer in enumerate(base.layers):
	#	print(i, layer.name)
	sleep(10.0)
	for layer in m.layers[:126]:
		layer.trainable = False
	for layer in m.layers[126:]:
		layer.trainable = True
	
	m.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
	return m

def trainModel(m:k.Model, genTrn:DirectoryIterator, genVld:DirectoryIterator) -> History:
	hist = m.fit_generator(genTrn, epochs=128, validation_data=genVld, shuffle=True, use_multiprocessing=False, workers=1, verbose=1)
	return hist

def testModel(m:k.Model, genTst:DirectoryIterator) -> np.ndarray:
	metrics = m.evaluate_generator(genTst, use_multiprocessing=False, workers=1)
	return metrics

def printArray(arr:np.ndarray) -> None:
	for row in arr:
		print("%3.2f | %3.2f" % (row[0]*100, row[1]*100))

def load(name:str) -> k.Model:
	# load json and create model
	jsonF = open(f'{path}models/{name}.json', 'r')
	lm_json = jsonF.read()
	jsonF.close()
	lm = model_from_json(lm_json)
	# load weights into new model
	lm.load_weights(f'{path}/models/{name}.h5')
	print("Loaded model from disk")
	
	m = lm
	m.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
	
	return m
	
def loadAndTest(name:str) -> k.Model:
	m = load(name)
	predictions = testModel(m, genTest)
	printArray(predictions)
	
	return m

if __name__ == '__main__':
	#uncomment to fetch imgs from google
	#loadImageData()
	genTrain, genValid, genTest = getData()

	#sys.exit(1) # debug code
	model = makeModel(genTrain, genValid)
	hist = trainModel(model, genTrain, genValid).history
	print("")
	print(genTrain.samples)
	print(hist.keys())
	print(f"Training Accuracy:\t{hist['acc'][-5:]}")
	print(f"Training Loss:\t\t{hist['loss'][-5:]}")
	print(f"Validation Accuracy:\t{hist['val_acc'][-5:]}")
	print(f"Validation Loss:\t{hist['val_loss'][-5:]}")

	#serialize model to JSON
	model_json = model.to_json()
	with open(f'{path}/models/test_model.json', 'w') as json_file:
		json_file.write(model_json)
	#serialize weights to HDF5
	model.save_weights(f'{path}/models/test_model.h5')
	print("Saved model to disk")
	
	
	# load json and create model
	with open(f'{path}/models/test_model.json', 'r') as json_file:
		loaded_model_json = json_file.read()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(f'{path}/models/test_model.h5')
	print("Loaded model from disk")
	
	model = loaded_model
	model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
	metrics = testModel(model, genTest)

	print(metrics)
