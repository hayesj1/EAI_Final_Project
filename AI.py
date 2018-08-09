
# coding: utf-8
import os

import numpy as np

from typing import List, Dict, Union

from fastai.conv_learner import *

arch = resnet34
size=224
tfms = tfms_from_model(arch, size, aug_tfms=transforms_side_on, max_zoom=1.1)


class CNN:
	learner: ConvLearner
	optimalLR: Union[float, List[float]]

	dataPath: str
	testPath: str
	data: ImageClassifierData
	nTestData: int
	predictions: List[int]
	probabilities: List[float]
	testPredictions: List[int]
	testProbabilities: List[float]

	created: bool
	fitted: bool
	lrOptimized: bool
	predictedData: bool
	predictedTestData: bool

	def __init__(self, create:bool=False, optimize:bool=False, dataPath:str="/data", data:ImageClassifierData=None, testDir:str="test", **kwargs) -> None:
		self.learner = None
		self.optimalLR = 0.01

		self.dataPath = dataPath
		self.testPath = testDir
		self.data = data
		self.nTestData = 0
		self.predictions = []
		self.probabilities = []
		self.testPredictions = []
		self.testProbabilities = []

		self.created = False
		self.fitted = False
		self.lrOptimized = False
		self.predictedData = False
		self.predictedTestData = False
		if create:
			self.create(data=data, **kwargs)
			self.created = True

		if create and optimize:
			self.optimizeLR(verbose=False)
		if ( self.testPath is not None ) and ( len(self.testPath) > 0 ):
			if os.path.isdir('./'+self.dataPath+self.testPath):
				p = os.path.abspath('./'+self.dataPath+self.testPath)
			elif os.path.isdir(os.path.abspath('./'+self.testPath)):
				p = os.path.abspath('./'+self.testPath)
			else:
				p = None
			self.nTestData = -1 if p is None else len(os.listdir(p))


	def create(self, data=None, precompute=True, **kwargs) -> None:
		if self.created: return

		self.data = data if data is not None else ImageClassifierData.from_paths(self.dataPath, tfms=tfms, test_name=self.testPath)
		self.learner = ConvLearner.pretrained(arch, self.data, precompute=precompute, **kwargs)


	def optimizeLR(self, verbose:bool=False, **kwargs) -> None:
		if self.lrOptimized: return
		elif not self.created: self.create()

		lrf = self.learner.lr_find2(**kwargs)
		self.optimalLR = self.learner.sched.best
		self.lrOptimized = True

		if verbose:
			print("Optimal Learning Rate:", self.optimalLR)
			self.learner.sched.plot_lr()
			self.learner.sched.plot_loss()
			lrf.plot()


	def fit(self, nEpochs:int, learningRate:Union[float, List[float]]=None, verbose:bool=False, **kwargs) -> None:
		if self.fitted: return
		elif not self.created: self.create()

		if learningRate is None or learningRate <= 0.0:
			self.optimizeLR(verbose)
			learningRate = self.optimalLR

		self.learner.fit(learningRate, nEpochs, **kwargs)
		self.fitted = True

		if verbose:
			print("Classes:", self.data.classes)
			self.learner.sched.plot_loss()


	def predict(self, isTest:bool=False, verbose:bool=False):
		if self.predictedData and not isTest: return

		logPredictions = self.learner.predict(is_test=isTest)
		if isTest:
			self.testPredictions = np.argmax(logPredictions, axis=1)  # from log probabilities to 0 or 1
			self.testProbabilities = np.exp(logPredictions[:, 1])     # pr(passenger)
			print("Test Predictions:", self.testPredictions)
			print("Test Probabilities:", self.testProbabilities)
		else:
			self.predictions = np.argmax(logPredictions, axis=1)  # from log probabilities to 0 or 1
			self.probabilities = np.exp(logPredictions[:, 1])     # pr(passenger)

		if verbose:
			if isTest:
				print("Test Predictions:", self.testPredictions)
				print("Test Probabilities:", self.testProbabilities)

			self.printSummary(self.nTestData if isTest else 8, isTest)


	def test(self, testDir:str=None, verbose=False) -> [ List[int], List[float] ]:
		if testDir is None:
			#if self.predictedTestData: return self.testPredictions, self.testProbabilities

			self.predict(True, verbose)
		else:
			self.testPredictions, self.testProbabilities = None, None
			self.predictedTestData = False
			self.nTestData = -1
			newData = self.data.from_paths(path=self.dataPath, tfms=tfms, test_name=testDir)
			self.data = newData
			self.learner.set_data(newData, precompute=True)
			self.predict(True, verbose)

		return self.testPredictions, self.testProbabilities

	def printSummary(self, count:int=4, isTest:bool=False):
		def loadImgID(ds, idx): return np.array(PIL.Image.open(ds.path+ds.fnames[idx]))
		def randByMask(mask): return np.empty(1, dtype=np.int) if isTest else np.random.choice(np.where(mask)[0], count, replace=False)
		def randByCorrect(isCorrect): return np.empty(1, dtype=np.int) if isTest else randByMask( (self.predictions == self.data.val_y) == isCorrect)
		def mostByMask(mask, mult):
			idxs = np.where(mask)[0]
			return np.empty(1, dtype=np.int) if isTest else idxs[ np.argsort(mult * self.probabilities[idxs])[:count] ]
		def mostByCorrect(y, is_correct):
			mult = -1 if (y==1)==is_correct else 1
			return np.empty(1, dtype=np.int) if isTest else mostByMask((( self.predictions == self.data.val_y ) == is_correct) & (self.data.val_y == y), mult)

		def plotValWithTitle(idxs, title):
			imgs = [ loadImgID(self.data.val_ds,x) for x in idxs ]
			titleProbs = [ self.probabilities[x] for x in idxs ]
			print(title)
			return plot(imgs, rows=(count // 4)+1, titles=titleProbs, figsize=(16,8))
		def plotTestWithTitle(idxs, title):
			imgs = [ loadImgID(self.data.test_ds,x) for x in idxs ]
			titleProbs = [ self.testProbabilities[x] for x in idxs ]
			print(title)
			return plot(imgs, rows=(count // 4)+1, titles=titleProbs, figsize=(16,8))
		def plot(imgs, figsize=(12,6), rows=count // 4, titles=None):
			f = plt.figure(figsize=figsize)
			for i in range(len(imgs)):
				sp = f.add_subplot(rows, len(imgs)//rows, i+1)
				sp.axis('Off')
				if titles is not None:
					sp.set_title(titles[i], fontsize=16)
				plt.imshow(imgs[i])

		most_uncertain = np.argsort(np.abs(np.array(self.probabilities) - 0.5))[:count]
		if isTest:
			plotTestWithTitle(np.arange(self.nTestData), "Test Data ( All Classes )")
		else:
			# 1. A few correct labels at random
			plotValWithTitle(randByCorrect(True), "Correctly classified ( Random )")
			# 2. A few incorrect labels at random
			plotValWithTitle(randByCorrect(False), "Incorrectly classified ( Random )")

			plotValWithTitle(most_uncertain, "Most uncertain predictions ( All Classes )")
			plotValWithTitle(mostByCorrect(0, True), "Most correct ( Class 0 )")
			plotValWithTitle(mostByCorrect(1, True), "Most correct ( Class 1 )")
			plotValWithTitle(mostByCorrect(0, False), "Most incorrect ( Class 0 )")
			plotValWithTitle(mostByCorrect(1, False), "Most incorrect ( Class 1 )")


	def save(self, name:str="model") -> None:
		self.learner.save(name)


	def load(self, name:str="model") -> None:
		if self.learner is not None:
			self.learner.load(name)
			self.created = True
			self.fitted = True
		else:
			self.created = False
			self.fitted= False




if __name__ == '__main__':
	from fastai.imports import *
	from fastai.transforms import *
	from fastai.model import *
	from fastai.dataset import *
	from fastai.sgdr import *
	from fastai.plots import *

	from google_images_download import google_images_download as gid

	if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
		print("Either torch.cuda or torch.backends.cudnn is unavailable!", torch.cuda.is_available(), torch.backends.cudnn.enabled)
	else:
		print("Torch and Cuda Available!")


	path = "data/trains/"
	testPath = "../../testImgs/passenger"
	subsets = ["train", "valid", "test"]
	classes = ["passenger", "freight"]
	keywords = ["'passenger train -freight -locomotive -engine'", "'freight train -passenger -locomotive -engine'"]
	limit = 100

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
		arguments = { "format":"jpg", "type":"photo", "limit":limit, "output_directory":path }
		for subset in subsets[0:1]:
			for kw, clazz in zip(keywords, classes):
				arguments["keywords"] = kw
				arguments["image_directory"] = f'{subset}/{clazz}/'
				arguments["prefix"] = clazz
				absoluteImgPaths.append(response.download(arguments))
		return absoluteImgPaths
	def preprocessImgData() -> None:
		nSubsets = len(subsets)
		for i, subset in enumerate(subsets[1:]):
			for clazz in classes:
				files = os.listdir(path=f'./{path}{subsets[0]}/{clazz}/')
				nFiles = len(files)
				n = math.floor(nFiles / nSubsets) * (i+1)
				for file in files[n:(2*n)]:
					os.rename(os.path.abspath(f'./{path}{subsets[0]}/{clazz}/'+file), os.path.abspath(f'./{path}{subset}/{clazz}/'+file))

		for subset in subsets:
			for clazz in classes:
				files = os.listdir(path=f'./{path}{subset}/{clazz}/')
				for i, file in enumerate(files):
					newFile = file.replace(file[0:file.find(".jpg")] , f'{clazz}{i+1}').replace(" ", "_").replace(file[ (file.find(".jpg")+4): ], "")
					#print(file, "|", newFile);
					os.rename(os.path.abspath(f'./{path}{subset}/{clazz}/'+file), os.path.abspath(f'./{path}{subset}/{clazz}/'+newFile))

	# def createCNN(**kwargs) -> [ConvLearner, ImageClassifierData]:
	# 	arch = resnet34
	# 	sz=224
	# 	tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
	# 	data = ImageClassifierData.from_paths(path, tfms=tfms, test_name="../../testImgs/passenger")
	#
	# 	cnn = ConvLearner.pretrained(arch, data, precompute=True, **kwargs)
	# 	return cnn, data
	#
	# def findLR(cnn:ConvLearner, verbose:bool=False, **kwargs) -> float:
	# 	cnn.lr_find2(**kwargs)
	# 	res = cnn.sched.best
	# 	if verbose:
	# 		print("Optimal Learning Rate:", res)
	# 		cnn.sched.plot_lr()
	# 		cnn.sched.plot_loss()
	# 	return res
	#
	# def load_img_id(ds, idx): return np.array(PIL.Image.open(path+ds.fnames[idx]))
	# def rand_by_mask(mask, count=4): return np.random.choice(np.where(mask)[0], count, replace=False)
	# def rand_by_correct(isCorrect, count=4): return rand_by_mask((preds == imgData.val_y)==isCorrect, count)
	# def most_by_mask(mask, mult):
	# 	idxs = np.where(mask)[0]
	# 	return idxs[np.argsort(mult * probs[idxs])[:4]]
	# def most_by_correct(y, is_correct):
	# 	mult = -1 if (y==1)==is_correct else 1
	# 	return most_by_mask(((preds == imgData.val_y)==is_correct) & (imgData.val_y == y), mult)
	#
	#
	# def plots(ims, figsize=(12,6), rows=1, titles=None):
	# 	f = plt.figure(figsize=figsize)
	# 	for i in range(len(ims)):
	# 		sp = f.add_subplot(rows, len(ims)//rows, i+1)
	# 		sp.axis('Off')
	# 		if titles is not None:
	# 			sp.set_title(titles[i], fontsize=16)
	# 		plt.imshow(ims[i])
	# def plot_val_with_title(idxs, title):
	# 	imgs = [load_img_id(imgData.val_ds,x) for x in idxs]
	# 	titleProbs = [probs[x] for x in idxs]
	# 	print(title)
	# 	return plots(imgs, rows=1, titles=titleProbs, figsize=(16,8))
	# def plot_test_with_title(idxs, title):
	# 	imgs = [load_img_id(imgData.test_ds,x) for x in idxs]
	# 	titleProbs = [probs[x] for x in idxs]
	# 	print(title)
	# 	return plots(imgs, rows=1, titles=titleProbs, figsize=(16,8))
	#
	#
	#
	# cleanImgData()
	# paths = getImgData()
	# preprocessImgData()
	#
	#
	# learn, imgData = createCNN()
	# learn.fit(0.22, 6, cycle_len=2, cycle_mult=2)
	# learn.load("model")
	# logPreds = learn.predict()
	# preds = np.argmax(logPreds, axis=1)  # from log probabilities to 0 or 1
	# probs = np.exp(logPreds[:,1])        # pr(passenger)
	# print(imgData.classes)
	#
	#
	# # 1. A few correct labels at random
	# plot_val_with_title(rand_by_correct(True), "Correctly classified")
	# # 2. A few incorrect labels at random
	# plot_val_with_title(rand_by_correct(False), "Incorrectly classified")
	#
	# mostUncertain = np.argsort(np.abs(probs -0.5))[:4]
	# plot_val_with_title(mostUncertain, "Most uncertain predictions")
	# plot_val_with_title(most_by_correct(0, True), "Most correct Freight trains")
	# plot_val_with_title(most_by_correct(1, True), "Most correct Passenger trains")
	# plot_val_with_title(most_by_correct(0, False), "Most incorrect Freight trains")
	# plot_val_with_title(most_by_correct(1, False), "Most incorrect Passenger trains")
	#
	# learn, imgData = createCNN()
	# lr = findLR(learn, False)
	#
	# print(lr)
	# learn.sched.plot_lr()
	# learn.sched.plot_loss()
	#
	# learn.fit(lr, 6, cycle_len=2, cycle_mult=2)
	# learn.sched.plot_lr()
	# logPredsTest = learn.predict(is_test=True)
	# learn.save("model_test")
	# predsTest = np.argmax(logPreds, axis=1)  # from log probabilities to 0 or 1
	# probsTest = np.exp(logPreds[:,1])        # pr(passenger)
	# print(predsTest)
	# print(probsTest)
	# plot_test_with_title(rand_by_mask([True] * probs.shape[0], count=3), "Test Pics")

	learn = CNN(create=True, optimize=True, dataPath=path, data=None)
	learn.fit(6, verbose=False, cycle_len=2, cycle_mult=2)
	learn.predict(isTest=False, verbose=False)
	learn.test(testDir=None, verbose=True)
	learn.save("real_weights")
	print("Saved weights as","real_weights")
	# learn.test(testDir=testPath, verbose=True)

	#cleanImgData()

