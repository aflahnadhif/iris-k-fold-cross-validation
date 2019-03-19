import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import numpy as np

class DataSet:
	def __init__(self, header_cols, file_path):
		self.header_cols = header_cols
		self.file_path = file_path
		self.dataset = pd.read_csv(self.file_path, header=None, names=header_cols)
		self.data_length = len(self.dataset)

	def __str__(self):
		return "Validation Data:\n" + str(self.val_data) + "\nTraining Data:\n" + str(self.training_data)

	def splitData(self, low_bound, high_bound):
		self.temp_data = np.split(self.dataset, [low_bound, high_bound])
		self.val_data = self.temp_data[1]
		self.training_data = pd.concat([self.temp_data[0], self.temp_data[2]])
		return self.val_data, self.training_data

	def getLength(self):
		return self.data_length

class KFoldCrossValidation:
	category1 = []
	category2 = []
	theta1 = []
	theta2 = []
	theta3 = []
	theta4 = []
	theta5 = []
	theta6 = []
	theta7 = []
	theta8 = []
	bias1 = []
	bias2 = []
	target1 = []
	target2 = []
	sigmoid1 = []
	sigmoid2 = []
	prediction1 = []
	prediction2 = []
	error1 = []
	error2 = []
	dtheta1 = []
	dtheta2 = []
	dtheta3 = []
	dtheta4 = []
	dtheta5 = []
	dtheta6 = []
	dtheta7 = []
	dtheta8 = []
	dbias1 = []
	dbias2 = []
	totalerror1 = []
	totalerror2 = []
	accuracy = []
	firstepoch = 1
	validate = 0

	def __init__(self, dataset, k, learning_rate, epoch):
		self.dataset = dataset
		self.k = k
		self.learning_rate = learning_rate
		self.epoch = epoch

	def theta(self, n):
		if self.firstepoch == 1 and n > 0:
			self.theta1.append(self.theta1[n-1] - self.learning_rate * self.dtheta1[n-1])
			self.theta2.append(self.theta2[n-1] - self.learning_rate * self.dtheta2[n-1])
			self.theta3.append(self.theta3[n-1] - self.learning_rate * self.dtheta3[n-1])
			self.theta4.append(self.theta4[n-1] - self.learning_rate * self.dtheta4[n-1])
			self.theta5.append(self.theta5[n-1] - self.learning_rate * self.dtheta5[n-1])
			self.theta6.append(self.theta6[n-1] - self.learning_rate * self.dtheta6[n-1])
			self.theta7.append(self.theta7[n-1] - self.learning_rate * self.dtheta7[n-1])
			self.theta8.append(self.theta8[n-1] - self.learning_rate * self.dtheta8[n-1])
		elif self.firstepoch == 0 and n > 0:
			self.theta1[n] = self.theta1[n-1] - self.learning_rate * self.dtheta1[n-1]
			self.theta2[n] = self.theta2[n-1] - self.learning_rate * self.dtheta2[n-1]
			self.theta3[n] = self.theta3[n-1] - self.learning_rate * self.dtheta3[n-1]
			self.theta4[n] = self.theta4[n-1] - self.learning_rate * self.dtheta4[n-1]
			self.theta5[n] = self.theta5[n-1] - self.learning_rate * self.dtheta5[n-1]
			self.theta6[n] = self.theta6[n-1] - self.learning_rate * self.dtheta6[n-1]
			self.theta7[n] = self.theta7[n-1] - self.learning_rate * self.dtheta7[n-1]
			self.theta8[n] = self.theta8[n-1] - self.learning_rate * self.dtheta8[n-1]
		elif self.firstepoch == 0 and n < 1:
			self.theta1[n] = self.theta1[119] - self.learning_rate * self.dtheta1[119]
			self.theta2[n] = self.theta2[119] - self.learning_rate * self.dtheta2[119]
			self.theta3[n] = self.theta3[119] - self.learning_rate * self.dtheta3[119]
			self.theta4[n] = self.theta4[119] - self.learning_rate * self.dtheta4[119]
			self.theta5[n] = self.theta5[119] - self.learning_rate * self.dtheta5[119]
			self.theta6[n] = self.theta6[119] - self.learning_rate * self.dtheta6[119]
			self.theta7[n] = self.theta7[119] - self.learning_rate * self.dtheta7[119]
			self.theta8[n] = self.theta8[119] - self.learning_rate * self.dtheta8[119]
		else:
			self.theta1.append(random.uniform(0, 1))
			self.theta2.append(random.uniform(0, 1))
			self.theta3.append(random.uniform(0, 1))
			self.theta4.append(random.uniform(0, 1))
			self.theta5.append(random.uniform(0, 1))
			self.theta6.append(random.uniform(0, 1))
			self.theta7.append(random.uniform(0, 1))
			self.theta8.append(random.uniform(0, 1))

	def bias(self, n):
		if self.firstepoch == 1 and n > 0:
			self.bias1.append(self.bias1[n-1] - self.learning_rate * self.dbias1[n-1])
			self.bias2.append(self.bias2[n-1] - self.learning_rate * self.dbias2[n-1])
		elif self.firstepoch == 0 and n > 0:
			self.bias1[n] = self.bias1[n-1] - self.learning_rate * self.dbias1[n-1]
			self.bias2[n] = self.bias2[n-1] - self.learning_rate * self.dbias2[n-1]
		elif self.firstepoch == 0 and n < 1:
			self.bias1[n] = self.bias1[119] - self.learning_rate * self.dbias1[119]
			self.bias2[n] = self.bias2[119] - self.learning_rate * self.dbias2[119]
		else:
			self.bias1.append(random.uniform(0, 1))
			self.bias2.append(random.uniform(0, 1))

	def target(self, used_data, n):
		if self.firstepoch == 1:
			self.target1.append(used_data['sepal_length'].iloc[n]*self.theta1[n] + used_data['sepal_width'].iloc[n]*self.theta2[n] + used_data['petal_length'].iloc[n]*self.theta3[n] + used_data['petal_width'].iloc[n]*self.theta4[n] + self.bias1[n])
			self.target2.append(used_data['sepal_length'].iloc[n]*self.theta5[n] + used_data['sepal_width'].iloc[n]*self.theta6[n] + used_data['petal_length'].iloc[n]*self.theta7[n] + used_data['petal_width'].iloc[n]*self.theta8[n] + self.bias2[n])
		elif self.validate == 1:
			self.target1[n] = used_data['sepal_length'].iloc[n]*self.theta1[self.dataset.getLength() - len(used_data) - 1] + used_data['sepal_width'].iloc[n]*self.theta2[self.dataset.getLength() - len(used_data) - 1] + used_data['petal_length'].iloc[n]*self.theta3[self.dataset.getLength() - len(used_data) - 1] + used_data['petal_width'].iloc[n]*self.theta4[self.dataset.getLength() - len(used_data) - 1] + self.bias1[self.dataset.getLength() - len(used_data) - 1]
			self.target2[n] = used_data['sepal_length'].iloc[n]*self.theta5[self.dataset.getLength() - len(used_data) - 1] + used_data['sepal_width'].iloc[n]*self.theta6[self.dataset.getLength() - len(used_data) - 1] + used_data['petal_length'].iloc[n]*self.theta7[self.dataset.getLength() - len(used_data) - 1] + used_data['petal_width'].iloc[n]*self.theta8[self.dataset.getLength() - len(used_data) - 1] + self.bias2[self.dataset.getLength() - len(used_data) - 1]
		else:
			self.target1[n] = used_data['sepal_length'].iloc[n]*self.theta1[n] + used_data['sepal_width'].iloc[n]*self.theta2[n] + used_data['petal_length'].iloc[n]*self.theta3[n] + used_data['petal_width'].iloc[n]*self.theta4[n] + self.bias1[n]
			self.target2[n] = used_data['sepal_length'].iloc[n]*self.theta5[n] + used_data['sepal_width'].iloc[n]*self.theta6[n] + used_data['petal_length'].iloc[n]*self.theta7[n] + used_data['petal_width'].iloc[n]*self.theta8[n] + self.bias2[n]
	
	def sigmoid(self, n):
		if self.firstepoch == 1:
			self.sigmoid1.append(1/(1 + math.exp(-self.target1[n])))
			self.sigmoid2.append(1/(1 + math.exp(-self.target2[n])))
		else:
			self.sigmoid1[n] = 1/(1 + math.exp(-self.target1[n]))
			self.sigmoid2[n] = 1/(1 + math.exp(-self.target2[n]))

	def prediction(self, n):
		if self.firstepoch == 1:
			if self.sigmoid1[n] < 0.5:
				self.prediction1.append(0)
			else:
				self.prediction1.append(1)
			if self.sigmoid2[n] < 0.5:
				self.prediction2.append(0)
			else:
				self.prediction2.append(1)
		else:
			if self.sigmoid1[n] < 0.5:
				self.prediction1[n] = 0
			else:
				self.prediction1[n] = 1
			if self.sigmoid2[n] < 0.5:
				self.prediction2[n] = 0
			else:
				self.prediction2[n] = 1

	def category(self, used_data, n):
		if self.firstepoch == 1:
			if used_data['category'].iloc[n] == 'Iris-setosa':
				self.category1.append(1)
				self.category2.append(0)
			elif used_data['category'].iloc[n] == 'Iris-versicolor':
				self.category1.append(0)
				self.category2.append(1)
			elif used_data['category'].iloc[n] == 'Iris-virginica':
				self.category1.append(0)
				self.category2.append(0)
		else:
			if used_data['category'].iloc[n] == 'Iris-setosa':
				self.category1[n] = 1
				self.category2[n] = 0
			elif used_data['category'].iloc[n] == 'Iris-versicolor':
				self.category1[n] = 0
				self.category2[n] = 1
			elif used_data['category'].iloc[n] == 'Iris-virginica':
				self.category1[n] = 0
				self.category2[n] = 0

	def error(self, n):
		if self.firstepoch == 1:
			self.error1.append(abs(self.sigmoid1[n]-self.category1[n]) ** 2)
			self.error2.append(abs(self.sigmoid2[n]-self.category2[n]) ** 2)
		else:
			self.error1[n] = abs(self.sigmoid1[n]-self.category1[n]) ** 2
			self.error2[n] = abs(self.sigmoid2[n]-self.category2[n]) ** 2

	def totalError(self, n):
		self.totalerror1.append(self.error1[n])
		self.totalerror2.append(self.error2[n])

	def totalAccuracy(self, n):
		if (self.category1[n] == self.prediction1[n]) and (self.category2[n] == self.prediction2[n]):
			self.accuracy.append(1)
		else:
			self.accuracy.append(0)	

	def averageTotalError(self, x):
		self.totalerror1 = self.totalerror1/x
		self.totalerror2 = self.totalerror2/x

	def averageTotalAccuracy(self, x):
		self.accuracy = self.accuracy/x

	def dTheta(self, used_data, n):
		if self.firstepoch == 1:
			self.dtheta1.append(2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['sepal_length'].iloc[n])
			self.dtheta2.append(2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['sepal_width'].iloc[n])
			self.dtheta3.append(2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['petal_length'].iloc[n])
			self.dtheta4.append(2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['petal_width'].iloc[n])
			self.dtheta5.append(2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['sepal_length'].iloc[n])
			self.dtheta6.append(2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['sepal_width'].iloc[n])
			self.dtheta7.append(2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['petal_length'].iloc[n])
			self.dtheta8.append(2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['petal_width'].iloc[n])
		else:
			self.dtheta1[n] = 2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['sepal_length'].iloc[n]
			self.dtheta2[n] = 2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['sepal_width'].iloc[n]
			self.dtheta3[n] = 2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['petal_length'].iloc[n]
			self.dtheta4[n] = 2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]*used_data['petal_width'].iloc[n]
			self.dtheta5[n] = 2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['sepal_length'].iloc[n]
			self.dtheta6[n] = 2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['sepal_width'].iloc[n]
			self.dtheta7[n] = 2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['petal_length'].iloc[n]
			self.dtheta8[n] = 2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]*used_data['petal_width'].iloc[n]

	def dBias(self, n):
		if self.firstepoch == 1:
			self.dbias1.append(2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n])
			self.dbias2.append(2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n])
		else:
			self.dbias1[n] = 2*(self.sigmoid1[n]-self.category1[n])*(1-self.sigmoid1[n])*self.sigmoid1[n]
			self.dbias2[n] = 2*(self.sigmoid2[n]-self.category2[n])*(1-self.sigmoid2[n])*self.sigmoid2[n]

	def plot(self, x, text):
		plt.plot(x)
		plt.ylabel(text)
		plt.show()

	def train(self, used_data):
		self.validate = 0
		for k in range(0, self.epoch):
			for n in range(0, len(used_data)):
				self.theta(n)
				self.bias(n)
				self.target(used_data, n)
				self.sigmoid(n)
				self.prediction(n)
				self.category(used_data, n)
				self.error(n)
				self.dTheta(used_data, n)
				self.dBias(n)
			self.firstepoch = 0

	def val(self, used_data):
		self.validate = 1
		for n in range(0, len(used_data)):
			self.target(used_data, n)
			self.sigmoid(n)
			self.prediction(n)
			self.category(used_data, n)
			self.error(n)
			self.totalError(n)
			self.totalAccuracy(n)
			print("ITERATION ", n + 1)
			print("Accuracy", self.accuracy[n])
			print("Total error 1 = ", self.totalerror1[n])
			print("Total error 2 = ", self.totalerror2[n])
			print()
		plt.plot(self.totalerror1)
		plt.plot(self.totalerror2)
		plt.ylabel("ERROR")
		plt.show()
		self.plot(self.accuracy, "accuracy")

	def start(self):
		fold_size = int(self.dataset.getLength()/self.k)
		for n in range(0, self.k):
			val_data, training_data = self.dataset.splitData(fold_size*n, self.dataset.getLength() - fold_size*(self.k- 1 - n))
			self.train(training_data)
			self.val(val_data)

dataset = DataSet(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'category'], "irisdataset.csv")
KFoldCrossValidationObject = KFoldCrossValidation(dataset, 5, 0.1, 100)
KFoldCrossValidationObject.start()