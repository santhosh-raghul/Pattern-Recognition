import numpy as np
import matplotlib.pyplot as plt

class multi_Perceptron:

	def __init__(self,train_data,train_labels):

		self.__train_data=np.append(train_data,np.array([[1]]*len(train_data)),axis=1)
		self.__train_labels=train_labels
		self.__classes=np.unique(self.__train_labels)
		self.__c=len(self.__classes)

		# initialise weight vector, set default learning rate
		self.__weight=np.zeros_like(self.__train_data[0])
		self.__learning_rate=0.01
		self.__split_data_points()
		self.__weight=np.zeros((self.__c,self.__train_data.shape[1]),dtype=float)

	def __split_data_points(self):

		self.__data_points=dict([(c,[[],[]]) for c in self.__classes])
		for i,p in enumerate(self.__train_data):
			self.__data_points[self.__train_labels[i]][0].append(p[0])
			self.__data_points[self.__train_labels[i]][1].append(p[1])

		self.__class_wise_train_data=[]
		for i in range(self.__c):
			self.__class_wise_train_data.append(-self.__train_data.copy())

		for i in range(self.__c):
			for j in range(self.__train_data.shape[0]):
				if self.__train_labels[j]==i:
					self.__class_wise_train_data[i][j]=-self.__class_wise_train_data[i][j]

	# to set learning rate
	def set_learning_rate(self,learning_rate):
		self.__learning_rate=learning_rate

	# to do 1 iteration of learning
	def train(self):
		for i,train_data in enumerate(self.__class_wise_train_data):
			# print(train_data)
			gradient_of_Jp=np.zeros(self.__train_data[0].shape,dtype=float)
			for y in train_data:
				if not self.__weight[i] @ y > 0:
					gradient_of_Jp+=y
			# print(f'gradient of Jp = {gradient_of_Jp}')
			# print(f'new weight = old weight + learning rate * gradient of Jp\n           = {self.__weight} + {self.__learning_rate} * {gradient_of_Jp}')
			self.__weight[i]= self.__weight[i] + self.__learning_rate * gradient_of_Jp
			# print(f"           = {self.__weight}\n")

	# to plot the data and the decision boundary
	def show_plot(self,title=''):
		
		# styles
		plt.figure(figsize=(8,8))
		plt.figtext(0.5, 0.9, title, ha="center", fontsize=20)
		plt.axvline(0,color='black',linewidth=.8)
		plt.axhline(0,color='black',linewidth=.8)
		plt.grid(color='grey', linestyle=':', linewidth=.5)

		# plotting data
		for c in self.__data_points:
			plt.scatter(self.__data_points[c][0],self.__data_points[c][1],label=f'class {c}')

		# plotting decision boundary
		def plot_decision_boundary(weight,label,color):
			if np.any(weight[:-1]):
				a,b,c=weight
				if b==0:
					plt.axvline(-c/a, c=color, label=label)
				else:
					y_intercept=-c/b
					slope=-a/b
					plt.axline((0,y_intercept), slope=slope, c=color, label=label)

		colors=['b','g','y']
		for i in range(self.__weight.shape[0]):
			plot_decision_boundary(self.__weight[i],f'db b/w "class {i}" and "not class {i}"',colors[i])

		plt.legend(loc='best',fontsize=16)
		plt.figtext(0.35, 0.04, "weight vectors :", ha="center", fontsize=16)
		plt.figtext(0.6, 0.005, str(self.__weight), ha="center", fontsize=14)
		title=title.replace(' ','_')
		plt.savefig(f'output_images/{title}.png')
		plt.show()

	def get_weight(self):
		return self.__weight.copy()

def demo(data,label,plot_title='',learning_rate=None,weight=None):

	a=multi_Perceptron(data,label)
	if learning_rate is not None:
		a.set_learning_rate(learning_rate)
	if weight is not None:
		a.set_weight(weight)
	if plot_title!='':
		plot_title=plot_title+' '

	a.show_plot(plot_title+"perceptron before training")
	i=1
	prev_weight=a.get_weight()
	while True:
		a.train()
		if i in [100,500,1000]:
			a.show_plot(plot_title+f"perceptron iteration {i}")

		if i==1000:
			break
		prev_weight=a.get_weight()
		i+=1

	# a.show_plot(plot_title+"perceptron before training")
	# for i in range(1,11):
	# 	print(f"perceptron iteration {i}:\n")
	# 	a.train()
	# 	a.show_plot(plot_title+f"perceptron iteration {i}")