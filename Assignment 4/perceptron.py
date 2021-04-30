import numpy as np
import matplotlib.pyplot as plt

# g(x) = w transpose · x + w0 , w --> weight vector, w0 --> bias
# for any x, g(x)>0 => x is classified as class 0
#            g(x)<0 => x is classified as class 1
#            g(x)=0 => x is on the decision boundary

# d dimensional data is augmented into d+1 dimensions (for ease of calculations) by:
#		appending 1 at the end to all x, labelled y
#		new weight vector = w with w0 appended at the end, labelled as a
# now g(x) = g(y) = a transpose · y

# all y that belongs to class 1 is negated
# this is done so that we will only need to check 1 condition (g(x)>0) instead of 2 for ensuring the correctness of the classification

# the weight vector a is intialised to a random value and training is done to get closer to the actual a
# let a(k) denote a after k iterations of training. a(k+1) is calculated as:
#		a(k+1) = a(k) - learning_rate * gradient_of_Jp of J(a(k))
# this method of going towards the opposite direction of the gradient_of_Jp is called the gradient_of_Jp descent method

# for perceptron, the function J is Jp, (perceptron criterion)
# Jp(a(k)) = sum of (- a(k) transpose · y), for all misclassified y
# gradient_of_Jp of Jp(a(k)) = partial derivative of Jp(a(k)) wrt a(k) = sum of -y, for all misclassified y
# so, a(k+1) = a(k) - learning_rate * sum of -y, for of all misclassified y
# so the final equation is:
# a(k+1) = a(k) + learning_rate * sum of all misclassified y

class Perceptron:

	def __init__(self,train_data,train_labels):

		assert (len(train_data)==len(train_labels)),"length of train_data and train_labels must match"
		self.__raw_data=train_data

		# append 1 to train data to get y
		self.__train_data=np.append(train_data,np.array([[1]]*len(train_data)),axis=1)

		# check if train_labels is valid
		assert all(np.isin(train_labels,[0,1])),"'train_lables' should contain only 0s or 1s"
		self.train_labels=train_labels

		# negate y values from class 1
		for i,c in enumerate(train_labels):
			if c==1:
				self.__train_data[i]=-self.__train_data[i]

		# initialise weight vector, set default learning rate
		self.__weight=np.zeros_like(self.__train_data[0])
		self.__learning_rate=0.01
		self.__split_for_plotting()

	def __split_for_plotting(self):

		self.x1=[]
		self.y1=[]
		self.x2=[]
		self.y2=[]
		for i,p in enumerate(self.__raw_data):
			if self.train_labels[i]==1:
				self.x1.append(p[0])
				self.y1.append(p[1])
			else:
				self.x2.append(p[0])
				self.y2.append(p[1])

	# to manually set wait vector
	def set_weight(self,weight):
		if weight.shape!=self.__weight.shape:
			raise ValueError(f"given weight vector must be of shape 1x(d+1), 1x{self.__weight.shape[0]} here")
		self.__weight=weight

	# to set learning rate
	def set_learning_rate(self,learning_rate):
		self.__learning_rate=learning_rate

	# to do 1 iteration of learning
	def train(self):
		gradient_of_Jp=np.zeros_like(self.__weight)
		for y in self.__train_data:
			if not self.__weight @ y > 0:
				gradient_of_Jp+=y
		print(f'gradient of Jp = {gradient_of_Jp}')
		print(f'new weight = old weight + learning rate * gradient of Jp\n           = {self.__weight} + {self.__learning_rate} * {gradient_of_Jp}')
		self.__weight= self.__weight + self.__learning_rate * gradient_of_Jp
		print(f"           = {self.__weight}\n")

	# to plot the data and the decision boundary
	def show_plot(self,title=''):
		
		# styles
		plt.figure(figsize=(8,8))
		plt.figtext(0.5, 0.9, title, ha="center", fontsize=20)
		plt.axvline(0,color='black',linewidth=.8)
		plt.axhline(0,color='black',linewidth=.8)
		plt.grid(color='grey', linestyle=':', linewidth=.5)

		# plotting data
		plt.scatter(self.x1,self.y1)
		plt.scatter(self.x2,self.y2)

		# plotting decision boundary
		if np.any(self.__weight[:-1]):
			a,b,c=self.__weight
			if b==0:
				plt.axvline(-c/a, c='black', label='decision boundary')
			else:
				y_intercept=-c/b
				slope=-a/b
				plt.axline((0,y_intercept), slope=slope, c='black', label='decision boundary')
			plt.legend(loc='best',fontsize=16)
			

		plt.figtext(0.5, 0.04, "weight vector : "+str(self.__weight), ha="center", fontsize=20)
		title=title.replace(' ','_')
		plt.savefig(f'output_images/{title}.png')
		plt.show()

	def get_weight(self):
		return self.__weight

def demo(data,label,plot_title='',learning_rate=None,weight=None):

	a=Perceptron(data,label)
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
		print(f"perceptron iteration {i}:\n")
		a.train()
		a.show_plot(plot_title+f"perceptron iteration {i}")
		i+=1
		if np.allclose(prev_weight,a.get_weight()):
			break
		prev_weight=a.get_weight()
	
	print("no significant change in weight vector after this iteration. stopping.")