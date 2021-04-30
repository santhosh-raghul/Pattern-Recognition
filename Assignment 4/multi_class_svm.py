import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers

class svm:

	def __init__(self,train_data,train_labels):

		self.__X=train_data
		self.__train_labels=train_labels
		self.__classes=np.unique(self.__train_labels)
		self.__c=len(self.__classes)
		self.__Y_for_each_class=[]
		for i in range(self.__c):
			self.__Y_for_each_class.append(train_labels.copy())
			for j in range(train_labels.shape[0]):
				self.__Y_for_each_class[i][j] = 1.0 if self.__Y_for_each_class[i][j] == i else -1
		self.__Y_for_each_class=np.array(self.__Y_for_each_class)
		self.__split_for_plotting()
		self.__weight=np.array([])
		self.__bias=np.array([])

	def train(self):

		n=self.__X.shape[0]

		def train_for_one_class(Y):

			H=matrix(np.multiply((Y.T @ Y),(self.__X @ self.__X.T)).astype(np.float))
			f=matrix(np.array([-1]*n).astype(np.float),tc='d')
			A=matrix(-np.eye(n).astype(np.float))
			a=matrix(np.array([0.0]*n).astype(np.float))
			B=matrix(Y.astype(np.float),tc='d')
			b=matrix(0.0)

			solvers.options['show_progress'] = False
			solution = solvers.qp(H,f,A,a,B,b)
			alphas = np.array(solution['x'])

			weight=np.zeros_like(self.__X[0],dtype=float)
			for i,alpha in enumerate(alphas):
					weight+=alpha*Y[0][i]*self.__X[i]

			# max_index=np.argmax(alphas)
			# bias = Y[0][max_index] - weight.T @ self.__X[max_index]
			l1=[]
			l2=[]
			for i in range(len(self.__X)):
				if Y[0][i]==1:
					l1.append(weight.T @ self.__X[i])
				else:
					l2.append(weight.T @ self.__X[i])

			bias=0.5 * (np.min(l1)-np.max(l2))

			return weight,bias

		self.__weight=[]
		self.__bias=[]
		for i in range(self.__c):
			weight,bias=train_for_one_class(self.__Y_for_each_class[i].reshape(1,-1))
			self.__weight.append(weight)
			self.__bias.append(bias)

		self.__weight=np.array(self.__weight)
		self.__bias=np.array(self.__bias)

	def __split_for_plotting(self):

		self.__data_points=dict([(c,[[],[]]) for c in self.__classes])
		for i,p in enumerate(self.__X):
			self.__data_points[self.__train_labels[i]][0].append(p[0])
			self.__data_points[self.__train_labels[i]][1].append(p[1])

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
		def plot_decision_boundary(weight,bias,label,color):
			if np.any(weight[:-1]):
				a,b=weight
				c=bias
				if b==0:
					plt.axvline(-c/a, c=color, label=label)
				else:
					y_intercept=-c/b
					slope=-a/b
					plt.axline((0,y_intercept), slope=slope, c=color, label=label)

		colors=['b','g','y']
		for i in range(self.__bias.shape[0]):
			plot_decision_boundary(self.__weight[i],self.__bias[i],f'db b/w "class {i}" and "not class {i}"',colors[i])
	
		plt.figtext(0.5, 0.01, f'weight : {self.__weight}\nbias : {self.__bias}', ha="center", fontsize=20)
		plt.legend(loc='best',fontsize=16)
		title=title.replace(' ','_')
		plt.savefig(f'output_images/{title}.png')
		plt.show()

def demo(data,label,plot_title=''):

	a=svm(data,label)
	if plot_title!='':
		plot_title=plot_title+' '

	a.show_plot(plot_title+"svm before training")
	a.train()
	a.show_plot(plot_title+"svm after training")

	