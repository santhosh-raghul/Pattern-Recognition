import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers

class svm:

	def __init__(self,train_data,train_labels):

		self.__X=train_data
		self.__Y=np.array([train_labels,])
		self.__weight=[]
		self.__bias=None
		self.__split_for_plotting()

	def train(self):

		n=self.__X.shape[0]

		H=matrix(np.multiply((self.__Y.T @ self.__Y),(self.__X @ self.__X.T)).astype(np.float))
		f=matrix(np.array([-1]*n).astype(np.float),tc='d')
		A=matrix(-np.eye(n).astype(np.float))
		a=matrix(np.array([0.0]*n).astype(np.float))
		B=matrix(self.__Y.astype(np.float),tc='d')
		b=matrix(0.0)

		solvers.options['show_progress'] = False
		solution = solvers.qp(H,f,A,a,B,b)
		alphas = np.array(solution['x'])

		self.__weight=np.zeros_like(self.__X[0],dtype=float)
		for i,alpha in enumerate(alphas):
				self.__weight+=alpha*self.__Y[0][i]*self.__X[i]

		max_index=np.argmax(alphas)
		self.__bias = self.__Y[0][max_index] - self.__weight.T @ self.__X[max_index]

	def __split_for_plotting(self):

		self.x1=[]
		self.y1=[]
		self.x2=[]
		self.y2=[]
		for i,p in enumerate(self.__X):
			if self.__Y[0][i]==1:
				self.x1.append(p[0])
				self.y1.append(p[1])
			else:
				self.x2.append(p[0])
				self.y2.append(p[1])

	def show_plot(self,title=''):

		# styles
		plt.figure(figsize=(8,8))
		plt.figtext(0.5, 0.9, title, ha="center", fontsize=20)
		plt.axvline(0,color='black',linewidth=.8)
		plt.axhline(0,color='black',linewidth=.8)
		plt.grid(color='grey', linestyle=':', linewidth=.5)

		# plotting data
		plt.scatter(self.x2,self.y2)
		plt.scatter(self.x1,self.y1)

		# plotting decision boundary
		if np.any(self.__weight):
			a,b=self.__weight
			c=self.__bias
			if b==0:
				plt.axvline(-c/a, c='black', label='decision boundary')
			else:
				y_intercept=-c/b
				slope=-a/b
				plt.axline((0,y_intercept), slope=slope, c='black', label='decision boundary')
			plt.legend(loc='best',fontsize=16)
			
		plt.figtext(0.5, 0.01, f'weight : {self.__weight}\nbias : {self.__bias}', ha="center", fontsize=20)
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