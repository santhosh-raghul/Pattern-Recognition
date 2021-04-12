import numpy as np
import scikitplot as skplt
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Matrix, solve, symbols
from sympy.plotting import plot_implicit
from sympy.plotting import plot_parametric

def move_sympyplot_to_axes(p, ax):
	backend = p.backend(p)
	backend.ax = ax
	backend._process_series(backend.parent._series, ax, backend.parent)
	backend.ax.spines['right'].set_color('none')
	backend.ax.spines['bottom'].set_position('zero')
	backend.ax.spines['top'].set_color('none')
	plt.close(backend.fig)

def plot_sympyplot_in_mpl(p, ax):
	backend = p.backend(p)
	backend.ax = ax
	backend._process_series(backend.parent._series, ax, backend.parent)
	# backend.process_series()
	backend.ax.spines['right'].set_color('none')
	backend.ax.spines['bottom'].set_position('zero')
	backend.ax.spines['top'].set_color('none')
	plt.close(backend.fig)

def p_X_wi(X,mean,cov):
	return ( -0.5 * np.dot(np.dot(np.transpose(X-mean),np.linalg.inv(cov)),(X-mean)) - 0.5 * np.log(np.linalg.det(cov)) )

	# a = 1 / ( (2*math.pi) * np.linalg.det(cov)**0.5 )
	# b = -0.5 * ( (X-mean).T @ np.linalg.inv(cov) @ (X-mean) )
	# # print(a)
	# # print(b)
	# return math.log(a) * b

def p_X_w1(X):
	return p_X_wi(X,mean_d1,cov_d1)

def p_X_w2(X):
	return p_X_wi(X,mean_d2,cov_d2)
	
def p_X_w3(X):
	return p_X_wi(X,mean_d3,cov_d3)

# 𝑃(𝜔𝑖|𝑋)=𝑃(𝑋|𝜔𝑖).𝑃(𝜔𝑖)/𝑃(𝑋) - 𝑃(𝑋) can be ignored since it is  constant for all the classes

def p_w1_X(X):
	return p_X_w1(X) * a_priori_w1
	
def p_w2_X(X):
	return p_X_w2(X) * a_priori_w2

def p_w3_X(X):
	return p_X_w3(X) * a_priori_w3

def classify(X):
	probabilities=[p_w1_X(X),p_w2_X(X),p_w3_X(X)]
	i=probabilities.index(max(probabilities))
	return classes[i]

classes=['Setosa','Versicolor','Virginica']
data = pd.read_csv("../Assignment 1/q4 iris classifier/Iris_dataset.csv")
data = data.drop(columns=['sepal.length','sepal.width'])

train_d1 = data.iloc[0:50]
train_d2 = data.iloc[50:100]
train_d3 = data.iloc[100:150]

a_priori_w1,a_priori_w2,a_priori_w3 = 1/3,1/3,1/3

mean_d1 = np.mean(train_d1).values
mean_d2 = np.mean(train_d2).values
mean_d3 = np.mean(train_d3).values

cov_d1 = np.cov(train_d1[['petal.length','petal.width']].values.T)
cov_d2 = np.cov(train_d2[['petal.length','petal.width']].values.T)
cov_d3 = np.cov(train_d3[['petal.length','petal.width']].values.T)

# 𝑃(𝑋|𝜔𝑖)=1/(2𝜋)^2|𝛴𝑖|1/2𝑒𝑥𝑝[−12{(𝑋−μ𝑖)𝑡𝛴𝑖−1(𝑋−μ𝑖)}]

x,y=symbols('x y')
M=np.array([x,y])

values_x=np.arange(-5,10.1,0.1)
values_y=np.arange(-5,10.1,0.1)

boundaries=[p_w1_X(M)-p_w2_X(M),p_w2_X(M)-p_w3_X(M),p_w3_X(M)-p_w1_X(M)]

p=plot_implicit(boundaries[0], (x, -10, 10), (y, -10, 10), show=False, line_color='yellow')
p.extend(plot_implicit(boundaries[1], (x, -10, 10), (y, -10, 10), show=False, line_color='pink'))
p.extend(plot_implicit(boundaries[2], (x, -10, 10), (y, -10, 10), show=False, line_color='blue'))

# equations=[]
# for boundary in boundaries:
	# equations.extend(solve(boundary,(x,y)))
# p=plot_parametric(equations[0],label="decision boundary",line_color='black',show=False)
# for equation in equations:#[1:]:
	# p.extend(plot_parametric(equation,label='decision boundary',line_color='yellow',show=False))

fig,ax=plt.subplots(figsize=(14,8))
plot_sympyplot_in_mpl(p, ax)
# move_sympyplot_to_axes(p,ax)

plt.scatter(x=train_d1[['petal.length']],y=train_d1[['petal.width']],label="class 1")
plt.scatter(x=train_d2[['petal.length']],y=train_d2[['petal.width']],label="class 2")
plt.scatter(x=train_d3[['petal.length']],y=train_d3[['petal.width']],label="class 3")

plt.plot(0, 0, c='yellow', label="decision boundary between Setosa and Versicolor")
plt.plot(0, 0, c='pink', label="decision boundary between Versicolor and Virginica")
plt.plot(0, 0, c='blue', label="decision boundary between Setosa and Virginica")

plt.legend(loc='best')

plt.xlabel("petal length")
plt.ylabel("petal width")

plt.show()