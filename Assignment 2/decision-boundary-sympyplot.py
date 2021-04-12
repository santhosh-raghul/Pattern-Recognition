import numpy as np
from matplotlib import pyplot as plt
from sympy import Matrix, solve, symbols
from sympy.plotting import plot_parametric

def move_sympyplot_to_axes(p, ax):
	backend = p.backend(p)
	backend.ax = ax
	backend._process_series(backend.parent._series, ax, backend.parent)
	backend.ax.spines['bottom'].set_position('zero')
	backend.ax.spines['right'].set_color('none')
	backend.ax.spines['top'].set_color('none')
	plt.close(backend.fig)

def gi(x,mean,cov,p_wi):
	return ( -0.5 * np.dot(np.dot(np.transpose(x-mean),np.linalg.inv(cov)),(x-mean)) - 0.5 * np.log(np.linalg.det(cov)) + np.log(p_wi) )[0][0]

def decision_boundary(dataset,a_priori_prob,title):

	mean=[np.mean(dataset[0],axis=0),np.mean(dataset[1],axis=0)]
	cov=[np.cov(dataset[0],rowvar=False),np.cov(dataset[1],rowvar=False)]

	x1,x2=symbols('x1 x2')
	M=Matrix([x1,x2])
	g=gi(M,mean[0].reshape(2,1),cov[0],a_priori_prob[0])-gi(M,mean[1].reshape(2,1),cov[1],a_priori_prob[1])
	soln=solve(g,(x1,x2))

	p1=plot_parametric(soln[0],label="decision boundary",line_color='black',show=False)
	fig,ax=plt.subplots(figsize=(14,8))
	ax.set_title(title)
	move_sympyplot_to_axes(p1,ax)

	plt.scatter(x=dataset[0][:,0],y=dataset[0][:,1],label="class 1")
	plt.scatter(x=dataset[1][:,0],y=dataset[1][:,1],label="class 2")
	plt.legend(loc='best')

	return fig

# question 1
dataset_q1 = np.array([	[[1,6],[3,4],[3,8],[5,6]],
						[[3,0],[1,-2],[3,-4],[5,-2]] ])
a_priori_prob_q1=(0.5,0.5)

# question 2
dataset_q2 = np.array([	[[1,-1],[2,-5],[3,-6],[4,-10],[5,-12],[6,-15]],
						[[-1,1],[-2,5],[-3,6],[-4,10],[-5,12],[-6, 15]] ])
a_priori_prob_q2=(0.3,0.7)

# question 3
dataset_q3 = np.array([	[[2,6],[3,4],[3,8],[4,6]],
						[[3,0],[1,-2],[3,-4],[5,-2]] ])
a_priori_prob_q3=(0.5,0.5)

q1=decision_boundary(dataset_q1,a_priori_prob_q1,"Question 1")
q2=decision_boundary(dataset_q2,a_priori_prob_q2,"Question 2")
q3=decision_boundary(dataset_q3,a_priori_prob_q3,"Question 3")

plt.show()