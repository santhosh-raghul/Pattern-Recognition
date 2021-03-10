import numpy as np
from matplotlib import pyplot as plt
from sympy import Matrix, solve, symbols

def decision_boundary(dataset,a_priori_prob,title):

	mean=[np.mean(dataset[0],axis=0),np.mean(dataset[1],axis=0)]
	cov=[np.cov(dataset[0],rowvar=False),np.cov(dataset[1],rowvar=False)]

	def gi(x,mean,cov,p_wi):
		return ( -0.5 * np.dot(np.dot(np.transpose(x-mean),np.linalg.inv(cov)),(x-mean)) - 0.5 * np.log(np.linalg.det(cov)) + np.log(p_wi) )[0][0]

	x1,x2=symbols('x1 x2')
	M=Matrix([x1,x2])
	values_x=np.arange(np.min(dataset[:,:,0])-6,np.max(dataset[:,:,0])+6.1,0.1)
	values_y=np.arange(np.min(dataset[:,:,1])-6,np.max(dataset[:,:,1])+6.1,0.1)
	g=gi(M,mean[0].reshape(2,1),cov[0],a_priori_prob[0])-gi(M,mean[1].reshape(2,1),cov[1],a_priori_prob[1])
	soln=solve(g,(x1,x2))
	# print(soln)
	dec_bdry_plot=plt.figure(title)
	plt.title(title)
	try:
		plt.plot(values_x,[soln[0][1].subs(x1,i) for i in values_x],label="decision boundary",color='black')
	except:
		plt.plot([soln[0][0].subs(x2,i) for i in values_y],values_y,label="decision boundary",color='black')
	plt.scatter(x=dataset[0][:,0],y=dataset[0][:,1],label="class 1")
	plt.scatter(x=dataset[1][:,0],y=dataset[1][:,1],label="class 2")
	plt.legend(loc="lower left")
	return dec_bdry_plot

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