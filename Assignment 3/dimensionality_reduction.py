import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg.linalg import isComplexType

def PCA(X):

	mean_=np.mean(X , axis = 0)
	X_meaned = X - mean_
	 
	cov_mat = np.cov(X_meaned , rowvar = False)
	 
	eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
	
	sorted_index = np.argsort(eigen_values)[::-1]
	sorted_eigenvectors = eigen_vectors[:,sorted_index]
	sorted_eigenvalue = eigen_values[sorted_index].astype(np.float64)
	cumulative_eigenvalue = sorted_eigenvalue.cumsum()
	cumul_on_total = cumulative_eigenvalue / cumulative_eigenvalue[-1]
	d_=0
	while(cumul_on_total[d_]<0.95):
		d_+=1
	d_+=1

	plt.bar(list(range(1,eigen_vectors.shape[0]+1)),sorted_eigenvalue)
	plt.ylabel("eigen values")

	eigenvector_subset = sorted_eigenvectors[:,0:d_]	 
	X_reduced = np.array(np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose())
	 
	return X_reduced,eigenvector_subset,mean_,d_

def LDA(X,labels):

	d = X.shape[1]
	classes=np.unique(labels)
	c=len(classes)
	d_=c-1
	class_dict={}
	for i in range(len(classes)):
		class_dict[classes[i]]=i

	class_wise_data=[np.empty((0,)+X[0].shape,float) for i in classes]
	for i in range(len(X)):
		class_wise_data[class_dict[labels[i]]]=np.append(class_wise_data[class_dict[labels[i]]], np.array([X[i],]),axis=0)

	means=[]
	for i in class_wise_data:
		means.append(np.mean(i,axis=0))

	Sw = np.zeros((d,d))
	for i,data in enumerate(class_wise_data):
		z=data-means[i]
		Sw+=(z.T @ z)
	Sw_inv=np.linalg.inv(Sw)

	overall_mean = np.mean(X,axis=0)
	Sb = np.zeros((d,d))
	for i, data in enumerate(means):
		Ni=len(class_wise_data[i])
		z=np.array([means[i]-overall_mean])	
		Sb+=(Ni * (z.T @ z))

	M = Sw_inv @ Sb
	eigen_values , eigen_vectors = np.linalg.eigh(M)
	sorted_index = np.argsort(eigen_values)[::-1]
	sorted_eigenvectors = eigen_vectors[:,sorted_index]
	sorted_eigenvalue = eigen_values[sorted_index]
	eigenvector_subset = sorted_eigenvectors[:,0:d_]

	plt.bar(list(range(1,eigen_vectors.shape[0]+1)),sorted_eigenvalue)
	plt.ylabel("eigen values")

	Y=X @ eigenvector_subset
	return Y,eigenvector_subset