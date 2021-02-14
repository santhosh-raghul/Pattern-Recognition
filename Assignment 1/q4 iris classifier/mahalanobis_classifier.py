import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def separate_categories(dataset):
	variety_list=np.unique(dataset['variety'].tolist())
	categories=list()
	for variety in variety_list:
		df=dataset[dataset['variety']==variety]
		categories.append(df)
	return categories

def classify(dataset,test_cases):

	categories=separate_categories(dataset)
	output=[]

	for i in categories:	
		test_cases['mahala_'+i.iloc[0]['variety']] = mahalanobis(x=test_cases[['petal.length','petal.width']], data=i[['petal.length','petal.width']])

	for i,test_case in test_cases.iterrows():
		print(f"Flower {i}:")
		variety_list=np.unique(dataset['variety'].tolist())
		distances=[]
		for variety in variety_list:
			distances.append(test_case['mahala_'+variety])
		test_case['variety']='classified as Iris '+(variety_list[distances.index(min(distances))])
		print(f"Distance vector : {distances}")
		print(test_case['variety'])
		output.append(test_case)
	return pd.DataFrame(output)

dataset=pd.read_csv("Iris_dataset.csv")

test_cases=pd.DataFrame([dataset.iloc[0],dataset.iloc[50],dataset.iloc[100]])

dataset=dataset.drop(dataset.index[0])
dataset=dataset.drop(dataset.index[50])
dataset=dataset.drop(dataset.index[100])

plot1=plt.figure('Dataset without test cases')
sns.scatterplot(data=dataset,x='petal.length',y='petal.width',hue='variety')
plt.title('dataset without test cases')

test_cases=classify(dataset,test_cases)

dataset=dataset.append(test_cases)

plot2=plt.figure('Dataset with classified test cases')
sns.scatterplot(data=dataset,x='petal.length',y='petal.width',hue='variety')
plt.title('dataset with classified test cases')

plt.show()