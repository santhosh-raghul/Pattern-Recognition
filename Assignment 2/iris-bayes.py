import numpy as np
import scikitplot as skplt
import pandas as pd
import math
import matplotlib.pyplot as plt

def p_X_wi(X,mean,cov):
	a = 1 / ( (2*math.pi)**2 * np.linalg.det(cov)**0.5 )
	b = -0.5 * ( (X-mean).T @ np.linalg.inv(cov) @ (X-mean) )
	return a * math.exp(b)

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

train_d1 = data.iloc[0:40]
train_d2 = data.iloc[50:90]
train_d3 = data.iloc[100:140]

test_d1 = data.iloc[40:50]
test_d2 = data.iloc[90:100]
test_d3 = data.iloc[140:150]

# print(train_d1,train_d2,train_d3)
# print(test_d1,test_d2,test_d3)

a_priori_w1,a_priori_w2,a_priori_w3 = 1/3,1/3,1/3

mean_d1 = np.mean(train_d1).values
mean_d2 = np.mean(train_d2).values
mean_d3 = np.mean(train_d3).values

cov_d1 = np.cov(train_d1[['sepal.length','sepal.width','petal.length','petal.width']].values.T)
cov_d2 = np.cov(train_d2[['sepal.length','sepal.width','petal.length','petal.width']].values.T)
cov_d3 = np.cov(train_d3[['sepal.length','sepal.width','petal.length','petal.width']].values.T)

# 𝑃(𝑋|𝜔𝑖)=1/(2𝜋)^2|𝛴𝑖|1/2𝑒𝑥𝑝[−12{(𝑋−μ𝑖)𝑡𝛴𝑖−1(𝑋−μ𝑖)}]


output=[]
for test_data in [test_d1,test_d2,test_d3]:
	output_=[]
	for i,row in test_data.iterrows():
		X=row.values[:4]
		result=classify(X)
		row['classification']=result
		output_.append(row)
	output.append(pd.DataFrame(output_))

print(output[0])
print(output[1])
print(output[2])

print("Accuracy :")

total_correct=0
total_test_cases=0
actual_variety=[]
classified_variety=[]
for test_data in output:
	correct=0
	test_cases=len(test_data)
	total_test_cases+=test_cases
	for i,row in test_data.iterrows():
		classified_variety.append(row['classification'])
		actual_variety.append(row['variety'])
		if row['classification']==row['variety']:
			correct+=1
	total_correct+=correct
	class_ = np.unique(test_data['variety'])[0]
	print(f"\t{class_} : {correct/test_cases*100} %")

print(f"\tOverall : {total_correct/total_test_cases*100} %")

skplt.metrics.plot_confusion_matrix(actual_variety, classified_variety, normalize=True)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()