import numpy as np
import scikitplot as skplt
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def p_X_wi(X,mean,cov):
	a = 1 / ( (2*math.pi)**64 * np.linalg.det(cov)**0.5 ) 
	b = -0.5 * ( (X-mean).T @ np.linalg.inv(cov) @ (X-mean) )
	return a * math.exp(b)

def p_X_male(X):
	return p_X_wi(X,mean_d1,cov_d1)

def p_X_female(X):
	return p_X_wi(X,mean_d2,cov_d2)


# 𝑃(𝜔𝑖|𝑋)=𝑃(𝑋|𝜔𝑖).𝑃(𝜔𝑖)/𝑃(𝑋) - 𝑃(𝑋) can be ignored since it is  constant for all the classes

def p_male_X(X):
	return p_X_male(X) * a_priori_male
	
def p_female_X(X):
	return p_X_female(X) * a_priori_female

def classify(X):
	# probabilities=[p_male_X(X),p_female_X(X)]
	# i=probabilities.index(max(probabilities))
	# return classes[i]
	m=p_male_X(X)
	f=p_female_X(X)
	# print(m,f,end='')
	if m>f:
		# print('\tmale')
		return "male"
	# print('\tfemale')
	return "female"

FACTOR=1000

data = pd.read_csv("face feature vectors.csv")
data = data.drop(columns ='Unnamed: 0')

train_d1 = data.iloc[5:399]
train_d2 = data.iloc[404:800]

test_d1 = data.iloc[0:5]
test_d2 = data.iloc[399:404]

a_priori_male,a_priori_female = 394/790,396/790

train_d1_np=train_d1.values[:,1:].astype(np.float64)*FACTOR
train_d2_np=train_d2.values[:,1:].astype(np.float64)*FACTOR

mean_d1=np.mean(train_d1_np,axis=0)
mean_d2=np.mean(train_d2_np,axis=0)

cov_d1 = np.cov(train_d1_np.T)
cov_d2 = np.cov(train_d2_np.T)

# 𝑃(𝑋|𝜔𝑖) = (0/(2𝜋)^64*|𝛴𝑖|^1/2) * 𝑒𝑥𝑝[−12{(𝑋−μ𝑖)𝑡𝛴𝑖−1(𝑋−μ𝑖)}]


output=[]
for test_data in [test_d1,test_d2]:
	output_=[]
	for i,row in test_data.iterrows():
		X=row.values[1:]
		result=classify(X*FACTOR)
		row['classification']=result
		output_.append(row)
	output.append(pd.DataFrame(output_))

print(output[0].head())
print(output[1].head())

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
		actual_variety.append(row['Unnamed: 1'])
		if row['classification']==row['Unnamed: 1']:
			correct+=1
	total_correct+=correct
	class_ = np.unique(test_data['Unnamed: 1'])[0]
	print(f"\t{class_} : {correct/test_cases*100} %")

print(f"\tOverall : {total_correct/total_test_cases*100} %")

skplt.metrics.plot_confusion_matrix(actual_variety, classified_variety)#, normalize=True)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()

x=accuracy_score(actual_variety, classified_variety)