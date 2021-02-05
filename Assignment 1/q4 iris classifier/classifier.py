import numpy as np
import matplotlib.pyplot as plt
from numpy.testing._private.utils import tempdir
import pandas as pd
import seaborn as sns
import math

def create_mean_dataset(dataset):
	variety_list=np.unique(dataset['variety'].tolist())
	mean_dataset_list=list()
	for variety in variety_list:
		variety_mean = np.mean(dataset[dataset['variety']==variety])
		variety_mean['variety']=variety
		mean_dataset_list.append(variety_mean)
	return pd.DataFrame(mean_dataset_list)

def classify(test_case,mean_dataset):
	distance=list()
	for i,category in mean_dataset.iterrows():
		distance.append(math.sqrt(((test_case['petal.length']-category['petal.length'])**2) + ((test_case['petal.width']-category['petal.width'])**2)))
	print("Distance vector : ", distance)
	l=mean_dataset['variety'].tolist()
	min=0
	min_dist=distance[0]
	for i in range(1,len(distance)):
		if(distance[i]<min_dist):
			min_dist=distance[i]
			min=i
	return 'Iris '+l[min]

dataset=pd.read_csv("Iris_dataset.csv")

test_case_1=dataset.iloc[0]
test_case_2=dataset.iloc[50]
test_case_3=dataset.iloc[100]

dataset=dataset.drop(dataset.index[0])
dataset=dataset.drop(dataset.index[50])
dataset=dataset.drop(dataset.index[100])

mean_dataset=create_mean_dataset(dataset)

print("Flower 0:")
test_case_1['variety']="classified as " + classify(test_case_1,mean_dataset)
print(test_case_1['variety'])
print("Flower 50:")
test_case_2['variety']="classified as " + classify(test_case_2,mean_dataset)
print(test_case_2['variety'])
print("Flower 100:")
test_case_3['variety']="classified as " + classify(test_case_3,mean_dataset)
print(test_case_3['variety'])

dataset=dataset.append([test_case_1,test_case_2,test_case_3])

sns.scatterplot(data=dataset,x='petal.length',y='petal.width',hue='variety')
plt.show()