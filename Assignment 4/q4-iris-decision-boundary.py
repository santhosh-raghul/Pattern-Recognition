import numpy as np, multi_class_perceptron,multi_class_svm

# extract required data from file
data = np.genfromtxt('../Assignment 2/Iris_dataset.csv', delimiter=',',dtype=str)

pl_index = np.where(data[0] == 'petal.length')[0][0]
sw_index = np.where(data[0] == 'sepal.width')[0][0]
variety_index = np.where(data[0] == 'variety')[0][0]

train_data=data[1:,[pl_index,sw_index]].astype(np.float)

train_label=data[1:,[variety_index]]
unique_labels = np.unique(train_label)
d={}
for i,label in enumerate(unique_labels):
	d[label]=i
train_label_encoded=np.array([d[label[0]] for label in train_label])

multi_class_perceptron.demo(train_data,train_label_encoded,plot_title='q4')
multi_class_svm.demo(train_data,train_label_encoded,plot_title='q4')