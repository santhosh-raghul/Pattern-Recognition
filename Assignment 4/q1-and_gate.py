import perceptron,svm
import numpy as np

data = np.array([[0,0],[0,1],[1,0],[1,1]])
perc_label = np.array([0, 0, 0,  1])
svm_label  = np.array([1, 1, 1, -1])

perceptron.demo(data,perc_label,learning_rate=0.5,plot_title="q1")

svm.demo(data,svm_label,plot_title="q1")