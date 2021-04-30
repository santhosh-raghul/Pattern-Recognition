import perceptron, svm
import numpy as np

data = np.array([[2,2],[-1,-3],[-1,2],[0,-1],[1,3],[-1,-2],[1,-2],[-1,-1]])
perc_label = np.array([1,  0, 1,  0, 1,  0,  0, 1])
svm_label  = np.array([1, -1, 1, -1, 1, -1, -1, 1])

svm.demo(data,svm_label,plot_title="q2")

print('learning rate = 0.01\n')
perceptron.demo(data,perc_label,learning_rate=0.01,plot_title="q2 lr1")
print('\nlearning rate = 0.5\n')
perceptron.demo(data,perc_label,learning_rate=0.5,plot_title="q2 lr2")