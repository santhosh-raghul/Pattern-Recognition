import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from dimensionality_reduction import LDA

url = "face.csv"
data = pd.read_csv(url)

x = data.iloc[:,:-1]
target = data.iloc[:,-1]

train_data = pd.concat([data.iloc[i*10+2:(i+1)*10] for i in range(40)])
test_data = pd.concat([data.iloc[i*10:i*10+2] for i in range(40)])
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

reduced,eigenvector_subset = LDA(np.array(train_data.iloc[:,:-1]),list(train_data['target']))
reduced = pd.DataFrame(reduced)

model = GaussianNB()
model.fit(reduced,train_data["target"])

test_reduced=(test_data.iloc[:,:-1]).dot(eigenvector_subset)
predicted= model.predict(test_reduced) 
test_reduced['target']=test_data['target']

test_reduced['predicted'] = predicted
correctness=[]
for i in test_reduced.index:
	if test_reduced['target'][i] == test_reduced['predicted'][i]:
		correctness.append("correct")
	else:
		correctness.append("wrong")

test_reduced["correctness"]=correctness
print(test_reduced)
x=accuracy_score(test_reduced["target"], predicted)
print(f"Accuracy ={x*100}%")

skplt.metrics.plot_confusion_matrix(test_reduced["target"], predicted)#, normalize=True)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()