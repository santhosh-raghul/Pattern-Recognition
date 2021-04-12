import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from dimensionality_reduction import LDA

url = "gender_feature_vectors.csv"
data = pd.read_csv(url)
data = data.drop(columns='Unnamed: 0')
 
x = data.iloc[:,1:]
target = data.iloc[:,0]

train_data = pd.concat([x.iloc[10:399],x.iloc[409:800]])
test_data = pd.concat([x.iloc[0:10],x.iloc[399:409]])

train_target = pd.concat([target.iloc[10:399],target.iloc[409:800]])
test_target = pd.concat([target.iloc[0:10],target.iloc[399:409]])

reduced,eigenvector_subset = LDA(np.array(train_data),list(train_target))
reduced = pd.DataFrame(reduced,columns=['LD1'])
reduced = pd.concat([reduced , pd.DataFrame(list(train_target))] , axis = 1)
reduced.rename(columns = {0:'Label'}, inplace = True)

model = GaussianNB()
model.fit(np.array(reduced.iloc[:,0]).reshape(-1,1),reduced["Label"])

test_reduced=test_data.dot(eigenvector_subset)
test_reduced.rename(columns = {0:'LD1'}, inplace = True)
predicted= model.predict(np.array(test_reduced).reshape(-1,1)) # 0:Overcast, 2:Mild
test_reduced = pd.concat([test_reduced , test_target] , axis = 1)
test_reduced.rename(columns = {'Unnamed: 1':'Label'}, inplace = True)

test_reduced['predicted'] = predicted
correctness=[]
for i in list(range(0,10))+list(range(399,409)):
	if test_reduced['Label'][i] == test_reduced['predicted'][i]:
		correctness.append("correct")
	else:
		correctness.append("wrong")

test_reduced["correctness"]=correctness
print(test_reduced)
x=accuracy_score(test_reduced["Label"], predicted)
print(f"Accuracy ={x*100}%")

skplt.metrics.plot_confusion_matrix(test_reduced["Label"], predicted)#, normalize=True)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()