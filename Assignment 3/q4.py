import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import scikitplot as skplt
from dimensionality_reduction import PCA

url = "face.csv"
data = pd.read_csv(url)

train_data = pd.concat([data.iloc[i*10+2:(i+1)*10] for i in range(40)])
test_data = pd.concat([data.iloc[i*10:i*10+2] for i in range(40)])
train_data.reset_index(drop=True,inplace=True)
test_data.reset_index(drop=True,inplace=True)

mat_reduced,eigenvector_subset,mean_,d_ = PCA(train_data.iloc[:,:-1])
print(f"d' = {d_}")

principal_df = pd.DataFrame(mat_reduced , columns = [f'PC{i}' for i in range(1,d_+1)])
principal_df = pd.concat([principal_df , train_data['target']] , axis = 1)
principal_df.rename(columns = {0:'Label'}, inplace = True)

model = GaussianNB()
model.fit(principal_df[[f'PC{i}' for i in range(1,d_+1)]],principal_df["target"])

# test_np=np.array(test_data.iloc[:,:-1])
test_reduced = (eigenvector_subset.T @ (test_data.iloc[:,:-1]-mean_).T).T
predicted= model.predict(test_reduced)
test_reduced = pd.concat([test_reduced , test_data['target']] , axis = 1)

test_reduced['predicted'] = predicted
correctness=[]
for i in range(len(test_reduced)):
	if test_reduced['target'][i] == test_reduced['predicted'][i]:
		correctness.append("correct")
	else:
		correctness.append("wrong")

test_reduced["correctness"]=correctness
print(test_reduced)
x=accuracy_score(test_reduced["target"], predicted)
print(f"Accuracy ={x*100}%")

skplt.metrics.plot_confusion_matrix(test_reduced["target"], predicted,figsize=(15,15))#, normalize=True)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()