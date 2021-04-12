import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import scikitplot as skplt
from dimensionality_reduction import PCA

url = "gender_feature_vectors.csv"
data = pd.read_csv(url)
data = data.drop(columns='Unnamed: 0')
 
x = data.iloc[:,1:]
target = data.iloc[:,0]

train_target = pd.concat([target.iloc[10:399],target.iloc[409:800]])
test_target = pd.concat([target.iloc[0:10],target.iloc[399:409]])
train_data = pd.concat([x.iloc[10:399],x.iloc[409:800]])
test_data = pd.concat([x.iloc[0:10],x.iloc[399:409]])

mat_reduced,eigenvector_subset,mean_,d_ = PCA(train_data)
print(f"d' = {d_}")

principal_df = pd.DataFrame(mat_reduced , columns = [f'PC{i}' for i in range(1,d_+1)])
principal_df = pd.concat([principal_df , pd.DataFrame(list(train_target))] , axis = 1)
principal_df.rename(columns = {0:'Label'}, inplace = True)

model = GaussianNB()
model.fit(principal_df[[f'PC{i}' for i in range(1,d_+1)]],principal_df["Label"])

test_np=np.array(test_data)
test_reduced = np.dot(eigenvector_subset.transpose() , (test_data-mean_).transpose() ).transpose()

predicted = model.predict(test_reduced)

test_df = pd.DataFrame(test_reduced , columns = [f'PC{i}' for i in range(1,d_+1)])
test_df = pd.concat([test_df , pd.DataFrame(list(test_target))] , axis = 1)
test_df.rename(columns = {0:'Label'}, inplace = True)

test_df['predicted'] = predicted
correctness=[]
for i in range(len(test_df)):
	if test_df['Label'][i] == test_df['predicted'][i]:
		correctness.append("correct")
	else:
		correctness.append("wrong")

test_df["correctness"]=correctness
print(test_df)
x=accuracy_score(test_df["Label"], predicted)
print(f"Accuracy ={x*100}%")

skplt.metrics.plot_confusion_matrix(test_df["Label"], predicted)#, normalize=True)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()