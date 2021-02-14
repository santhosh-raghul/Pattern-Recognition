import cv2,numpy,math
from matplotlib import pyplot as plt
import copy

def normalize(A):
	norm=max(A)-min(A)
	# norm=numpy.linalg.norm(A)
	A/=norm

def chi2_distance(A,B): 
	chi=0.0
	for i in range(len(A)):
		chi=chi+(((A[i]-B[i])**2)/A[i])
	return chi[0]

query_img=cv2.imread('../images/Query_image.jpg',0)   
ref_img_1=cv2.imread('../images/Reference_image1.jpg',0)   
ref_img_2=cv2.imread('../images/Reference_image2.jpg',0) 

query_hist=cv2.calcHist([query_img],[0],None,[256],[0,256])
ref_1_hist=cv2.calcHist([ref_img_1],[0],None,[256],[0,256])
ref_2_hist=cv2.calcHist([ref_img_2],[0],None,[256],[0,256])

normalize(query_hist)
normalize(ref_1_hist)
normalize(ref_2_hist)

plot1=plt.figure('Query_image.jpg')
plt.plot(query_hist,color='black')
plt.title('grayscale histogram for Query_image.jpg')

plot2=plt.figure('Reference_image1.jpg')
plt.plot(ref_1_hist,color='black')
plt.title('grayscale histogram for Reference_image1.jpg')

plot3=plt.figure('Reference_image2.jpg')
plt.plot(ref_2_hist,color='black')
plt.title('grayscale histogram for Reference_image2.jpg')

query_hist_copy=copy.copy(query_hist)
ref_1_hist_copy=copy.copy(ref_1_hist)
ref_2_hist_copy=copy.copy(ref_2_hist)

ref_1_distance=chi2_distance(query_hist,ref_1_hist)
ref_2_distance=chi2_distance(query_hist,ref_2_hist)

print(f"distance between the query image and reference image 1 = {ref_1_distance}")
print(f"distance between the query image and reference image 2 = {ref_2_distance}")

print("\nVerification using inbuilt function:")

# cv2.normalize(query_hist_copy, query_hist_copy, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# cv2.normalize(ref_1_hist_copy, ref_1_hist_copy, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# cv2.normalize(ref_2_hist_copy, ref_2_hist_copy, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

ref_1_distance=cv2.compareHist(query_hist_copy,ref_1_hist_copy,method=1)
ref_2_distance=cv2.compareHist(query_hist_copy,ref_2_hist_copy,method=1)

print(f"distance between the query image and reference image 1 = {ref_1_distance}")
print(f"distance between the query image and reference image 2 = {ref_2_distance}")

plt.show()