import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def euclidian_distance(A,B):
	euc=0.0
	for i in range(len(A)):
		euc += ( (A[i][0]-B[i][0])**2 ) + ( (A[i][1]-B[i][1])**2 ) + ( (A[i][2]-B[i][2])**2 )
	return math.sqrt(euc)

def image_to_histogram(image):
	img = cv2.imread(image)
	image=image[10:]
	size=img.shape[0]*img.shape[1]

	hist_b = cv2.calcHist([img],[0],None,[256],[0,256])
	hist_g = cv2.calcHist([img],[1],None,[256],[0,256])
	hist_r = cv2.calcHist([img],[2],None,[256],[0,256])

	plot=plt.figure(image)
	plt.plot(hist_b,color='b')
	plt.plot(hist_g,color='g')
	plt.plot(hist_r,color='r')
	plt.title(f'histogram for {image}')

	hist = np.array([hist_r,hist_g,hist_b])/size
	return hist.transpose()[0],plot

h1,p1=image_to_histogram('../images/Reference_image1.jpg')
h2,p2=image_to_histogram('../images/Reference_image2.jpg')
hq,pq=image_to_histogram('../images/Query_image.jpg')

d1=euclidian_distance(h1,hq)
d2=euclidian_distance(h2,hq)
print(f"Distance between query image and reference image 1 = {d1}\nDistance between query image and reference image 2 = {d2}")

print("\nVerification using inbuilt function:")
d1 = np.linalg.norm(h1-hq)
d2 = np.linalg.norm(h2-hq)
print(f"Distance between query image and reference image 1 = {d1}\nDistance between query image and reference image 2 = {d2}")

plt.show()