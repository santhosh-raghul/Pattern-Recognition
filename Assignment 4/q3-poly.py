import numpy as np, cv2, svm, perceptron

# reading images
images=[]
for i in range(1,15):
	images.append(cv2.imread(f'poly_images/poly{i}.png'))

# for img in images:
# 	cv2.imshow("image",img)
# 	cv2.waitKey(0)

# choosing features:
def greenish_pixels(image):
	X,Y=image.shape[:2]
	value=0
	for x in range(X):
		for y in range(Y):
			b,g,r=image[x,y]
			# if int(g)>int(b)+int(r):
			if g>b and g>r:
				value+=1
	return value/X/Y

def reddish_pixels(image):
	X,Y=image.shape[:2]
	value=0
	for x in range(X):
		for y in range(Y):
			b,g,r=image[x,y]
			# if int(g)>int(b)+int(r):
			if r>b and r>g:
				value+=1
	return value/X/Y

x1=[]
x2=[]

# extracting features:
for image in images:
	x1.append(greenish_pixels(image))
	x2.append(reddish_pixels(image))

# training
X=np.array(list(zip(x1,x2)))
Y_perc=np.array([1,1,1,1,1,1,1, 0, 0, 0, 0, 0, 0, 0])
Y_svm =np.array([1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1])

svm.demo(X,Y_svm,plot_title="q3")

perceptron.demo(X,Y_perc,learning_rate=0.01,plot_title="q3")