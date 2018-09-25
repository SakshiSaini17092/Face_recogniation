
# 1. Read a video stream using opencv
# 2. extract face out of it

import cv2
import numpy as np
import os
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []

skip = 0
dataset_path = './Data/'

label = []

class_id = 0	#Lables for the given file 
names ={}   	#mapping

# Data prepration

def distance(x1 ,x2):
	return np.sqrt( (( x1 -x2 )**2).sum() )
	
def knn( train , test , k = 5 ):
	
	dist = []
	
	for i in range ( train.shape[0] ):
		
		#get the vector and the label
		ix = train[i,: -1]
		iy = train[i, -1]
		
		# computing the distance and get top key
		
		d = distance(test , ix)
		dist.append([d , iy])
	
	#sort based on distance and get top k
	dk = sorted( dist , key = lambda x: x[0])[:k]
	# retrieve only the lables
	lables = np.array(dk)[:,-1]
	
	
	# Get frequencies of each label
	output = np.unique( lables , return_counts = True )
	
	# Find the max frequency
	index = np.argmax(output[1])
	return output[0][index]
	

for fx in os.listdir(dataset_path):
	
	if fx.endswith('.npy'):
		
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)
		
		#create the mapping
		names [class_id] = fx[:-4]
		#create lables for the calss
		target = class_id*np.ones((data_item.shape[0],))
		calss_id  = class_id +1
		label.append(target)

face_dataset = np.concatenate(face_data , axis = 0)
face_lables = np.concatenate( label , axis = 0 ).reshape((-1,1))
train_dataset = np.concatenate((face_dataset,face_lables) , axis = 1)

#print(face_dataset.shape)
#print(face_lables.shape)	
print(train_dataset.shape)

while True:
	
	ret, frame = cap.read()
	
	if ret == False:
		continue
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
		
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	
	for face in faces[-1:]:
		x,y,w,h = face
		
		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		
		#predicted lable
		output  = knn( train_dataset , face_section.flatten() )

		pred_name = names[int(output)]
		cv2.putText( frame , pred_name , ( x, y -10) , cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0 , 0) , 2, cv2.LINE_AA )
		cv2.rectangle(frame , ( x , y) , ( x+w,y+h) , (0,255,255) ,2)
		#cv2.imshow( 'Video captured' ,frame)
		#cv2.imshow('Face section' , face_section)
	
	cv2.imshow("Faces" , frame)
	
	keyPressed = cv2.waitKey(1) & 0xFF
	if keyPressed == ord('q'):
		break		
	
cap.release()
cv2.destroyAllWindows()