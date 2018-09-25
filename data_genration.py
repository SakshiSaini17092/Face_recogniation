import cv2
import numpy as np
#import os
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []

skip = 0
dataset_path = './Data/'

file_name = input("Enter the name of the person")

while True:
	
	ret, frame = cap.read()
	
	if ret == False:
		continue
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
		
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		
		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
			
	
	keyPressed = cv2.waitKey(1) & 0xFF
	if keyPressed == ord('q'):
		break

	cv2.imshow( 'Video captured' ,frame)
			
			
# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')
			
			
	
cap.release()
cv2.destroyAllWindows()