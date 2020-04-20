import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('F:\\Videos\\Videos\\FG\\FR\\haarcascade_frontalface_default.xml') #<--- here add your haarcascade.xml path
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor = 1.3,minNeighbors = 5) #neighbor should be more try 1 if you wanted...
    
    return faces,gray_img

def labels_for_training_data(directory): 
	faces = []
	faceId = []

	for path,subdirnames,filenames in os.walk(directory):
		for filename in filenames:
			if filename.startswith("."):
				print("Skipping files with '.' ")
				continue


			id = os.path.basename(path)
			img_path = os.path.join(path,filename)
			print("img_path: ",img_path)
			print("Id: ",id)
			test_img = cv2.imread(img_path)
			if test_img is None:
				print("Image not loaded properly")
				continue
			faces_rect, gray_img, = faceDetection(test_img)
			if len(faces_rect)!=1:
				continue  #Sinec we're assuming only single person images are being fed to classifier
			(x,y,w,h) = faces_rect[0]
			roi_gray = gray_img[y:y+w,x:x+h] #region of interest
			faces.append(roi_gray)
			faceId.append(int(id))
	return faces,faceId

def train_classifier(faces,faceId): 
	face_recognizer = cv2.face.LBPHFaceRecognizer_create() #towardsdatascience
	face_recognizer.train(faces,np.array(faceId))
	return face_recognizer

def draw_rect(test_img,face):
	(x,y,w,h) = face
	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0), thickness=5)

def put_text(test_img,text,x,y):
	cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,5,(255,0,0),6)


