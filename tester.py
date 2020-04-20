import cv2
import os
import numpy as np
import faceRecognition as fr

test_img = cv2.imread('F:\\Videos\\Videos\\FG\\known_faces\\Sentdex\\vishalms.jpg')#<--- here add your Test image  path
# cv2.imread('F:\\Videos\\Videos\\FG\\known_faces\\Sentdex\\VishalYO.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces detected: ",faces_detected)

# for (x,y,w,h) in faces_detected:
# 	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0), thickness=5)

# resized_img = cv2.resize(test_img,(1000,700))
# cv2.imshow("face detection ", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows() #only for detection

# faces,faceId = fr.labels_for_training_data('F:/Videos/Videos/FG/FR/Train') 
# face_recognizer = fr.train_classifier(faces,faceId)
# face_recognizer.save('trained_faces_Data.yml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('F:\\Videos\\Videos\\FG\\FR\\trained_faces_Data.yml')#<--- here add your yml path

name = {0:"Emma",1:"Vishal"}

for face in faces_detected:
	(x,y,w,h) = face
	roi_gray = gray_img[y:y+h,x:x+h]
	label, confidence = face_recognizer.predict(roi_gray)
	#confidence should be 0 for 100%
	print("Confidence: ",confidence)
	print("Label: ",label)
	fr.draw_rect(test_img,face)
	predict_name = name[label]
	if(confidence>37):
		continue
	fr.put_text(test_img,predict_name,x,y)

resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow("face Recognition ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows 
