import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the face detection model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
font = cv2.FONT_HERSHEY_COMPLEX

# Load the trained model
model = load_model('keras_model.h5')

# Define function to get class name
def get_className(classNo):
    if classNo == 0:
        return "Tony Stark"
    elif classNo == 1:
        return "Batman"

while True:
    success, imgOrignal = cap.read()
    if not success:
        break
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = imgOrignal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction, axis=1)
        probabilityValue = np.amax(prediction)
        
        # Draw rectangle and text
        if classIndex == 0:
            label = str(get_className(classIndex)) + " " + str(round(probabilityValue*100, 2)) + "%"
            color = (0, 255, 0)
        elif classIndex == 1:
            label = str(get_className(classIndex)) + " " + str(round(probabilityValue*100, 2)) + "%"
            color = (0, 255, 0)
        
        cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), color, -1)
        cv2.putText(imgOrignal, label, (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Result", imgOrignal)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
