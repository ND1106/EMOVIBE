from flask import Flask, render_template, request, redirect, Response
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import datetime
import time
import pandas as pd
import statistics
import threading
app = Flask(__name__)
most_common_pred_global = None
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6
cv2.ocl.setUseOpenCL(False)

ResNet50V2_model_path = "ResNet50V2_Model.h5"
ResNet50V2_model = load_model(ResNet50V2_model_path)

Emotion_Classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
Music_Player = pd.read_csv("data_moods.csv")
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity','id']]

def Recommend_Songs(pred_class):
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        return (Play['id'].tolist())

    if pred_class == 'Happy' or pred_class == 'Sad':
        Play = Music_Player[Music_Player['mood'] == 'Happy']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        return (Play['id'].tolist())

    if pred_class == 'Fear' or pred_class == 'Angry':
        Play = Music_Player[Music_Player['mood'] == 'Calm']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        return (Play['id'].tolist())

    if pred_class == 'Surprise' or pred_class == 'Neutral':
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        return (Play['id'].tolist())

most_common_pred_global = None

def load_and_prep_image(img, img_shape=224):
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(GrayImg, 1.1, 4)

    for x, y, w, h in faces:
        roi_GrayImg = GrayImg[y: y + h, x: x + w]
        roi_Img = img[y: y + h, x: x + w]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        faces = face_cascade.detectMultiScale(roi_Img, 1.1, 4)

        if len(faces) == 0:
            print("No Faces Detected")
        else:
            for (ex, ey, ew, eh) in faces:
                img = roi_Img[ey: ey+eh, ex: ex+ew]

    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (img_shape, img_shape))
    RGBImg = RGBImg / 255.

    return RGBImg

def pred_and_plot(class_names):
    while True:
        success,frame=cap.read()
        if not success:
            break
        else:
            global most_common_pred_global 
            if not cap.isOpened():
                print("Error: Unable to open the camera.")
                return None

            predictions = []

            start_time = time.time()

            while time.time() - start_time < 20:
            
                ret, img = cap.read()

                if not ret:
                    print("Error: Failed to capture a frame.")
                    break

                cropped_img = load_and_prep_image(img)

                if img is None:
                    continue

                
                pred = ResNet50V2_model.predict(np.expand_dims(cropped_img, axis=0))

                pred_index = int(np.argmax(pred, axis=1)[0])

                pred_class = class_names[pred_index]

                predictions.append(pred_class)

                cv2.putText(img, f"Prediction: {pred_class}", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                ret,buffer=cv2.imencode('.jpg',img)
                frame=buffer.tobytes()

                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows() 

            if predictions:
                most_common_pred = statistics.mode(predictions)
                most_common_pred_global = most_common_pred 
                return most_common_pred
            else:
                return None

@app.route('/', methods = ['GET', 'POST'])
def hello_world():
    return render_template('index.html')
@app.route('/camera')
def video():
    return Response(pred_and_plot(Emotion_Classes),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/model')
def modelpage():
    return render_template('index2.html')
@app.route('/playlist', methods = ['GET', 'POST'])
def predict1():
    song_id = []
    song_id = Recommend_Songs(most_common_pred_global)
    return render_template('index1.html', song_id = song_id)
if __name__ == "__main__":
    app.run(debug = True)