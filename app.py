# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:51:36 2022

@author: kalya_kl8c3da
"""
from flask import Flask, render_template, Response, request, flash, redirect, url_for
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename

# import pickle
app=Flask(__name__)
camera = cv2.VideoCapture(0)

UPLOAD_FOLDER = 'received_files'
UPLOAD_FOLDER2 = '/student_images/'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

path = 'student_images'
images = []
classNames = []
mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

@app.route('/image_upload', methods=['GET', 'POST'])
def image_upload():
    # app = Flask(__name__)
    app = Flask(__name__, template_folder='templates')
    app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
    # Upload API
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER2'], filename))
            print("saved file successfully")
            # send file name as parameter to downlad
            return redirect('/downloadfile/' + filename)
    return render_template('image_upload.html')






def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            # face_names = []
            # for face_encoding in face_encodings:
            for encode_face, faceloc in zip(face_encodings,face_locations):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(encoded_face_train, encode_face)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(encoded_face_train, encode_face)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    # print(matches)
                    name = classNames[best_match_index].upper().lower()
                    # print(name)
                    y1,x2,y2,x1 = faceloc
                    # since we scaled down by 4 times
                    y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    # cv2.rectangle(frame,(x1,y1),(x2,y2),(169,169,169),1)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255, 255, 255),1)
                    # cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(frame,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,169,0),2)
                    
                    # markAttendance(name)
                else:
                    y1,x2,y2,x1 = faceloc
                    y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                
                    # cv2.rectangle(frame,(x1,y1),(x2,y2),(169,169,169),1)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255, 255, 255),1)
                    cv2.putText(frame,'Unknown', (x1+6,y2-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,169),1)
            
           #     if matches[best_match_index]:
            #         name = encoded_face_train[best_match_index]

            #     face_names.append(name)
            

            # # Display the results
            # for (top, right, bottom, left), name in zip(face_locations, face_names):
            #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            #     top *= 4
            #     right *= 4
            #     bottom *= 4
            #     left *= 4

            #     # Draw a box around the face
            #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            #     # Draw a label with a name below the face
            #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #     font = cv2.FONT_HERSHEY_DUPLEX
            #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/downloadfile/<filename>", methods=['GET'])
def download_file(filename):
    return render_template('download.html', value=filename)

@app.route('/live_feed')
def index():
    return render_template('index.html')
@app.route('/home')
def hello_world():
    return render_template('index2.html')











if __name__=='__main__':
    app.run(debug=True)
