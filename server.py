from flask import Flask, request, flash, redirect, render_template
import cv2
import FocFace
import face_recognition

import os

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = './temp_photos'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

faces = []

@app.route('/show', methods=['GET', 'POST'])
def show():
    if request.method == 'GET':
        addComma = False
        ret = "{'faces':["
        for face in faces:
            if addComma:
                ret += ","
            addComma = True
            ret += face.toJson()
        ret += "]}"

        return ret

@app.route('/init', methods=['GET', 'POST'])
def init():
    if request.method == 'POST':
        print('Hello in init')
        faces.clear()
        jsonReq = request.get_json()
        facesJson = jsonReq.get("faces")
        for faceJson in facesJson:
            FocFace.readFaceFromJson(faces, faceJson)
        return ""

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']

        originalFilename = file.filename
        originalFilename, file_extension = os.path.splitext(originalFilename)
        filename = file.filename
        fullfilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fullfilename)
        image = cv2.imread(fullfilename)
        os.remove(fullfilename)
        newFaces = []
        FocFace.detect(newFaces, originalFilename, image)

        flash('File successfully uploaded')

        addComma = False
        ret = "{'faces':["
        for face in newFaces:
            if addComma:
                ret += ","
            addComma = True
            ret += face.toJson()
            faces.append(face)
        ret += "]}"

        return ret

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        file = request.files['image']

        filename = file.filename
        fullfilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fullfilename)
        image = cv2.imread(fullfilename)
        os.remove(fullfilename)
        newFaces = []
        FocFace.detect(newFaces, "0", image)

        flash('File successfully uploaded')

        known_face_encodings = []
        for face in faces:
            known_face_encodings.append(face.encoding)

        addComma = False
        for newFace in newFaces:
            ret = "{'similar_faces':["
            # Calculate face distance
            face_distances = face_recognition.face_distance(known_face_encodings, newFace.encoding)
            for faceIdx in range(len(face_distances)):
                face_distance = face_distances[faceIdx]
                if face_distance < 0.5:
                    face = faces[faceIdx]
                    if addComma:
                        ret += ","
                    addComma = True
                    ret += "{'ref':'"+str(face.ref)+"',"
                    ret += "'distance':'" + str(face_distance) + "'}"
        ret += "]}"

    return ret

if __name__ == "__main__":
    app.run(host='0.0.0.0')
