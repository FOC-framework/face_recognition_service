import face_recognition
import pickle
import base64

class FocFace:
    ref = 0
    top = 0
    right = 0
    bottom = 0
    left = 0
    encoding = []

    def toJson(self):
        ret = "{"
        ret += "'ref':'"
        ret += str(self.ref)
        ret += "','top':"
        ret += str(self.top)
        ret += ",'bottom':"
        ret += str(self.bottom)
        ret += ",'right':"
        ret += str(self.right)
        ret += ",'left':"
        ret += str(self.left)
        ret += ",'encoding':'"

        encoding_bytes = pickle.dumps(self.encoding, protocol=0)
        encoding_b64bytes = base64.b64encode(encoding_bytes)
        encoding_string = encoding_b64bytes.decode("ascii")
        ret += encoding_string
        # encoding_string = base64.b64encode(encoding_bytes)
        # ret += encoding_string
        ret += "'"
        ret += "}"
        return ret

def readFaceFromJson(faces, faceJson):
    face = FocFace()
    face.ref = faceJson.get("ref")
    face.top = faceJson.get("top")
    face.bottom = faceJson.get("bottom")
    face.left = faceJson.get("left")
    face.right = faceJson.get("right")
    encoding_string = faceJson.get("encoding")
    encoding_b64bytes = encoding_string.encode("ascii")
    encoding_bytes = base64.b64decode(encoding_b64bytes)
    encoding_ndarray = pickle.loads(encoding_bytes)

    face.encoding = encoding_ndarray

    test_encoding_bytes = pickle.dumps(encoding_ndarray, protocol=0)
    test_encoding_b64bytes = base64.b64encode(test_encoding_bytes)
    test_encoding_string = test_encoding_b64bytes.decode("ascii")

    faces.append(face)

def detect(faces, fileName, image):
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        face = FocFace()

        # print location of each face in this image
        face.ref = int(fileName)
        face.top, face.right, face.bottom, face.left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(face.top, face.left,
                                                                                                    face.bottom,
                                                                                                    face.right))

        # access and show each face in the image
        face_image = image[face.top:face.bottom, face.left:face.right]
        # pil_image = Image.fromarray(face_image)

        encodings = face_recognition.face_encodings(face_image)
        if len(encodings) > 0:
            face.encoding = encodings[0]
            # face.encoding = face.encoding.tolist()
            # print("encoding : " + image_encoding)
            faces.append(face)
    return faces