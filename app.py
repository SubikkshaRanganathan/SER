import cv2
import numpy as np
from collections import deque
from datetime import datetime
from keras.models import load_model


# Load pre-trained emotion detection model
emotion_model = load_model("cnn_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_history = deque(maxlen=5)  # Store the last 5 predictions for smoothing



# Function to highlight face only
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Load DNN model for face detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Open video capture (0 for webcam)
video = cv2.VideoCapture(0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        print("No frame captured from video.")
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:
        cv2.imshow("Detecting emotion", frame)
        continue

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

        if face.size == 0:
            print("Empty face region detected, skipping blob creation.")
            continue

        # Emotion Detection
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_resized = np.stack((face_resized,) * 3, axis=-1)  # Convert grayscale to RGB-like format
        face_resized = face_resized.astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        emotionPreds = emotion_model.predict(face_resized)[0]
        emotion_history.append(emotionPreds)  # Add prediction to history

        # Average prediction over the last 5 frames
        avg_emotionPreds = np.mean(emotion_history, axis=0)
        emotion = emotion_labels[avg_emotionPreds.argmax()]

        # Display emotion label
        label = f'{emotion}'
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the resulting frame
    cv2.imshow("Detecting emotion", resultImg)

# Release resources
video.release()
cv2.destroyAllWindows()
