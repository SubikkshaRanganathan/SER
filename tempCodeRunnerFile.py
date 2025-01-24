
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