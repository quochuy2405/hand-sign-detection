import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import traceback
import streamlit as st
import os

# Initialize session state if not already present
if 'running' not in st.session_state:
    st.session_state.running = False

img_size = 400

# Set relative path for your model and labels
model_path = "../models/best_model_1106.h5"
labels_path = "../models/labels_7chars.txt"
white_image_path = "../assets/white.jpg"  # Updated to relative path

classifier = Classifier(model_path, labels_path)
white = np.ones((img_size, img_size), np.uint8) * 255
cv2.imwrite(white_image_path, white)

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Streamlit app title
st.title("Hand Sign Detection")

# Placeholder for the video frame
frame_placeholder = st.empty()
offset = 26
step = 1
flag = False
suv = 0

labels = ['chung toi', 'cong bang', 'den truong', 'di', 'doi xu', 'duoc', 'giao tiep',
          'hoa nhap', 'hoc', 'moi nguoi', 'muon', 'toi', 'xung quanh']

# Create a Start/Stop button
if st.button("Start/Stop"):
    st.session_state.running = not st.session_state.running

while st.session_state.running:
    try:
        _, frame = capture.read()
        if frame is None:
            st.error("Failed to capture video frame.")
            break

        frame = cv2.flip(frame, 1)
        hands, frame = hd.findHands(frame, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            white = cv2.imread(white_image_path)  # Use relative path for white.jpg
            handz, image = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']

                os = ((img_size - w) // 2) - 15
                os1 = ((img_size - h) // 2) - 15
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                prediction, index = classifier.getPrediction(white, draw=False)

                if prediction[index] > 0.9:
                    label = labels[index]
                else:
                    label = "ko nhan ra"

                frame = cv2.putText(frame, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    except Exception:
        st.error("An error occurred: " + traceback.format_exc())
        break

capture.release()
cv2.destroyAllWindows()

st.stop()
