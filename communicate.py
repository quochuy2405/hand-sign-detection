import cv2
import numpy as np
import traceback
import streamlit as st
from helpers.classification import  Classifier
from helpers.hand_tracking import  HandDetector
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import time
import os

# Initialize session state if not already present
if 'running' not in st.session_state:
    st.session_state.running = False

# Setting up paths using relative paths instead of absolute ones
MODEL_PATH = "./models/keras_model_13chars.h5"
LABELS_PATH = "./models/labels_13chars.txt"
WHITE_IMG_PATH = "./assets/white.jpg"  # Adjust path if you need to store image in a different folder

# Create white background image if it doesn't exist
if not os.path.exists(WHITE_IMG_PATH):
    img_size = 400
    white = np.ones((img_size, img_size), np.uint8) * 255
    cv2.imwrite(WHITE_IMG_PATH, white)

# Load the classifier
classifier = Classifier(MODEL_PATH, LABELS_PATH)

# Initialize HandDetector and HandTracking
detector = HandDetector(maxHands=1)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Labels for 13 characters model
labels = ['chúng tôi', 'công bằng', 'đến trường', 'đi', 'đối xử', 'được', 'giao tiếp',
          'hòa nhập với', 'học', 'mọi người', 'muốn', 'tôi', 'xung quanh']

# Streamlit app title
st.title("Hand Sign Detection")

class VideoProcessor:
    def __init__(self):
        self.offset = 26
        self.img_size = 400
        self.current_label = ""

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        try:
            frm = cv2.flip(frm, 1)  # Flip the frame horizontally
            hands, frm = hd.findHands(frm, draw=False, flipType=True)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = frm[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                # Read the white image dynamically (relative path)
                white = cv2.imread(WHITE_IMG_PATH)

                handz, image = hd2.findHands(image, draw=True, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']

                    os = ((self.img_size - w) // 2) - 15
                    os1 = ((self.img_size - h) // 2) - 15

                    # Draw lines and points on the white image (to match the model input)
                    for t in range(0, 4):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)

                    cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                    # Draw circles on landmarks
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                    # Get the prediction
                    prediction, index = classifier.getPrediction(white, draw=False)
                    if prediction[index] > 0.9:
                        self.current_label = labels[index]
                    else:
                        self.current_label = "không nhận ra"

            return av.VideoFrame.from_ndarray(frm, format='bgr24')
        except Exception as e:
            st.error(f"An error occurred: {traceback.format_exc()}")
            return av.VideoFrame.from_ndarray(frm, format='bgr24')

# Function to retrieve the current label
def fun_label(video_processor):
    return video_processor.current_label


# Create two columns for video and result display
col1, col2 = st.columns([2, 1])

# Left column for the video stream
with col1:
    webrtc_ctx = webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                    rtc_configuration=RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                    ))

# Right column for result display
with col2:
    result_container = st.empty()

# While the WebRTC context is active
while webrtc_ctx.video_processor:
    current_label = fun_label(webrtc_ctx.video_processor)
    
    # Display current label
    result_container.markdown("<div style='font-size:30px;'>Predict:</div>"
                              f"<div style='font-size:25px;'>{current_label}</div>", unsafe_allow_html=True)

    time.sleep(1)

# Help button and navigation
huongdan_button = st.button("Trợ giúp")
if huongdan_button:
    st.query_params["page"] = "huongdan_button"

query_params = st.query_params
if "page" in query_params:
    page = query_params["page"]
    if page == "huongdan_button":
        with open("huong_dan_dung.py", encoding="utf-8") as f:
            exec(f.read())
