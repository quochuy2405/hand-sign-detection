import math
import cv2
from helpers.hand_tracking import HandDetector
from helpers.Classification import Classifier
import numpy as np
import traceback

# Relative paths
model_path = "../models/best_model_1106.h5"
labels_path = "../models/labels_7chars.txt"
white_image_path = "../assets/white.jpg"  # Updated to relative path

img_size = 400

# Initialize the classifier with relative paths
classifier = Classifier(model_path, labels_path)

# Create a blank white image for hand gesture processing
white = np.ones((img_size, img_size), np.uint8) * 255
cv2.imwrite(white_image_path, white)  # Save the white image (optional, can be skipped if not needed)

# OpenCV video capture
capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 26
step = 1
flag = False
suv = 0

labels = ['chung toi', 'cong bang', 'den truong', 'di','doi xu','duoc','giao tiep',
          'hoa nhap','hoc','moi nguoi', 'muon', 'toi','xung quanh']

bfh = 0
dicttt = dict()
count = 0
kok = []

while True:
    try:
        # Capture video frame
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        hands, frame = hd.findHands(frame, draw=False, flipType=True)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            white = cv2.imread(white_image_path)  # Use the relative path

            # Find landmarks for the hand
            handz, image = hd2.findHands(image, draw=True, flipType=True)
            
            if handz:
                hand = handz[0]
                pts = hand['lmList']

                # Adjust the image for skeleton drawing
                os = ((img_size - w) // 2) - 15
                os1 = ((img_size - h) // 2) - 15
                
                # Draw skeleton on the white image
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

                # Predict the gesture using the classifier
                prediction, index = classifier.getPrediction(white, draw=False)
                
                if prediction[index] > 0.9:  # Only show if confidence > 90%
                    label = labels[index]
                else:
                    label = "ko nhan ra"  # "Not recognized"

                # Display the label on the frame
                frame = cv2.putText(frame, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("frame", frame)

        # Check for ESC key to exit
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:  # esc key
            break

    except Exception:
        print("==", traceback.format_exc())

# Release resources
capture.release()
cv2.destroyAllWindows()
