import cv2
import numpy as np
import tensorflow as tf
import os


class Classifier:
    """
    Classifier class that handles image classification using a pre-trained Keras model.
    """

    def __init__(self, modelPath, labelsPath=None):
        """
        Initializes the Classifier with the model and labels.

        :param modelPath: str, path to the Keras model
        :param labelsPath: str, path to the labels file (optional)
        """
        self.model_path = modelPath
        np.set_printoptions(suppress=True)  # Disable scientific notation for clarity

        # Load the Keras model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Create a NumPy array with the right shape to feed into the Keras model
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        self.labels_path = labelsPath

        # If a labels file is provided, read and store the labels
        if self.labels_path:
            try:
                with open(self.labels_path, "r") as label_file:
                    self.list_labels = [line.strip() for line in label_file]
            except FileNotFoundError:
                print(f"Labels file not found: {self.labels_path}")
                self.list_labels = []
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        """
        Classifies the image and optionally draws the result on the image.

        :param img: image to classify
        :param draw: whether to draw the prediction on the image
        :param pos: position where to draw the text
        :param scale: font scale
        :param color: text color
        :return: list of predictions, index of the most likely prediction
        """
        # Resize and normalize the image
        imgS = cv2.resize(img, (224, 224))
        image_array = np.asarray(imgS)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the data array
        self.data[0] = normalized_image_array

        # Run inference
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        # Draw the prediction text on the image if specified
        if draw and self.labels_path:
            label_text = str(self.list_labels[indexVal]) if self.list_labels else "Unknown"
            cv2.putText(img, label_text, pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal


if __name__ == "__main__":
    # Ensure the model and labels are loaded from valid paths
    path = "./models"  # Adjust this to your relative path
    model_path = os.path.join(path, "keras_model.h5")
    labels_path = os.path.join(path, "labels.txt")

    # Initialize Classifier
    maskClassifier = Classifier(model_path, labels_path)

    # Open video capture
    cap = cv2.VideoCapture(0)  # Use the default webcam (index 0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, img = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Get prediction from the classifier
        prediction, indexVal = maskClassifier.getPrediction(img)
        print(f"Prediction: {prediction}, Predicted label: {maskClassifier.list_labels[indexVal]}")

        # Display the image with prediction
        cv2.imshow("Image", img)

        # Exit gracefully on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
