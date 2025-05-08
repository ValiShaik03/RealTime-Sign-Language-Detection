import tensorflow as tf
import numpy as np
import cv2

class MyClassifier:
    def __init__(self, model_path, labels_path):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the labels from the labels.txt file
        self.labels = open(labels_path).read().splitlines()

    def getPrediction(self, image, draw=True):
        # Preprocess the image
        img = cv2.resize(image, (224, 224))  # Resize to 224x224 (model's expected input size)
        img = img / 255.0  # Normalize the pixel values to the range [0, 1]
        img = np.expand_dims(img, axis=0)  # Expand dimensions to match the model input shape (1, 224, 224, 3)

        # Make the prediction
        prediction = self.model.predict(img)
        
        # Debug: print the raw prediction values (probabilities for each class)
        print("Prediction output (probabilities):", prediction)

        # Get the index of the class with the highest probability
        index = np.argmax(prediction)  # This gives the index of the max probability
        print("Predicted index:", index)

        # Return the prediction probabilities and the predicted class index
        return prediction, index
