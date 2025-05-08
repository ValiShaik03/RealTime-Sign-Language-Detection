from myClassifier import MyClassifier
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = MyClassifier("C:\\Users\\mdval\\OneDrive\\Desktop\\MODELL\\keras_model.h5", "C:\\Users\\mdval\\OneDrive\\Desktop\\MODELL\\labels.txt")

offset = 20
imgSize = 300
labels = ["Hello", "ILoveYou", "No", "OK", "Please", "Thank You", "Yes"]

# For FPS calculation
pTime = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Prevent out of frame error
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        aspectRatio = h / w

        # Adjusted resizing block to handle shape mismatch
        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize based on width
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Paste resized image into white canvas
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize based on height
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Paste resized image into white canvas

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw label
        cv2.rectangle(imgOutput, (x1, y1 - 70), (x1 + 400, y1 + 40), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-6)
    pTime = cTime

    # Display FPS
    cv2.putText(imgOutput, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    cv2.imshow('Image', imgOutput)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
