import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load saved model
model = load_model("hand_gesture_model.keras")

# Gesture names
gestures = [
    "Palm",
    "L",
    "Fist",
    "Fist Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm Moved",
    "C",
    "Down"
]

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Region of interest
    roi = frame[100:300, 100:300]

    # Resize
    img = cv2.resize(roi, (64, 64))

    # Normalize
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    gesture = gestures[np.argmax(prediction)]

    # Draw box
    cv2.rectangle(frame,
                  (100,100),
                  (300,300),
                  (0,255,0),
                  2)

    # Display prediction
    cv2.putText(
        frame,
        gesture,
        (100,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()