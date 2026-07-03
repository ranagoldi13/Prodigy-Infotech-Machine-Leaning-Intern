import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

to_categorical = tf.keras.utils.to_categorical
Sequential = tf.keras.Sequential

Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

path = "leapGestRecog"

X = []
y = []

img_size = 64

print("Loading dataset...")

for person in os.listdir(path):

    p = os.path.join(path, person)

    if not os.path.isdir(p):
        continue

    for gesture in os.listdir(p):

        g = os.path.join(p, gesture)

        label = int(gesture[:2]) - 1

        cnt = 0

        for img in os.listdir(g):

            if cnt == 500:
                break

            img_path = os.path.join(g, img)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (img_size, img_size))

            X.append(image)
            y.append(label)

            cnt += 1

print("Images loaded :", len(X))

X = np.array(X, dtype="float32")
X = X / 255.0

y = to_categorical(y, 10)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training images :", len(X_train))
print("Testing images :", len(X_test))

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(64,64,3)
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nTraining started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1
)

print("\nTesting model...\n")

loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy :", round(accuracy * 100, 2), "%")

model.save("hand_gesture_model.keras")

print("\nModel saved successfully")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.show()