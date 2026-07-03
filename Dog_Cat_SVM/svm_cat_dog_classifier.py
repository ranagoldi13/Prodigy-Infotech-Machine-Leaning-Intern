# SVM Cat vs Dog Image Classification

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# SETTINGS

DATASET_PATH = "PetImages"

IMAGE_SIZE = 96

MAX_IMAGES_PER_CLASS = 2000

# LOAD DATA

images = []
labels = []

cat_path = os.path.join(DATASET_PATH, "Cat")
dog_path = os.path.join(DATASET_PATH, "Dog")

print("Loading Cat images...")

cat_count = 0

for file in os.listdir(cat_path):

    if cat_count >= MAX_IMAGES_PER_CLASS:
        break

    img_path = os.path.join(cat_path, file)

    img = cv2.imread(img_path)

    if img is None:
        continue


    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    images.append(img.flatten())
    labels.append(0)

    cat_count += 1



print("Loading Dog images...")


dog_count = 0

for file in os.listdir(dog_path):

    if dog_count >= MAX_IMAGES_PER_CLASS:
        break


    img_path = os.path.join(dog_path, file)

    img = cv2.imread(img_path)

    if img is None:
        continue


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE)
    )


    images.append(img.flatten())
    labels.append(1)

    dog_count += 1



# Convert to numpy

X = np.array(images, dtype=np.float32) / 255.0

y = np.array(labels)


# Shuffle data

index = np.random.permutation(len(X))

X = X[index]
y = y[index]


print("\nDataset Loaded")

print("Cats :", cat_count)

print("Dogs :", dog_count)

print("Total:", len(X))


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining:", len(X_train))

print("Testing :", len(X_test))

# TRAIN SVM

print("\nTraining SVM...")


model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)


model.fit(
    X_train,
    y_train
)


print("Training Completed")

# PREDICTION

y_pred = model.predict(X_test)


# EVALUATION

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nAccuracy:",
      round(accuracy*100,2),
      "%")


print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# SHOW PREDICTIONS

num_images = 8


plt.figure(figsize=(16,6))


for i in range(num_images):

    plt.subplot(
        2,
        4,
        i+1
    )


    img = X_test[i].reshape(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )


    plt.imshow(img)


    actual = (
        "Cat"
        if y_test[i]==0
        else "Dog"
    )


    predicted = (
        "Cat"
        if y_pred[i]==0
        else "Dog"
    )


    plt.title(
        f"A:{actual}\nP:{predicted}"
    )


    plt.axis("off")


plt.tight_layout()

plt.show()