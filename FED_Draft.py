import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

data = []
labels = []

dataset_path = 'dataset/train'

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_face_mesh.process(img_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)
                data.append(landmarks)
                labels.append(label)

df = pd.DataFrame(data)
df['label'] = labels

df.to_csv('facial_expression_landmarks.csv', index=False)

print("Data Saved Successfully!")