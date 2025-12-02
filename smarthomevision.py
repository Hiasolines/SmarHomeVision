import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import requests

# Definition
res_x = 1280
res_y = 720
labels = ["paper", "rock", "scissors"]
training_folder = 'training_images_rps'

url_cam = "http://admin:12345@10.100.91.200/image/jpeg.cgi"
url_shelly = "http://10.100.91.14:8080/rest/items/ShellyLight_Betrieb"

display_image = False


def get_picture_from_url(url = url_cam):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    # PIL returns RGB, keep it as RGB for now
    return np.array(image)

def normalize_landmarks(landmarks_array):
    """Normalize landmarks to be scale and translation invariant"""
    # Reshape to get x, y, z separately
    landmarks = landmarks_array.reshape(-1, 3)
    
    # Get wrist (landmark 0) as reference point
    wrist = landmarks[0].copy()
    
    # Translate so wrist is at origin
    landmarks = landmarks - wrist
    
    # Calculate the scale (max distance from wrist)
    distances = np.linalg.norm(landmarks, axis=1)
    max_distance = np.max(distances)
    
    # Normalize by max distance to make scale-invariant
    if max_distance > 0:
        landmarks = landmarks / max_distance
    
    # Flatten back
    return landmarks.flatten()

def switch_light(predicted_label):
    if predicted_label == "rock":
        control_shelly_light(state="ON")
    elif predicted_label == "paper":
        control_shelly_light(state="OFF")
    elif predicted_label == "scissors":
        control_shelly_light(state="OFF")

def control_shelly_light(state="ON", auth_token="oh.ToggleLight.i994VVCmkKJgzwUanlkAFP1Pi86QpajtliS9OYdETG1vBh1c58DQTnZa0mjXp95MS8KBpYn7Fu5GtYRgbiaQ", url = url_shelly):
    
    headers = {
        "Content-Type": "text/plain"
    }
    
    # Add auth token if provided
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        # Send POST request with the state as data
        response = requests.post(url, data=state, headers=headers)
        
        if response.status_code == 200:
            print(f"✓ Light successfully set to {state}")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
        else:
            print(f"✗ Failed to control light")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
        return response
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error occurred: {e}")
        return None

def process_image_by_mediapipe(image, hands):
    hand_results = hands.process(image)
    
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            landmarks_array.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks_array = np.array(landmarks_array)
        
        # Normalize the landmarks
        landmarks_array = normalize_landmarks(landmarks_array)
        
        return landmarks_array
    else: 
        return None

list_hand_landmarks = []
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5)

model = tf.keras.models.load_model('model.keras')

while True:
    frame = get_picture_from_url(url_cam)  # Returns RGB image
    image_rgb = cv2.flip(frame, 1)  # Flip horizontally, keep RGB
    
    # Make a copy for MediaPipe processing
    image_rgb_processing = image_rgb.copy()
    image_rgb_processing.flags.writeable = False
    hand_results = hands.process(image_rgb_processing)

    landmarks = process_image_by_mediapipe(image_rgb_processing, hands)
    
    if landmarks is None:
        if display_image:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(image_bgr, "No hand detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Hands', image_bgr)
    else:
        prediction = model.predict(np.array([landmarks], dtype=np.float32), verbose=0)
        predicted_label = labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        print(f"Prediction: {predicted_label} with confidence {confidence:.2f}")
        switch_light(predicted_label)
        if display_image:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
            text = f"{predicted_label}: {confidence:.2%}"
            cv2.putText(image_bgr, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Hands', image_bgr)
    
    if cv2.waitKey(10) & 0xFF == ord('q') & display_image:
        cv2.destroyAllWindows()
        break
