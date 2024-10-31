import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands with multi-hand detection enabled
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Load background and character images
backgrounds = [cv2.imread('bg1.jpg'), cv2.imread('bg2.jpg')]
character = cv2.imread('character.png')

# Function to detect gestures and control background or character
def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    # Gesture 1: Thumbs up
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return 'thumbs_up'
    # Gesture 2: Open hand
    elif (thumb_tip.x < index_tip.x) and (middle_tip.x > index_tip.x):
        return 'open_hand'
    # Gesture 3: Peace sign
    elif (index_tip.y < thumb_tip.y) and (middle_tip.y < thumb_tip.y):
        return 'peace'
    return None

# Main application loop
cap = cv2.VideoCapture(0)
cv2.namedWindow("Gesture Feed")
cv2.namedWindow("Display")

desired_width = 640
desired_height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Track the state of background and character display
background_idx = 0  # Index of current background
show_character = True  # Toggle character display

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Initialize display frame with the current background
    display_frame = backgrounds[background_idx].copy()

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks on the gesture feed
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture
            gesture = detect_gesture(hand_landmarks.landmark)

            # Right hand controls the background
            if hand_idx == 0 and gesture:
                if gesture == 'thumbs_up':
                    background_idx = 0  # Show first background
                elif gesture == 'open_hand':
                    background_idx = 1  # Show second background

            # Left hand controls the character
            elif hand_idx == 1 and gesture:
                if gesture == 'thumbs_up':
                    show_character = True  # Toggle character ON
                elif gesture == 'open_hand':
                    show_character = False  # Toggle character OFF
                elif gesture == 'peace' and show_character:
                    # Move character to left hand position if character is ON
                    x = int(hand_landmarks.landmark[8].x * display_frame.shape[1])  # Index finger x position
                    y = int(hand_landmarks.landmark[8].y * display_frame.shape[0])  # Index finger y position

                    # Resize character image if necessary
                    char_h, char_w = character.shape[:2] 

                    # Ensure character doesn't go out of bounds
                    if y + char_h <= display_frame.shape[0] and x + char_w <= display_frame.shape[1]:
                        # Overlay character directly onto the background
                        display_frame[y:y+char_h, x:x+char_w] = character

    # Show the gesture feed and the display frame with background and character
    cv2.imshow("Gesture Feed", frame)
    cv2.imshow("Display", display_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()