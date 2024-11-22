import cv2 as cv
import os
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageOps

# Streamlit Title
st.title("FableFlow")

st.markdown(
    """
    Welcome to **FableFlow**, your interactive storytelling assistant. 
    Explore the power of gestures and visuals to create captivating stories.
    """
)

# Horizontal rule
st.markdown("---")

# Second title for the gestures section
st.header("Gesture-Controlls")

# Display images with descriptions, custom size, and borders
images_and_descriptions = [
    {"path": "./Gestures/Gesture1.png", "description": "Control the position of subjects"},
    {"path": "./Gestures/Gesture2.png", "description": "Freeze the current subject in place"},
    {"path": "./Gestures/Gesture3.png", "description": "Remove the last forzen subject"},
    {"path": "./Gestures/Gesture4.png", "description": "Change the character to the next one"},
    {"path": "./Gestures/Gesture5.png", "description": "Move to next background"},
    {"path": "./Gestures/Gesture6.png", "description": "Move to previous background"}
]

# Create two columns
cols = st.columns(3)

st.markdown("---")

st.markdown("Made by Riya Rai [230] & Kartikeiya Rai [200]")

# Display images and descriptions in two columns
for idx, item in enumerate(images_and_descriptions):
    # Load image
    image = Image.open(item["path"])

    # Resize image to 200x200
    image = image.resize((210, 210))

    # Add a border (e.g., 5px black border)
    border_width = 5
    image_with_border = ImageOps.expand(image, border=border_width, fill="black")

    # Assign to columns alternately
    with cols[idx % 3]:
        # Display the image with its description
        st.image(image_with_border, caption=item["description"], use_container_width=False)

# Sidebar for user controls
st.sidebar.title("Controls")
camera_button = st.sidebar.button("Turn On Camera")
exit_button = st.sidebar.button("Exit")

# Parameters
width, height = 1280, 720
gestureThreshold = int(height * 0.75)
folderPath = "Background"  # Path for background slides
characterPath = "Characters"  # Path for character images

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
delay = 10
buttonPressed = False
counter = 0
imgNumber = 0
characterNumber = 0
showCharacter = False
lockedCharacters = []  # Store locked characters and their positions
lastImgNumber = -1  # Track the last image number to detect changes

# Get list of presentation images and character images
pathImages = sorted([f for f in os.listdir(folderPath) if f.endswith(".jpg") or f.endswith(".png")], key=len)
characterImages = [
    cv.resize(cv.imread(os.path.join(characterPath, f), cv.IMREAD_UNCHANGED), (200, 200), interpolation=cv.INTER_AREA)
    for f in sorted(os.listdir(characterPath), key=len)
]

# Function to overlay transparent character image on an RGBA background
def overlay_character(background, character, position):
    if background is None or character is None:
        return  # Skip if background or character not loaded

    # Ensure the background is RGBA
    if background.shape[2] == 3:
        background = cv.cvtColor(background, cv.COLOR_BGR2BGRA)

    # Resize the character image to 100x100
    character = cv.resize(character, (100, 100), interpolation=cv.INTER_AREA)
    h, w = character.shape[:2]
    x, y = position

    # Ensure overlay stays within background bounds
    x = min(max(0, x), background.shape[1] - 1)
    y = min(max(0, y), background.shape[0] - 1)

    # Calculate the overlay region’s valid width and height
    valid_width = min(w, background.shape[1] - x)
    valid_height = min(h, background.shape[0] - y)

    # Adjust the character image if it exceeds the background bounds
    char_rgb = character[:valid_height, :valid_width, :3]
    alpha_mask = character[:valid_height, :valid_width, 3] / 255.0

    # Blend character with background
    for c in range(3):  # Blend RGB channels
        background[y:y + valid_height, x:x + valid_width, c] = (
            alpha_mask * char_rgb[:, :, c] +
            (1 - alpha_mask) * background[y:y + valid_height, x:x + valid_width, c]
        )

if camera_button:
    # Camera Setup
    cap = cv.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    stframe1 = st.empty()  # Placeholder for background slide
    stframe2 = st.empty()  # Placeholder for camera feed

    while cap.isOpened():
        # Get image frame
        ret, img = cap.read()
        img = cv.flip(img, 1)

        # Load and convert background image to RGBA
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv.imread(pathFullImage)
        if imgCurrent is None:
            st.error(f"Failed to load image: {pathFullImage}")
            break
        imgCurrent = cv.cvtColor(imgCurrent, cv.COLOR_BGR2BGRA)

        # Detect if the background has changed
        if imgNumber != lastImgNumber:
            lockedCharacters = []  # Clear locked characters when background changes
            lastImgNumber = imgNumber

        # Detect hands
        hands, img = detectorHand.findHands(img)
        cv.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

        if hands and not buttonPressed:
            hand = hands[0]
            cx, cy = hand["center"]
            lmList = hand["lmList"]
            fingers = detectorHand.fingersUp(hand)
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
            indexFinger = (xVal, yVal)

            if cy <= gestureThreshold:
                if fingers == [1, 0, 0, 0, 0]:  # Left Slide
                    imgNumber = max(0, imgNumber - 1)
                    buttonPressed = True
                elif fingers == [0, 0, 0, 0, 1]:  # Right Slide
                    imgNumber = min(len(pathImages) - 1, imgNumber + 1)
                    buttonPressed = True
                elif fingers == [0, 1, 0, 0, 1]:  # Switch Character
                    characterNumber = (characterNumber + 1) % len(characterImages)
                    buttonPressed = True
                elif fingers == [0, 1, 1, 1, 0]:  # Lock Character
                    lockedCharacters.append((characterNumber, indexFinger))
                    buttonPressed = True
                elif fingers == [1, 1, 1, 1, 1]:  # Clear Last Locked Character
                    if lockedCharacters:
                        lockedCharacters.pop()
                    buttonPressed = True

            elif fingers == [0, 1, 0, 0, 0]:  # Show Character
                showCharacter = True

        if buttonPressed:
            counter += 1
            if counter > delay:
                counter = 0
                buttonPressed = False

        # Overlay all locked characters on the current background
        for charNum, pos in lockedCharacters:
            overlay_character(imgCurrent, characterImages[charNum], pos)

        # Overlay character if enabled (real-time)
        if showCharacter and characterImages:
            overlay_character(imgCurrent, characterImages[characterNumber], indexFinger)

        # Convert frames to PIL format for Streamlit
        imgCurrent = cv.cvtColor(cv.resize(imgCurrent, (900, 600)), cv.COLOR_BGRA2RGB)
        imgSmall = cv.cvtColor(cv.resize(img, (600, 400)), cv.COLOR_BGR2RGB)

        stframe1.image(imgCurrent, caption="Presentation", use_column_width=True)
        stframe2.image(imgSmall, caption="Camera Feed", use_column_width=True)

        # Break loop if user presses the Exit button
        if exit_button:
            break

    cap.release()
    cv.destroyAllWindows()
