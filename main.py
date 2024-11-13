import cv2 as cv
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Parameters
width, height = 1280, 720
gestureThreshold = int(height*0.75)
folderPath = "Background"  # Path for background slides
characterPath = "Characters"  # Path for character images

# Camera Setup
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

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
# Load and resize character images to 100x100 pixels
characterImages = [cv.resize(cv.imread(os.path.join(characterPath, f), cv.IMREAD_UNCHANGED), 
                             (200, 200), interpolation=cv.INTER_AREA) 
                   for f in sorted(os.listdir(characterPath), key=len)]

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
        background[y:y+valid_height, x:x+valid_width, c] = (
            alpha_mask * char_rgb[:, :, c] + 
            (1 - alpha_mask) * background[y:y+valid_height, x:x+valid_width, c]
        )

while True:
    # Get image frame
    ret, img = cap.read()
    img = cv.flip(img, 1)

    # Load and convert background image to RGBA
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv.imread(pathFullImage)
    if imgCurrent is None:
        print(f"Failed to load image: {pathFullImage}")
        continue
    imgCurrent = cv.cvtColor(imgCurrent, cv.COLOR_BGR2BGRA)

    # Detect if the background has changed
    if imgNumber != lastImgNumber:
        # Clear locked characters only when the background changes
        lockedCharacters = []
        lastImgNumber = imgNumber  # Update the last image number to the current one

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
                print("Left")
                buttonPressed = True
                imgNumber = max(0, imgNumber - 1)  # Move to previous image
            elif fingers == [0, 0, 0, 0, 1]:  # Right Slide
                print("Right")
                buttonPressed = True
                imgNumber = min(len(pathImages) - 1, imgNumber + 1)  # Move to next image
            elif fingers == [0, 1, 0, 0, 1]:  # Switch Character
                characterNumber = (characterNumber + 1) % len(characterImages)
                buttonPressed = True
            elif fingers == [0, 1, 1, 1, 0]:  # Lock Character
                print("Character Locked")
                lockedCharacters.append((characterNumber, indexFinger))  # Lock the current character at its position
                buttonPressed = True
            elif fingers == [1, 1, 1, 1, 1]:  # Fist gesture to clear all locked characters
                if lockedCharacters:
                    print("Clear Last Locked Character")
                    lockedCharacters.pop()  # Pop last locked character
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

    # Resize the image to fit the presentation window size (1200x800)
    imgCurrent = cv.resize(imgCurrent, (900, 600))

    # Resize the camera feed to fit the smaller window (400x400)
    imgSmall = cv.resize(img, (600, 400))

    # Display the presentation window
    cv.imshow("Presentation", imgCurrent)

    # Display the camera feed in a smaller window
    cv.imshow("Camera Feed", imgSmall)

    # Exit
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
