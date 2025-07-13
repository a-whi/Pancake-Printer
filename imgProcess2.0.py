"""
Created by: Alex W
Created: 27/06/25

Document Purpose: A, hopefully, better version of the image processign code

Input: Image

Output: 

Process:
1. Remvoe background using rembg, it'll output a png of the object
After rembg:
2. Convert image to grayscale
3. Threshold or blur → use cv2.Canny() for initial edge detection
4. Use cv2.findContours() to find the biggest contour (face outline)
5. Use cv2.approxPolyDP() or smoothing filters to clean it
6. Fill it to a mask (for ignoring everything outside)

✅ Step 3: Internal Features (Eyes, Mouth)
Use something like:

Mediapipe Face Mesh (Google)
Gives 468 facial landmarks, easy to draw outlines of eyes, mouth, nose.

Or use cv2.findContours() again after masking the face

✅ Step 4: Smooth & Fill
Use morphological ops: cv2.morphologyEx() with cv2.MORPH_CLOSE

Optional: Simplify contours with approxPolyDP()

Convert everything to lines for G-code using marching squares or contours
"""
from rembg import remove
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp

# Image file name, located in 'images' folder.
image_name = 'square.jpg'
input_path = f'./images/{image_name}'
output_path = f'./processed/edges_{image_name}'


# Remove background
inp = Image.open(input_path)
img_no_bg = remove(inp)

# Convert to OpenCV format (RGBA to BGRA to BGR)
img_no_bg_cv = cv2.cvtColor(np.array(img_no_bg), cv2.COLOR_RGBA2BGR)

img_no_bg_cv = cv2.flip(img_no_bg_cv, 1) # Flip horizontally

# Convert to greyscale
gray = cv2.cvtColor(img_no_bg_cv, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# ---- Step 4: Find contours ----
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ---- Step 5: Get largest contour and smooth it ----
if contours:
    biggest_contour = max(contours, key=cv2.contourArea)

    # Optional: Simplify the contour
    epsilon = 0.01 * cv2.arcLength(biggest_contour, True)
    approx = cv2.approxPolyDP(biggest_contour, epsilon, True)

    # ---- Step 6: Create mask from the filled contour ----
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [approx], -1, 255, thickness=cv2.FILLED)

    # Optional: Apply the mask to keep only the area inside
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # Save results
    os.makedirs('./processed', exist_ok=True)
    cv2.imwrite(output_path, masked)
    print(f"Saved masked face outline to {output_path}")

else:
    print("No contours found.")