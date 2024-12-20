import cv2
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Image Loading, Grayscale Conversion, and Histogram
# Load the image
image = cv2.imread('input.jpg')  
if image is None:
    print("Error: Unable to load the image.")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute intensity histogram
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Plot original image, grayscale image, and histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(histogram, color='black')
plt.title("Intensity Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Part 2: Image Transformations
# Resize the image to 50%
resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imwrite('resized_image.jpg', resized_image)

# Rotate the image by 45 degrees
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
cv2.imwrite('rotated_image.jpg', rotated_image)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
cv2.imwrite('blurred_image.jpg', blurred_image)

# Display transformed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title("Blurred Image")
plt.axis('off')
plt.tight_layout()
plt.show()

# Part 3: Object Detection with Haar Cascades
# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Unable to load Haar Cascade XML file.")
    exit()

# Detect faces
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected faces
image_with_faces = image.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(image_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 3)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Image Loading, Grayscale Conversion, and Histogram
# Load the image
image = cv2.imread('input.jpg')  # Replace 'input.jpg' with your image file name
if image is None:
    print("Error: Unable to load the image.")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute intensity histogram
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Plot original image, grayscale image, and histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(histogram, color='black')
plt.title("Intensity Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Part 2: Image Transformations
# Resize the image to 50%
resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imwrite('resized_image.jpg', resized_image)

# Rotate the image by 45 degrees
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
cv2.imwrite('rotated_image.jpg', rotated_image)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
cv2.imwrite('blurred_image.jpg', blurred_image)

# Display transformed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title("Blurred Image")
plt.axis('off')
plt.tight_layout()
plt.show()

# Part 3: Object Detection with Haar Cascades
# Load Haar Cascade for face detection
import cv2

# Load Haar Cascade XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image containing faces
image = cv2.imread('face.jpg')  
if image is None:
    print("Error: Unable to load the image.")
    exit()

# Convert the image to grayscale (required for Haar Cascade detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Save the image with detected faces
cv2.imwrite('faces_detected.jpg', image)
print(f"{len(faces)} faces detected. Saved as 'faces_detected.jpg'")


# Part 4: Combine All Steps
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(histogram, color='black')
plt.title("Intensity Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB))
plt.title("Faces Detected")
plt.axis('off')
plt.tight_layout()
plt.show()
imwrite('faces_detected.jpg', image_with_faces)

# Part 4: Combine All Steps
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(histogram, color='black')
plt.title("Intensity Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB))
plt.title("Faces Detected")
plt.axis('off')
plt.tight_layout()
plt.show()
