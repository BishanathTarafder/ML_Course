import cv2

# Load image
image = cv2.imread("C:/Users/emaduzzaman/PycharmProjects/pythonProject/venv/Scripts/image.jpg")


# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create HOG descriptor
hog = cv2.HOGDescriptor()

# Compute HOG features
features = hog.compute(gray)

# Print the shape of the feature vector
print("Feature vector shape:", features.shape)