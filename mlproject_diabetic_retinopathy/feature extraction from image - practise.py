import cv2

# List of image paths
image_paths = ["C:/Users/emaduzzaman/PycharmProjects/pythonProject/venv/Scripts/image.jpg",
               "C:/Users/emaduzzaman/PycharmProjects/pythonProject/venv/Scripts/image2.jpg"]

# Create HOG descriptor
hog = cv2.HOGDescriptor()

for path in image_paths:
    # Load image
    image = cv2.imread(path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute HOG features
    features = hog.compute(gray)

    # Print the shape of the feature vector
    print(f"Feature vector shape for {path}: {features.shape}")

