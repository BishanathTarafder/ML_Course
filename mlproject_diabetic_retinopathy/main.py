
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fundus image
img = cv2.imread('D:/ML using python/Project final/sample/10_left.jpeg')

# Show the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show the grayscale image
plt.imshow(gray, cmap='gray')
plt.title('RGB Image')
plt.show()

# Resize the image to a fixed size (e.g., 256x256)
resized = cv2.resize(gray, (256, 256))

# Show the resized image
plt.imshow(resized, cmap='gray')
plt.title('Resized Image')
plt.show()

# Normalize the pixel intensities to a range of [0, 1]
normalized = resized / 255.0

# Show the normalized image
plt.imshow(normalized, cmap='gray')
plt.title('Normalized Image')
plt.show()

"""


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical


from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import StandardScaler
from skimage import io
from sklearn.metrics import accuracy_score





# Define the directory that contains the fundus images
dir_path = 'D:/ML using python/Project final/new'

# Define the image size
img_size = (224, 224)


# Define a function to preprocess the images
def preprocess_images(dir_path, img_size):
    # Get the list of image file names
    file_names = os.listdir(dir_path)

    # Create an empty NumPy array to store the preprocessed images
    preprocessed_images = np.empty((len(file_names), img_size[0], img_size[1], 3))

    # Loop over the image file names and preprocess each image
    for i, file_name in enumerate(file_names):
        # Load the image
        img = cv2.imread(os.path.join(dir_path, file_name))

        # Convert the image to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image
        resized = cv2.resize(rgb, img_size)

        # Normalize the pixel intensities
        normalized = resized / 223.0

        # Store the preprocessed image in the NumPy array
        preprocessed_images[i] = normalized

    return preprocessed_images

# Preprocess the images
preprocessed_images = preprocess_images(dir_path, img_size)

# Save the preprocessed images to a NumPy binary file
np.save('preprocessed_images.npy', preprocessed_images)

# Load the .npy file
data = np.load('preprocessed_images.npy')



"""
# Show each image in the data array
for i in range(data.shape[0]):
    plt.imshow(data[i], cmap='gray')
    plt.title(f'Image {i+1}')
    plt.show()
"""



# Load the CSV file containing the image labels
df = pd.read_csv('D:/ML using python/Project final/sample_Label.csv')

# Extract the label column from the DataFrame
labels = df['level'].values


"""
# Display the first 5 images and their labels
for i in range(10):
    plt.imshow(data[i], cmap='gray')
    plt.title(f'Label: {labels[i]}')
    plt.show()
"""




# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42)










# Extract features using pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=img_size+(3,))
X_train_features = base_model.predict(X_train)
X_test_features = base_model.predict(X_test)

# Reshape the features for feeding to CNN model
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Normalize the data
X_train_norm = X_train_features / X_train_features.max()
X_test_norm = X_test_features / X_train_features.max()



# Train and evaluate the CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=X_train_norm.shape[1:]))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_cnn_model()
model.fit(X_train_norm, y_train, epochs=10, batch_size=32, validation_data=(X_test_norm, y_test))
cnn_acc = model.evaluate(X_test_norm, y_test)[1]
y_score_cnn = model.predict(X_test_norm)
y_pred_cnn = model.predict(X_test_norm)
y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)
cnn_cm = confusion_matrix(y_test, y_pred_cnn_classes)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_cnn[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_cnn.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for CNN using pre-trained VGG16 model')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("CNN with VGG16 model accuracy:", cnn_acc)
print("CNN confusion matrix:")
print(cnn_cm)
sns.heatmap(cnn_cm, annot=True, cmap='Blues')












# Define the directory that contains the fundus images
dir_path = 'D:/ML using python/Project final/new'

# Define the image size
img_size = (256, 256)


# Define a function to preprocess the images
def preprocess_images(dir_path, img_size):
    # Get the list of image file names
    file_names = os.listdir(dir_path)

    # Create an empty NumPy array to store the preprocessed images
    preprocessed_images = np.empty((len(file_names), img_size[0], img_size[1]))

    # Loop over the image file names and preprocess each image
    for i, file_name in enumerate(file_names):
        # Load the image
        img = cv2.imread(os.path.join(dir_path, file_name))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the image
        resized = cv2.resize(gray, img_size)

        # Normalize the pixel intensities
        normalized = resized / 255.0

        # Store the preprocessed image in the NumPy array
        preprocessed_images[i] = normalized

    return preprocessed_images


# Preprocess the images
preprocessed_images = preprocess_images(dir_path, img_size)

# Save the preprocessed images to a NumPy binary file
np.save('preprocessed_images.npy', preprocessed_images)

# Load the .npy file
data = np.load('preprocessed_images.npy')


"""

# Show each image in the data array
for i in range(data.shape[0]):
    plt.imshow(data[i], cmap='gray')
    plt.title(f'Image {i+1}')
    plt.show()

"""


# Load the CSV file containing the image labels
df = pd.read_csv('D:/ML using python/Project final/sample_Label.csv')

# Extract the label column from the DataFrame
labels = df['level'].values

"""

# Display the first 5 images and their labels
for i in range(10):
    plt.imshow(data[i], cmap='gray')
    plt.title(f'Label: {labels[i]}')
    plt.show()

"""



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42)



# Define a function to normalize pixel values
def normalize_pixels(data):
    return data.astype('float32') / 255.0

# Preprocess the data
X_train_norm = normalize_pixels(X_train)
X_test_norm = normalize_pixels(X_test)




# Train and evaluate the CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=img_size+(1,)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_cnn_model()
model.fit(X_train_norm, y_train, epochs=10, batch_size=32, validation_data=(X_test_norm, y_test))
cnn_acc = model.evaluate(X_test_norm, y_test)[1]
y_score_cnn = model.predict(X_test_norm)
y_pred_cnn_classes = np.argmax(y_score_cnn, axis=1)
cnn_cm = confusion_matrix(y_test, y_pred_cnn_classes)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_cnn[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_cnn.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for CNN')
plt.legend(loc="lower right")
plt.show()

print("CNN accuracy:", cnn_acc)
print("CNN confusion matrix:")
print(cnn_cm)
sns.heatmap(cnn_cm, annot=True, cmap='Blues')


# Extract HOG features from the training data
X_train_hog = []
for image in X_train_norm:
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    X_train_hog.append(hog_features)
X_train_hog = np.array(X_train_hog)

# Extract HOG features from the test data
X_test_hog = []
for image in X_test_norm:
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    X_test_hog.append(hog_features)
X_test_hog = np.array(X_test_hog)

# Train and evaluate the Random Forest model with HOG features
rf_hog = RandomForestClassifier(n_estimators=100, random_state=42)
rf_hog.fit(X_train_hog, y_train)

# Predict on test set and get accuracy and confusion matrix
rf_hog_acc = rf_hog.score(X_test_hog, y_test)
y_pred_rf_hog = rf_hog.predict(X_test_hog)
rf_hog_cm = confusion_matrix(y_test, y_pred_rf_hog)

# Predict probabilities for ROC curve
y_score_rf_hog = rf_hog.predict_proba(X_test_hog)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_one_hot.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_rf_hog[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_rf_hog.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Random Forest with HOG features')
plt.legend(loc="lower right")
plt.show()

# Print accuracy and confusion matrix
print("Random Forest with HOG features accuracy:", rf_hog_acc)
print("Random Forest with HOG features confusion matrix:")
print(rf_hog_cm)
sns.heatmap(rf_hog_cm, annot=True, cmap='Blues')


# Train and evaluate the SVM model with HOG features
svm_hog = SVC(kernel='linear', C=1, random_state=42)
svm_hog.fit(X_train_hog, y_train)
svm_hog_acc = svm_hog.score(X_test_hog, y_test)
y_score_svm_hog = svm_hog.decision_function(X_test_hog)
y_pred_svm_hog = svm_hog.predict(X_test_hog)
svm_hog_cm = confusion_matrix(y_test, y_pred_svm_hog)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_svm_hog[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_svm_hog.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for SVM with HOG features')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("SVM with HOG features accuracy:", svm_hog_acc)
print("SVM with HOG features confusion matrix:")
print(svm_hog_cm)
sns.heatmap(svm_hog_cm, annot=True, cmap='Blues')


# Train and evaluate the KNN model with HOG features
knn_hog = KNeighborsClassifier(n_neighbors=5)
knn_hog.fit(X_train_hog, y_train)
knn_hog_acc = knn_hog.score(X_test_hog, y_test)
y_score_knn_hog = knn_hog.predict_proba(X_test_hog)
y_pred_knn_hog = knn_hog.predict(X_test_hog)
knn_hog_cm = confusion_matrix(y_test, y_pred_knn_hog)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_knn_hog[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_knn_hog.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for KNN with HOG features')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("KNN with HOG features accuracy:", knn_hog_acc)
print("KNN with HOG features confusion matrix:")
print(knn_hog_cm)
sns.heatmap(knn_hog_cm, annot=True, cmap='Blues')






# Define LBP parameters
radius = 1
n_points = 8 * radius
METHOD = 'uniform'

# Extract LBP features from the training data
X_train_lbp = []
for image in X_train_norm:
    image = (image * 255).astype(np.uint8)

    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    X_train_lbp.append(hist)
X_train_lbp = np.array(X_train_lbp)

# Extract LBP features from the test data
X_test_lbp = []
for image in X_test_norm:
    image = (image * 255).astype(np.uint8)

    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    X_test_lbp.append(hist)
X_test_lbp = np.array(X_test_lbp)

# Train and evaluate the Random Forest model with LBP features
rf_lbp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_lbp.fit(X_train_lbp, y_train)
rf_lbp_acc = rf_lbp.score(X_test_lbp, y_test)
y_score_rf_lbp = rf_lbp.predict_proba(X_test_lbp)
y_pred_rf_lbp = rf_lbp.predict(X_test_lbp)
rf_lbp_cm = confusion_matrix(y_test, y_pred_rf_lbp)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_rf_lbp[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_rf_lbp.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Random Forest with LBP features')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("Random Forest with LBP features accuracy:", rf_lbp_acc)
print("Random Forest with LBP features confusion matrix:")
print(rf_lbp_cm)
sns.heatmap(rf_lbp_cm, annot=True, cmap='Blues')


# Train and evaluate the SVM model with LBP features
svm_lbp = SVC(kernel='linear', C=1, random_state=42)
svm_lbp.fit(X_train_lbp, y_train)
svm_lbp_acc = svm_lbp.score(X_test_lbp, y_test)
y_score_svm_lbp = svm_lbp.decision_function(X_test_lbp)
y_pred_svm_lbp = svm_lbp.predict(X_test_lbp)
svm_lbp_cm = confusion_matrix(y_test, y_pred_svm_lbp)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_svm_lbp[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_svm_lbp.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for SVM with LBP features')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("SVM with LBP features accuracy:", svm_lbp_acc)
print("SVM with LBP features confusion matrix:")
print(svm_lbp_cm)
sns.heatmap(svm_lbp_cm, annot=True, cmap='Blues')


# Train and evaluate the KNN model with LBP features
knn_lbp = KNeighborsClassifier(n_neighbors=5)
knn_lbp.fit(X_train_lbp, y_train)
knn_lbp_acc = knn_lbp.score(X_test_lbp, y_test)
y_score_knn_lbp = knn_lbp.predict_proba(X_test_lbp)
y_pred_knn_lbp = knn_lbp.predict(X_test_lbp)
knn_lbp_cm = confusion_matrix(y_test, y_pred_knn_lbp)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_knn_lbp[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_knn_lbp.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for KNN with LBP features')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("KNN with LBP features accuracy:", knn_lbp_acc)
print("KNN with LBP features confusion matrix:")
print(knn_lbp_cm)
sns.heatmap(knn_lbp_cm, annot=True, cmap='Blues')




################




# Define Gabor filter bank
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                           sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

# Extract GWT features from the training data
X_train_gwt = []
for image in X_train_norm:
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    X_train_gwt.append(feats.ravel())
X_train_gwt = np.array(X_train_gwt)

# Extract GWT features from the test data
X_test_gwt = []
for image in X_test_norm:
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    X_test_gwt.append(feats.ravel())
X_test_gwt = np.array(X_test_gwt)



# Train and evaluate the Random Forest model with GWT features
rf_gwt = RandomForestClassifier(n_estimators=100, random_state=42)
rf_gwt.fit(X_train_gwt, y_train)
rf_gwt_acc = rf_gwt.score(X_test_gwt, y_test)
y_score_rf_gwt = rf_gwt.predict_proba(X_test_gwt)
y_pred_rf_gwt = rf_gwt.predict(X_test_gwt)
rf_gwt_cm = confusion_matrix(y_test, y_pred_rf_gwt)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_rf_gwt[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_rf_gwt.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Random Forest with GWT features')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("Random Forest with GWT features accuracy:", rf_gwt_acc)
print("Random Forest with GWT features confusion matrix:")
print(rf_gwt_cm)
sns.heatmap(rf_gwt_cm, annot=True, cmap='Blues')




# Train and evaluate the SVM model with GWT features
svm_gwt = SVC(kernel='linear', C=1, random_state=42)
svm_gwt.fit(X_train_gwt, y_train)
svm_gwt_acc = svm_gwt.score(X_test_gwt, y_test)
y_score_svm_gwt = svm_gwt.decision_function(X_test_gwt)
y_pred_svm_gwt = svm_gwt.predict(X_test_gwt)
svm_gwt_cm = confusion_matrix(y_test, y_pred_svm_gwt)

# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_svm_gwt[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_svm_gwt.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for SVM with GWT features')
plt.legend(loc="lower right")
plt.show()

# Print the accuracy and confusion matrix
print("SVM with GWT features accuracy:", svm_gwt_acc)
print("SVM with GWT features confusion matrix:")
print(svm_gwt_cm)
sns.heatmap(svm_gwt_cm, annot=True, cmap='Blues')




# Train and evaluate the KNN model with GWT features
knn_gwt = KNeighborsClassifier(n_neighbors=5)
knn_gwt.fit(X_train_gwt, y_train)
knn_gwt_acc = knn_gwt.score(X_test_gwt, y_test)
y_score_knn_gwt = knn_gwt.predict_proba(X_test_gwt)
y_pred_knn_gwt = knn_gwt.predict(X_test_gwt)
knn_gwt_cm = confusion_matrix(y_test, y_pred_knn_gwt)


# Convert the target labels to one-hot encoding
y_test_one_hot = to_categorical(y_test)

# Get the number of classes
n_classes = y_test_one_hot.shape[1]

# Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_knn_gwt[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score_knn_gwt.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for KNN with GWT features')
plt.legend(loc="lower right")
plt.show()


# Print the accuracy and confusion matrix
print("KNN with GWT features accuracy:", knn_gwt_acc)
print("KNN with GWT features confusion matrix:")
print(knn_gwt_cm)
sns.heatmap(knn_gwt_cm, annot=True, cmap='Blues')



