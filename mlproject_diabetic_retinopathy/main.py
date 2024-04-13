<html>
<head>
<title>main.py</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style type="text/css">
.s0 { color: #a9b7c6;}
.s1 { color: #629755; font-style: italic;}
.s2 { color: #cc7832;}
.s3 { color: #808080;}
.s4 { color: #6a8759;}
.s5 { color: #6897bb;}
</style>
</head>
<body bgcolor="#2b2b2b">
<table CELLSPACING=0 CELLPADDING=5 COLS=1 WIDTH="100%" BGCOLOR="#606060" >
<tr><td><center>
<font face="Arial, Helvetica" color="#000000">
main.py</font>
</center></td></tr></table>
<pre>
<span class="s1">&quot;&quot;&quot; 
 
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
 
&quot;&quot;&quot;</span>


<span class="s2">import </span><span class="s0">cv2</span>
<span class="s2">import </span><span class="s0">numpy </span><span class="s2">as </span><span class="s0">np</span>
<span class="s2">import </span><span class="s0">pandas </span><span class="s2">as </span><span class="s0">pd</span>
<span class="s2">import </span><span class="s0">matplotlib.pyplot </span><span class="s2">as </span><span class="s0">plt</span>
<span class="s2">import </span><span class="s0">os</span>
<span class="s2">from </span><span class="s0">keras.models </span><span class="s2">import </span><span class="s0">Sequential</span>
<span class="s2">from </span><span class="s0">keras.layers </span><span class="s2">import </span><span class="s0">Dense</span><span class="s2">, </span><span class="s0">Conv2D</span><span class="s2">, </span><span class="s0">Flatten</span><span class="s2">, </span><span class="s0">MaxPooling2D</span>
<span class="s2">from </span><span class="s0">sklearn.model_selection </span><span class="s2">import </span><span class="s0">train_test_split</span>
<span class="s2">from </span><span class="s0">sklearn.ensemble </span><span class="s2">import </span><span class="s0">RandomForestClassifier</span>
<span class="s2">from </span><span class="s0">sklearn </span><span class="s2">import </span><span class="s0">svm</span>
<span class="s2">from </span><span class="s0">sklearn.svm </span><span class="s2">import </span><span class="s0">SVC</span>
<span class="s2">from </span><span class="s0">sklearn.neighbors </span><span class="s2">import </span><span class="s0">KNeighborsClassifier</span>
<span class="s2">from </span><span class="s0">sklearn.metrics </span><span class="s2">import </span><span class="s0">confusion_matrix</span>
<span class="s2">import </span><span class="s0">seaborn </span><span class="s2">as </span><span class="s0">sns</span>
<span class="s2">from </span><span class="s0">skimage.feature </span><span class="s2">import </span><span class="s0">hog</span>
<span class="s2">from </span><span class="s0">skimage.feature </span><span class="s2">import </span><span class="s0">local_binary_pattern</span>
<span class="s2">from </span><span class="s0">scipy </span><span class="s2">import </span><span class="s0">ndimage </span><span class="s2">as </span><span class="s0">ndi</span>
<span class="s2">from </span><span class="s0">skimage.filters </span><span class="s2">import </span><span class="s0">gabor_kernel</span>
<span class="s2">from </span><span class="s0">sklearn.metrics </span><span class="s2">import </span><span class="s0">roc_curve</span><span class="s2">, </span><span class="s0">auc</span>
<span class="s2">from </span><span class="s0">keras.utils </span><span class="s2">import </span><span class="s0">to_categorical</span>


<span class="s2">from </span><span class="s0">keras.applications.vgg16 </span><span class="s2">import </span><span class="s0">VGG16</span>
<span class="s2">from </span><span class="s0">sklearn.preprocessing </span><span class="s2">import </span><span class="s0">StandardScaler</span>
<span class="s2">from </span><span class="s0">skimage </span><span class="s2">import </span><span class="s0">io</span>
<span class="s2">from </span><span class="s0">sklearn.metrics </span><span class="s2">import </span><span class="s0">accuracy_score</span>





<span class="s3"># Define the directory that contains the fundus images</span>
<span class="s0">dir_path = </span><span class="s4">'D:/ML using python/Project final/new'</span>

<span class="s3"># Define the image size</span>
<span class="s0">img_size = (</span><span class="s5">224</span><span class="s2">, </span><span class="s5">224</span><span class="s0">)</span>


<span class="s3"># Define a function to preprocess the images</span>
<span class="s2">def </span><span class="s0">preprocess_images(dir_path</span><span class="s2">, </span><span class="s0">img_size):</span>
    <span class="s3"># Get the list of image file names</span>
    <span class="s0">file_names = os.listdir(dir_path)</span>

    <span class="s3"># Create an empty NumPy array to store the preprocessed images</span>
    <span class="s0">preprocessed_images = np.empty((len(file_names)</span><span class="s2">, </span><span class="s0">img_size[</span><span class="s5">0</span><span class="s0">]</span><span class="s2">, </span><span class="s0">img_size[</span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s5">3</span><span class="s0">))</span>

    <span class="s3"># Loop over the image file names and preprocess each image</span>
    <span class="s2">for </span><span class="s0">i</span><span class="s2">, </span><span class="s0">file_name </span><span class="s2">in </span><span class="s0">enumerate(file_names):</span>
        <span class="s3"># Load the image</span>
        <span class="s0">img = cv2.imread(os.path.join(dir_path</span><span class="s2">, </span><span class="s0">file_name))</span>

        <span class="s3"># Convert the image to RGB</span>
        <span class="s0">rgb = cv2.cvtColor(img</span><span class="s2">, </span><span class="s0">cv2.COLOR_BGR2RGB)</span>

        <span class="s3"># Resize the image</span>
        <span class="s0">resized = cv2.resize(rgb</span><span class="s2">, </span><span class="s0">img_size)</span>

        <span class="s3"># Normalize the pixel intensities</span>
        <span class="s0">normalized = resized / </span><span class="s5">223.0</span>

        <span class="s3"># Store the preprocessed image in the NumPy array</span>
        <span class="s0">preprocessed_images[i] = normalized</span>

    <span class="s2">return </span><span class="s0">preprocessed_images</span>

<span class="s3"># Preprocess the images</span>
<span class="s0">preprocessed_images = preprocess_images(dir_path</span><span class="s2">, </span><span class="s0">img_size)</span>

<span class="s3"># Save the preprocessed images to a NumPy binary file</span>
<span class="s0">np.save(</span><span class="s4">'preprocessed_images.npy'</span><span class="s2">, </span><span class="s0">preprocessed_images)</span>

<span class="s3"># Load the .npy file</span>
<span class="s0">data = np.load(</span><span class="s4">'preprocessed_images.npy'</span><span class="s0">)</span>



<span class="s4">&quot;&quot;&quot; 
# Show each image in the data array 
for i in range(data.shape[0]): 
    plt.imshow(data[i], cmap='gray') 
    plt.title(f'Image {i+1}') 
    plt.show() 
&quot;&quot;&quot;</span>



<span class="s3"># Load the CSV file containing the image labels</span>
<span class="s0">df = pd.read_csv(</span><span class="s4">'D:/ML using python/Project final/sample_Label.csv'</span><span class="s0">)</span>

<span class="s3"># Extract the label column from the DataFrame</span>
<span class="s0">labels = df[</span><span class="s4">'level'</span><span class="s0">].values</span>


<span class="s4">&quot;&quot;&quot; 
# Display the first 5 images and their labels 
for i in range(10): 
    plt.imshow(data[i], cmap='gray') 
    plt.title(f'Label: {labels[i]}') 
    plt.show() 
&quot;&quot;&quot;</span>




<span class="s3"># Split the dataset into training and testing sets</span>
<span class="s0">X_train</span><span class="s2">, </span><span class="s0">X_test</span><span class="s2">, </span><span class="s0">y_train</span><span class="s2">, </span><span class="s0">y_test = train_test_split(data</span><span class="s2">, </span><span class="s0">labels</span><span class="s2">, </span><span class="s0">test_size=</span><span class="s5">0.5</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>










<span class="s3"># Extract features using pre-trained VGG16 model</span>
<span class="s0">base_model = VGG16(weights=</span><span class="s4">'imagenet'</span><span class="s2">, </span><span class="s0">include_top=</span><span class="s2">False, </span><span class="s0">input_shape=img_size+(</span><span class="s5">3</span><span class="s2">,</span><span class="s0">))</span>
<span class="s0">X_train_features = base_model.predict(X_train)</span>
<span class="s0">X_test_features = base_model.predict(X_test)</span>

<span class="s3"># Reshape the features for feeding to CNN model</span>
<span class="s0">X_train_features = X_train_features.reshape(X_train_features.shape[</span><span class="s5">0</span><span class="s0">]</span><span class="s2">, </span><span class="s0">-</span><span class="s5">1</span><span class="s0">)</span>
<span class="s0">X_test_features = X_test_features.reshape(X_test_features.shape[</span><span class="s5">0</span><span class="s0">]</span><span class="s2">, </span><span class="s0">-</span><span class="s5">1</span><span class="s0">)</span>

<span class="s3"># Normalize the data</span>
<span class="s0">X_train_norm = X_train_features / X_train_features.max()</span>
<span class="s0">X_test_norm = X_test_features / X_train_features.max()</span>



<span class="s3"># Train and evaluate the CNN model</span>
<span class="s2">def </span><span class="s0">create_cnn_model():</span>
    <span class="s0">model = Sequential()</span>
    <span class="s0">model.add(Dense(</span><span class="s5">128</span><span class="s2">, </span><span class="s0">activation=</span><span class="s4">'relu'</span><span class="s2">, </span><span class="s0">input_shape=X_train_norm.shape[</span><span class="s5">1</span><span class="s0">:]))</span>
    <span class="s0">model.add(Dense(</span><span class="s5">5</span><span class="s2">, </span><span class="s0">activation=</span><span class="s4">'softmax'</span><span class="s0">))</span>
    <span class="s0">model.compile(optimizer=</span><span class="s4">'adam'</span><span class="s2">, </span><span class="s0">loss=</span><span class="s4">'sparse_categorical_crossentropy'</span><span class="s2">, </span><span class="s0">metrics=[</span><span class="s4">'accuracy'</span><span class="s0">])</span>

    <span class="s2">return </span><span class="s0">model</span>

<span class="s0">model = create_cnn_model()</span>
<span class="s0">model.fit(X_train_norm</span><span class="s2">, </span><span class="s0">y_train</span><span class="s2">, </span><span class="s0">epochs=</span><span class="s5">10</span><span class="s2">, </span><span class="s0">batch_size=</span><span class="s5">32</span><span class="s2">, </span><span class="s0">validation_data=(X_test_norm</span><span class="s2">, </span><span class="s0">y_test))</span>
<span class="s0">cnn_acc = model.evaluate(X_test_norm</span><span class="s2">, </span><span class="s0">y_test)[</span><span class="s5">1</span><span class="s0">]</span>
<span class="s0">y_score_cnn = model.predict(X_test_norm)</span>
<span class="s0">y_pred_cnn = model.predict(X_test_norm)</span>
<span class="s0">y_pred_cnn_classes = np.argmax(y_pred_cnn</span><span class="s2">, </span><span class="s0">axis=</span><span class="s5">1</span><span class="s0">)</span>
<span class="s0">cnn_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_cnn_classes)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_cnn[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_cnn.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for CNN using pre-trained VGG16 model'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;CNN with VGG16 model accuracy:&quot;</span><span class="s2">, </span><span class="s0">cnn_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;CNN confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(cnn_cm)</span>
<span class="s0">sns.heatmap(cnn_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>












<span class="s3"># Define the directory that contains the fundus images</span>
<span class="s0">dir_path = </span><span class="s4">'D:/ML using python/Project final/new'</span>

<span class="s3"># Define the image size</span>
<span class="s0">img_size = (</span><span class="s5">256</span><span class="s2">, </span><span class="s5">256</span><span class="s0">)</span>


<span class="s3"># Define a function to preprocess the images</span>
<span class="s2">def </span><span class="s0">preprocess_images(dir_path</span><span class="s2">, </span><span class="s0">img_size):</span>
    <span class="s3"># Get the list of image file names</span>
    <span class="s0">file_names = os.listdir(dir_path)</span>

    <span class="s3"># Create an empty NumPy array to store the preprocessed images</span>
    <span class="s0">preprocessed_images = np.empty((len(file_names)</span><span class="s2">, </span><span class="s0">img_size[</span><span class="s5">0</span><span class="s0">]</span><span class="s2">, </span><span class="s0">img_size[</span><span class="s5">1</span><span class="s0">]))</span>

    <span class="s3"># Loop over the image file names and preprocess each image</span>
    <span class="s2">for </span><span class="s0">i</span><span class="s2">, </span><span class="s0">file_name </span><span class="s2">in </span><span class="s0">enumerate(file_names):</span>
        <span class="s3"># Load the image</span>
        <span class="s0">img = cv2.imread(os.path.join(dir_path</span><span class="s2">, </span><span class="s0">file_name))</span>

        <span class="s3"># Convert the image to grayscale</span>
        <span class="s0">gray = cv2.cvtColor(img</span><span class="s2">, </span><span class="s0">cv2.COLOR_BGR2GRAY)</span>

        <span class="s3"># Resize the image</span>
        <span class="s0">resized = cv2.resize(gray</span><span class="s2">, </span><span class="s0">img_size)</span>

        <span class="s3"># Normalize the pixel intensities</span>
        <span class="s0">normalized = resized / </span><span class="s5">255.0</span>

        <span class="s3"># Store the preprocessed image in the NumPy array</span>
        <span class="s0">preprocessed_images[i] = normalized</span>

    <span class="s2">return </span><span class="s0">preprocessed_images</span>


<span class="s3"># Preprocess the images</span>
<span class="s0">preprocessed_images = preprocess_images(dir_path</span><span class="s2">, </span><span class="s0">img_size)</span>

<span class="s3"># Save the preprocessed images to a NumPy binary file</span>
<span class="s0">np.save(</span><span class="s4">'preprocessed_images.npy'</span><span class="s2">, </span><span class="s0">preprocessed_images)</span>

<span class="s3"># Load the .npy file</span>
<span class="s0">data = np.load(</span><span class="s4">'preprocessed_images.npy'</span><span class="s0">)</span>


<span class="s4">&quot;&quot;&quot; 
 
# Show each image in the data array 
for i in range(data.shape[0]): 
    plt.imshow(data[i], cmap='gray') 
    plt.title(f'Image {i+1}') 
    plt.show() 
 
&quot;&quot;&quot;</span>


<span class="s3"># Load the CSV file containing the image labels</span>
<span class="s0">df = pd.read_csv(</span><span class="s4">'D:/ML using python/Project final/sample_Label.csv'</span><span class="s0">)</span>

<span class="s3"># Extract the label column from the DataFrame</span>
<span class="s0">labels = df[</span><span class="s4">'level'</span><span class="s0">].values</span>

<span class="s4">&quot;&quot;&quot; 
 
# Display the first 5 images and their labels 
for i in range(10): 
    plt.imshow(data[i], cmap='gray') 
    plt.title(f'Label: {labels[i]}') 
    plt.show() 
 
&quot;&quot;&quot;</span>



<span class="s3"># Split the dataset into training and testing sets</span>
<span class="s0">X_train</span><span class="s2">, </span><span class="s0">X_test</span><span class="s2">, </span><span class="s0">y_train</span><span class="s2">, </span><span class="s0">y_test = train_test_split(data</span><span class="s2">, </span><span class="s0">labels</span><span class="s2">, </span><span class="s0">test_size=</span><span class="s5">0.5</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>



<span class="s3"># Define a function to normalize pixel values</span>
<span class="s2">def </span><span class="s0">normalize_pixels(data):</span>
    <span class="s2">return </span><span class="s0">data.astype(</span><span class="s4">'float32'</span><span class="s0">) / </span><span class="s5">255.0</span>

<span class="s3"># Preprocess the data</span>
<span class="s0">X_train_norm = normalize_pixels(X_train)</span>
<span class="s0">X_test_norm = normalize_pixels(X_test)</span>




<span class="s3"># Train and evaluate the CNN model</span>
<span class="s2">def </span><span class="s0">create_cnn_model():</span>
    <span class="s0">model = Sequential()</span>
    <span class="s0">model.add(Conv2D(</span><span class="s5">32</span><span class="s2">, </span><span class="s0">kernel_size=(</span><span class="s5">3</span><span class="s2">, </span><span class="s5">3</span><span class="s0">)</span><span class="s2">, </span><span class="s0">activation=</span><span class="s4">'relu'</span><span class="s2">, </span><span class="s0">input_shape=img_size+(</span><span class="s5">1</span><span class="s2">,</span><span class="s0">)))</span>
    <span class="s0">model.add(MaxPooling2D(pool_size=(</span><span class="s5">2</span><span class="s2">, </span><span class="s5">2</span><span class="s0">)))</span>
    <span class="s0">model.add(Conv2D(</span><span class="s5">64</span><span class="s2">, </span><span class="s0">kernel_size=(</span><span class="s5">3</span><span class="s2">, </span><span class="s5">3</span><span class="s0">)</span><span class="s2">, </span><span class="s0">activation=</span><span class="s4">'relu'</span><span class="s0">))</span>
    <span class="s0">model.add(MaxPooling2D(pool_size=(</span><span class="s5">2</span><span class="s2">, </span><span class="s5">2</span><span class="s0">)))</span>
    <span class="s0">model.add(Conv2D(</span><span class="s5">128</span><span class="s2">, </span><span class="s0">kernel_size=(</span><span class="s5">3</span><span class="s2">, </span><span class="s5">3</span><span class="s0">)</span><span class="s2">, </span><span class="s0">activation=</span><span class="s4">'relu'</span><span class="s0">))</span>
    <span class="s0">model.add(MaxPooling2D(pool_size=(</span><span class="s5">2</span><span class="s2">, </span><span class="s5">2</span><span class="s0">)))</span>
    <span class="s0">model.add(Flatten())</span>
    <span class="s0">model.add(Dense(</span><span class="s5">128</span><span class="s2">, </span><span class="s0">activation=</span><span class="s4">'relu'</span><span class="s0">))</span>
    <span class="s0">model.add(Dense(</span><span class="s5">5</span><span class="s2">, </span><span class="s0">activation=</span><span class="s4">'softmax'</span><span class="s0">))</span>
    <span class="s0">model.compile(optimizer=</span><span class="s4">'adam'</span><span class="s2">, </span><span class="s0">loss=</span><span class="s4">'sparse_categorical_crossentropy'</span><span class="s2">, </span><span class="s0">metrics=[</span><span class="s4">'accuracy'</span><span class="s0">])</span>

    <span class="s2">return </span><span class="s0">model</span>

<span class="s0">model = create_cnn_model()</span>
<span class="s0">model.fit(X_train_norm</span><span class="s2">, </span><span class="s0">y_train</span><span class="s2">, </span><span class="s0">epochs=</span><span class="s5">10</span><span class="s2">, </span><span class="s0">batch_size=</span><span class="s5">32</span><span class="s2">, </span><span class="s0">validation_data=(X_test_norm</span><span class="s2">, </span><span class="s0">y_test))</span>
<span class="s0">cnn_acc = model.evaluate(X_test_norm</span><span class="s2">, </span><span class="s0">y_test)[</span><span class="s5">1</span><span class="s0">]</span>
<span class="s0">y_score_cnn = model.predict(X_test_norm)</span>
<span class="s0">y_pred_cnn_classes = np.argmax(y_score_cnn</span><span class="s2">, </span><span class="s0">axis=</span><span class="s5">1</span><span class="s0">)</span>
<span class="s0">cnn_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_cnn_classes)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_cnn[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_cnn.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for CNN'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s0">print(</span><span class="s4">&quot;CNN accuracy:&quot;</span><span class="s2">, </span><span class="s0">cnn_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;CNN confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(cnn_cm)</span>
<span class="s0">sns.heatmap(cnn_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>


<span class="s3"># Extract HOG features from the training data</span>
<span class="s0">X_train_hog = []</span>
<span class="s2">for </span><span class="s0">image </span><span class="s2">in </span><span class="s0">X_train_norm:</span>
    <span class="s0">hog_features = hog(image</span><span class="s2">, </span><span class="s0">orientations=</span><span class="s5">9</span><span class="s2">, </span><span class="s0">pixels_per_cell=(</span><span class="s5">8</span><span class="s2">, </span><span class="s5">8</span><span class="s0">)</span><span class="s2">, </span><span class="s0">cells_per_block=(</span><span class="s5">2</span><span class="s2">, </span><span class="s5">2</span><span class="s0">)</span><span class="s2">, </span><span class="s0">visualize=</span><span class="s2">False</span><span class="s0">)</span>
    <span class="s0">X_train_hog.append(hog_features)</span>
<span class="s0">X_train_hog = np.array(X_train_hog)</span>

<span class="s3"># Extract HOG features from the test data</span>
<span class="s0">X_test_hog = []</span>
<span class="s2">for </span><span class="s0">image </span><span class="s2">in </span><span class="s0">X_test_norm:</span>
    <span class="s0">hog_features = hog(image</span><span class="s2">, </span><span class="s0">orientations=</span><span class="s5">9</span><span class="s2">, </span><span class="s0">pixels_per_cell=(</span><span class="s5">8</span><span class="s2">, </span><span class="s5">8</span><span class="s0">)</span><span class="s2">, </span><span class="s0">cells_per_block=(</span><span class="s5">2</span><span class="s2">, </span><span class="s5">2</span><span class="s0">)</span><span class="s2">, </span><span class="s0">visualize=</span><span class="s2">False</span><span class="s0">)</span>
    <span class="s0">X_test_hog.append(hog_features)</span>
<span class="s0">X_test_hog = np.array(X_test_hog)</span>

<span class="s3"># Train and evaluate the Random Forest model with HOG features</span>
<span class="s0">rf_hog = RandomForestClassifier(n_estimators=</span><span class="s5">100</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>
<span class="s0">rf_hog.fit(X_train_hog</span><span class="s2">, </span><span class="s0">y_train)</span>

<span class="s3"># Predict on test set and get accuracy and confusion matrix</span>
<span class="s0">rf_hog_acc = rf_hog.score(X_test_hog</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_pred_rf_hog = rf_hog.predict(X_test_hog)</span>
<span class="s0">rf_hog_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_rf_hog)</span>

<span class="s3"># Predict probabilities for ROC curve</span>
<span class="s0">y_score_rf_hog = rf_hog.predict_proba(X_test_hog)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Compute ROC curve and ROC area for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_rf_hog[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_rf_hog.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot ROC curve</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for Random Forest with HOG features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;Random Forest with HOG features accuracy:&quot;</span><span class="s2">, </span><span class="s0">rf_hog_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;Random Forest with HOG features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(rf_hog_cm)</span>
<span class="s0">sns.heatmap(rf_hog_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>


<span class="s3"># Train and evaluate the SVM model with HOG features</span>
<span class="s0">svm_hog = SVC(kernel=</span><span class="s4">'linear'</span><span class="s2">, </span><span class="s0">C=</span><span class="s5">1</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>
<span class="s0">svm_hog.fit(X_train_hog</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">svm_hog_acc = svm_hog.score(X_test_hog</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_svm_hog = svm_hog.decision_function(X_test_hog)</span>
<span class="s0">y_pred_svm_hog = svm_hog.predict(X_test_hog)</span>
<span class="s0">svm_hog_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_svm_hog)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_svm_hog[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_svm_hog.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for SVM with HOG features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;SVM with HOG features accuracy:&quot;</span><span class="s2">, </span><span class="s0">svm_hog_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;SVM with HOG features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(svm_hog_cm)</span>
<span class="s0">sns.heatmap(svm_hog_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>


<span class="s3"># Train and evaluate the KNN model with HOG features</span>
<span class="s0">knn_hog = KNeighborsClassifier(n_neighbors=</span><span class="s5">5</span><span class="s0">)</span>
<span class="s0">knn_hog.fit(X_train_hog</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">knn_hog_acc = knn_hog.score(X_test_hog</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_knn_hog = knn_hog.predict_proba(X_test_hog)</span>
<span class="s0">y_pred_knn_hog = knn_hog.predict(X_test_hog)</span>
<span class="s0">knn_hog_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_knn_hog)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_knn_hog[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_knn_hog.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for KNN with HOG features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;KNN with HOG features accuracy:&quot;</span><span class="s2">, </span><span class="s0">knn_hog_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;KNN with HOG features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(knn_hog_cm)</span>
<span class="s0">sns.heatmap(knn_hog_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>






<span class="s3"># Define LBP parameters</span>
<span class="s0">radius = </span><span class="s5">1</span>
<span class="s0">n_points = </span><span class="s5">8 </span><span class="s0">* radius</span>
<span class="s0">METHOD = </span><span class="s4">'uniform'</span>

<span class="s3"># Extract LBP features from the training data</span>
<span class="s0">X_train_lbp = []</span>
<span class="s2">for </span><span class="s0">image </span><span class="s2">in </span><span class="s0">X_train_norm:</span>
    <span class="s0">image = (image * </span><span class="s5">255</span><span class="s0">).astype(np.uint8)</span>

    <span class="s0">lbp = local_binary_pattern(image</span><span class="s2">, </span><span class="s0">n_points</span><span class="s2">, </span><span class="s0">radius</span><span class="s2">, </span><span class="s0">METHOD)</span>
    <span class="s0">hist</span><span class="s2">, </span><span class="s0">_ = np.histogram(lbp.ravel()</span><span class="s2">, </span><span class="s0">bins=np.arange(</span><span class="s5">0</span><span class="s2">, </span><span class="s0">n_points + </span><span class="s5">3</span><span class="s0">)</span><span class="s2">, </span><span class="s0">range=(</span><span class="s5">0</span><span class="s2">, </span><span class="s0">n_points + </span><span class="s5">2</span><span class="s0">))</span>
    <span class="s0">X_train_lbp.append(hist)</span>
<span class="s0">X_train_lbp = np.array(X_train_lbp)</span>

<span class="s3"># Extract LBP features from the test data</span>
<span class="s0">X_test_lbp = []</span>
<span class="s2">for </span><span class="s0">image </span><span class="s2">in </span><span class="s0">X_test_norm:</span>
    <span class="s0">image = (image * </span><span class="s5">255</span><span class="s0">).astype(np.uint8)</span>

    <span class="s0">lbp = local_binary_pattern(image</span><span class="s2">, </span><span class="s0">n_points</span><span class="s2">, </span><span class="s0">radius</span><span class="s2">, </span><span class="s0">METHOD)</span>
    <span class="s0">hist</span><span class="s2">, </span><span class="s0">_ = np.histogram(lbp.ravel()</span><span class="s2">, </span><span class="s0">bins=np.arange(</span><span class="s5">0</span><span class="s2">, </span><span class="s0">n_points + </span><span class="s5">3</span><span class="s0">)</span><span class="s2">, </span><span class="s0">range=(</span><span class="s5">0</span><span class="s2">, </span><span class="s0">n_points + </span><span class="s5">2</span><span class="s0">))</span>
    <span class="s0">X_test_lbp.append(hist)</span>
<span class="s0">X_test_lbp = np.array(X_test_lbp)</span>

<span class="s3"># Train and evaluate the Random Forest model with LBP features</span>
<span class="s0">rf_lbp = RandomForestClassifier(n_estimators=</span><span class="s5">100</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>
<span class="s0">rf_lbp.fit(X_train_lbp</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">rf_lbp_acc = rf_lbp.score(X_test_lbp</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_rf_lbp = rf_lbp.predict_proba(X_test_lbp)</span>
<span class="s0">y_pred_rf_lbp = rf_lbp.predict(X_test_lbp)</span>
<span class="s0">rf_lbp_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_rf_lbp)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_rf_lbp[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_rf_lbp.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for Random Forest with LBP features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;Random Forest with LBP features accuracy:&quot;</span><span class="s2">, </span><span class="s0">rf_lbp_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;Random Forest with LBP features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(rf_lbp_cm)</span>
<span class="s0">sns.heatmap(rf_lbp_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>


<span class="s3"># Train and evaluate the SVM model with LBP features</span>
<span class="s0">svm_lbp = SVC(kernel=</span><span class="s4">'linear'</span><span class="s2">, </span><span class="s0">C=</span><span class="s5">1</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>
<span class="s0">svm_lbp.fit(X_train_lbp</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">svm_lbp_acc = svm_lbp.score(X_test_lbp</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_svm_lbp = svm_lbp.decision_function(X_test_lbp)</span>
<span class="s0">y_pred_svm_lbp = svm_lbp.predict(X_test_lbp)</span>
<span class="s0">svm_lbp_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_svm_lbp)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_svm_lbp[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_svm_lbp.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for SVM with LBP features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;SVM with LBP features accuracy:&quot;</span><span class="s2">, </span><span class="s0">svm_lbp_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;SVM with LBP features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(svm_lbp_cm)</span>
<span class="s0">sns.heatmap(svm_lbp_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>


<span class="s3"># Train and evaluate the KNN model with LBP features</span>
<span class="s0">knn_lbp = KNeighborsClassifier(n_neighbors=</span><span class="s5">5</span><span class="s0">)</span>
<span class="s0">knn_lbp.fit(X_train_lbp</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">knn_lbp_acc = knn_lbp.score(X_test_lbp</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_knn_lbp = knn_lbp.predict_proba(X_test_lbp)</span>
<span class="s0">y_pred_knn_lbp = knn_lbp.predict(X_test_lbp)</span>
<span class="s0">knn_lbp_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_knn_lbp)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_knn_lbp[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_knn_lbp.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for KNN with LBP features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;KNN with LBP features accuracy:&quot;</span><span class="s2">, </span><span class="s0">knn_lbp_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;KNN with LBP features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(knn_lbp_cm)</span>
<span class="s0">sns.heatmap(knn_lbp_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>




<span class="s3">################</span>




<span class="s3"># Define Gabor filter bank</span>
<span class="s0">kernels = []</span>
<span class="s2">for </span><span class="s0">theta </span><span class="s2">in </span><span class="s0">range(</span><span class="s5">4</span><span class="s0">):</span>
    <span class="s0">theta = theta / </span><span class="s5">4. </span><span class="s0">* np.pi</span>
    <span class="s2">for </span><span class="s0">sigma </span><span class="s2">in </span><span class="s0">(</span><span class="s5">1</span><span class="s2">, </span><span class="s5">3</span><span class="s0">):</span>
        <span class="s2">for </span><span class="s0">frequency </span><span class="s2">in </span><span class="s0">(</span><span class="s5">0.05</span><span class="s2">, </span><span class="s5">0.25</span><span class="s0">):</span>
            <span class="s0">kernel = np.real(gabor_kernel(frequency</span><span class="s2">, </span><span class="s0">theta=theta</span><span class="s2">,</span>
                                           <span class="s0">sigma_x=sigma</span><span class="s2">, </span><span class="s0">sigma_y=sigma))</span>
            <span class="s0">kernels.append(kernel)</span>

<span class="s3"># Extract GWT features from the training data</span>
<span class="s0">X_train_gwt = []</span>
<span class="s2">for </span><span class="s0">image </span><span class="s2">in </span><span class="s0">X_train_norm:</span>
    <span class="s0">feats = np.zeros((len(kernels)</span><span class="s2">, </span><span class="s5">2</span><span class="s0">)</span><span class="s2">, </span><span class="s0">dtype=np.double)</span>
    <span class="s2">for </span><span class="s0">k</span><span class="s2">, </span><span class="s0">kernel </span><span class="s2">in </span><span class="s0">enumerate(kernels):</span>
        <span class="s0">filtered = ndi.convolve(image</span><span class="s2">, </span><span class="s0">kernel</span><span class="s2">, </span><span class="s0">mode=</span><span class="s4">'wrap'</span><span class="s0">)</span>
        <span class="s0">feats[k</span><span class="s2">, </span><span class="s5">0</span><span class="s0">] = filtered.mean()</span>
        <span class="s0">feats[k</span><span class="s2">, </span><span class="s5">1</span><span class="s0">] = filtered.var()</span>
    <span class="s0">X_train_gwt.append(feats.ravel())</span>
<span class="s0">X_train_gwt = np.array(X_train_gwt)</span>

<span class="s3"># Extract GWT features from the test data</span>
<span class="s0">X_test_gwt = []</span>
<span class="s2">for </span><span class="s0">image </span><span class="s2">in </span><span class="s0">X_test_norm:</span>
    <span class="s0">feats = np.zeros((len(kernels)</span><span class="s2">, </span><span class="s5">2</span><span class="s0">)</span><span class="s2">, </span><span class="s0">dtype=np.double)</span>
    <span class="s2">for </span><span class="s0">k</span><span class="s2">, </span><span class="s0">kernel </span><span class="s2">in </span><span class="s0">enumerate(kernels):</span>
        <span class="s0">filtered = ndi.convolve(image</span><span class="s2">, </span><span class="s0">kernel</span><span class="s2">, </span><span class="s0">mode=</span><span class="s4">'wrap'</span><span class="s0">)</span>
        <span class="s0">feats[k</span><span class="s2">, </span><span class="s5">0</span><span class="s0">] = filtered.mean()</span>
        <span class="s0">feats[k</span><span class="s2">, </span><span class="s5">1</span><span class="s0">] = filtered.var()</span>
    <span class="s0">X_test_gwt.append(feats.ravel())</span>
<span class="s0">X_test_gwt = np.array(X_test_gwt)</span>



<span class="s3"># Train and evaluate the Random Forest model with GWT features</span>
<span class="s0">rf_gwt = RandomForestClassifier(n_estimators=</span><span class="s5">100</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>
<span class="s0">rf_gwt.fit(X_train_gwt</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">rf_gwt_acc = rf_gwt.score(X_test_gwt</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_rf_gwt = rf_gwt.predict_proba(X_test_gwt)</span>
<span class="s0">y_pred_rf_gwt = rf_gwt.predict(X_test_gwt)</span>
<span class="s0">rf_gwt_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_rf_gwt)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_rf_gwt[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_rf_gwt.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for Random Forest with GWT features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;Random Forest with GWT features accuracy:&quot;</span><span class="s2">, </span><span class="s0">rf_gwt_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;Random Forest with GWT features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(rf_gwt_cm)</span>
<span class="s0">sns.heatmap(rf_gwt_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>




<span class="s3"># Train and evaluate the SVM model with GWT features</span>
<span class="s0">svm_gwt = SVC(kernel=</span><span class="s4">'linear'</span><span class="s2">, </span><span class="s0">C=</span><span class="s5">1</span><span class="s2">, </span><span class="s0">random_state=</span><span class="s5">42</span><span class="s0">)</span>
<span class="s0">svm_gwt.fit(X_train_gwt</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">svm_gwt_acc = svm_gwt.score(X_test_gwt</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_svm_gwt = svm_gwt.decision_function(X_test_gwt)</span>
<span class="s0">y_pred_svm_gwt = svm_gwt.predict(X_test_gwt)</span>
<span class="s0">svm_gwt_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_svm_gwt)</span>

<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_svm_gwt[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_svm_gwt.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for SVM with GWT features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>

<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;SVM with GWT features accuracy:&quot;</span><span class="s2">, </span><span class="s0">svm_gwt_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;SVM with GWT features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(svm_gwt_cm)</span>
<span class="s0">sns.heatmap(svm_gwt_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>




<span class="s3"># Train and evaluate the KNN model with GWT features</span>
<span class="s0">knn_gwt = KNeighborsClassifier(n_neighbors=</span><span class="s5">5</span><span class="s0">)</span>
<span class="s0">knn_gwt.fit(X_train_gwt</span><span class="s2">, </span><span class="s0">y_train)</span>
<span class="s0">knn_gwt_acc = knn_gwt.score(X_test_gwt</span><span class="s2">, </span><span class="s0">y_test)</span>
<span class="s0">y_score_knn_gwt = knn_gwt.predict_proba(X_test_gwt)</span>
<span class="s0">y_pred_knn_gwt = knn_gwt.predict(X_test_gwt)</span>
<span class="s0">knn_gwt_cm = confusion_matrix(y_test</span><span class="s2">, </span><span class="s0">y_pred_knn_gwt)</span>


<span class="s3"># Convert the target labels to one-hot encoding</span>
<span class="s0">y_test_one_hot = to_categorical(y_test)</span>

<span class="s3"># Get the number of classes</span>
<span class="s0">n_classes = y_test_one_hot.shape[</span><span class="s5">1</span><span class="s0">]</span>

<span class="s3"># Calculate the ROC curve and AUC for each class</span>
<span class="s0">fpr = dict()</span>
<span class="s0">tpr = dict()</span>
<span class="s0">roc_auc = dict()</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot[:</span><span class="s2">, </span><span class="s0">i]</span><span class="s2">, </span><span class="s0">y_score_knn_gwt[:</span><span class="s2">, </span><span class="s0">i])</span>
    <span class="s0">roc_auc[i] = auc(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i])</span>

<span class="s3"># Compute micro-average ROC curve and ROC area</span>
<span class="s0">fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">_ = roc_curve(y_test_one_hot.ravel()</span><span class="s2">, </span><span class="s0">y_score_knn_gwt.ravel())</span>
<span class="s0">roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">] = auc(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">])</span>

<span class="s3"># Plot the ROC curves</span>
<span class="s0">plt.figure()</span>
<span class="s0">lw = </span><span class="s5">2</span>
<span class="s0">plt.plot(fpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">tpr[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]</span><span class="s2">, </span><span class="s0">color=</span><span class="s4">'deeppink'</span><span class="s2">,</span>
         <span class="s0">lw=lw</span><span class="s2">, </span><span class="s0">label=</span><span class="s4">'micro-average ROC curve (area = {0:0.2f})'</span>
         <span class="s4">''</span><span class="s0">.format(roc_auc[</span><span class="s4">&quot;micro&quot;</span><span class="s0">]))</span>
<span class="s2">for </span><span class="s0">i </span><span class="s2">in </span><span class="s0">range(n_classes):</span>
    <span class="s0">plt.plot(fpr[i]</span><span class="s2">, </span><span class="s0">tpr[i]</span><span class="s2">, </span><span class="s0">lw=lw</span><span class="s2">,</span>
             <span class="s0">label=</span><span class="s4">'ROC curve of class {0} (area = {1:0.2f})'</span>
             <span class="s4">''</span><span class="s0">.format(i</span><span class="s2">, </span><span class="s0">roc_auc[i]))</span>
<span class="s0">plt.plot([</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s0">[</span><span class="s5">0</span><span class="s2">, </span><span class="s5">1</span><span class="s0">]</span><span class="s2">, </span><span class="s4">'k--'</span><span class="s2">, </span><span class="s0">lw=lw)</span>
<span class="s0">plt.xlim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.0</span><span class="s0">])</span>
<span class="s0">plt.ylim([</span><span class="s5">0.0</span><span class="s2">, </span><span class="s5">1.05</span><span class="s0">])</span>
<span class="s0">plt.xlabel(</span><span class="s4">'False Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.ylabel(</span><span class="s4">'True Positive Rate'</span><span class="s0">)</span>
<span class="s0">plt.title(</span><span class="s4">'Receiver operating characteristic for KNN with GWT features'</span><span class="s0">)</span>
<span class="s0">plt.legend(loc=</span><span class="s4">&quot;lower right&quot;</span><span class="s0">)</span>
<span class="s0">plt.show()</span>


<span class="s3"># Print the accuracy and confusion matrix</span>
<span class="s0">print(</span><span class="s4">&quot;KNN with GWT features accuracy:&quot;</span><span class="s2">, </span><span class="s0">knn_gwt_acc)</span>
<span class="s0">print(</span><span class="s4">&quot;KNN with GWT features confusion matrix:&quot;</span><span class="s0">)</span>
<span class="s0">print(knn_gwt_cm)</span>
<span class="s0">sns.heatmap(knn_gwt_cm</span><span class="s2">, </span><span class="s0">annot=</span><span class="s2">True, </span><span class="s0">cmap=</span><span class="s4">'Blues'</span><span class="s0">)</span>



</pre>
</body>
</html>