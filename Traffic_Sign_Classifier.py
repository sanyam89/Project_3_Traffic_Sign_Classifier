#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle


# TODO: Fill this in based on where you saved the training and testing data

training_file = "data/train.p"
validation_file= "data/valid.p"
testing_file = "data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[1000].shape
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[3]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import pandas as pd
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

import csv
data = pd.read_csv('signnames.csv')
class_id = data.ClassId
sign_name = data.SignName
plt.figure(figsize=(12,35))
for i in range (0,n_classes):
    plt.subplot(15, 3, i+1)
    plt.title(str(class_id[i])+'. '+sign_name[i])
    plt.imshow(X_train[y_train==i][0, :, :, :])
    plt.axis('off')


# In[4]:


# PLOTTING COUNT OF EACH SIGN IN THE TRAINING DATA SET

hist, bins = np.histogram(y_train, bins = n_classes)
labels = np.arange(len(sign_name))
fig, ax = plt.subplots(figsize = (15,20))
ax.barh(labels,hist)
ax.set_yticks(labels)
ax.set_yticklabels(sign_name)
ax.set_xlabel('Number of each traffic sign')
ax.set_title('Traffic sign distributon')
plt.show()


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[5]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import cv2

def normalize1(img):
    return (img-128)/128

def normalize2(img):
    return img / 255

def normalize3(img):
    return img / np.max(img)

def gaussian(img):
    gb_img = cv2.GaussianBlur(img,(5,5),0)
    gb_img = cv2.addWeighted(img, 2, gb_img, -1, 0)
    return gb_img

plt.figure(figsize=(12,35))
k=1
for i in range (0,10):
    j = np.random.randint(0,n_classes)
    img = X_train[y_train==j][0, :, :, :]
   
    plt.subplot(10,3,k)
    plt.imshow(img)
    k+=1
    plt.axis('off')
    plt.title(str(class_id[j])+'. '+sign_name[j])
    
    plt.subplot(10,3,k)
    plt.imshow(normalize1(img))  #(normalize2(img))
    k+=1
    plt.axis('off')
    plt.title('Normalize 1 Func')
    plt.subplot(10,3,k)
    plt.imshow(normalize3(img))  #(gaussian_img)
    k+=1
    plt.axis('off')
    plt.title('Normalize 3 Func')
    #print(normalize3(img).shape)
          
plt.show()


# ### Model Architecture

# In[6]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

EPOCHS = 25
BATCH_SIZE = 100
rate = 0.0005 # 0.001 works good


# ### 5-Layer CNN definition
# 
# ##### Initial Image : 32x32x3
# ##### Layer 1 = Convolutional layer 1 : 28x28x24
# ##### Layer 2 = Convolutional layer 2 : 10x10x64
# ##### Layer 3 = Fully Connected layer 1 : 1x280
# ##### Layer 4 = Fully Connected layer 2 : 1x84
# ##### Layer 5 = Fully Connected layer 3 : 1x43

# In[7]:


from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    op_depth_conv1 = 24
    op_depth_1by1 = 48
    op_depth_conv2 = 64
    
    print('image shape = ',x.shape)
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x op_depth_conv1.
    fw1 = tf.Variable(tf.truncated_normal((5, 5, 3, op_depth_conv1), mean = mu, stddev = sigma)) # (height, width, input_depth, output_depth)
    fb1 = tf.Variable(tf.zeros(op_depth_conv1))
    strides = [1, 1, 1, 1] # (batch, height, width, depth)
    padding = 'VALID'
    conv_l1 = tf.nn.conv2d(x, fw1, strides, padding) + fb1
    
    # Layer 1: Activation.
    conv_l1 = tf.nn.relu(conv_l1)
    print('conv_L1 shape before pooling = ',conv_l1.shape)
    
    # Layer 1: Pooling. Input = 28x28xop_depth_conv1. Output = 14x14xop_depth_conv1.
    conv_l1 = tf.nn.max_pool(conv_l1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
    print('conv_L1 shape = ',conv_l1.shape)
    
    
    # Layer 1a: 1x1 Convolution. Input = 28x28x op_depth_conv1. Output = 28x28x op_depth_conv1.
    fw1by1 = tf.Variable(tf.truncated_normal((1, 1, op_depth_conv1, op_depth_1by1), mean = mu, stddev = sigma)) # (height, width, input_depth, output_depth)
    fb1by1 = tf.Variable(tf.zeros(op_depth_1by1))
    strides = [1, 1, 1, 1] # (batch, height, width, depth)
    padding = 'VALID'
    conv_l1by1 = tf.nn.conv2d(conv_l1, fw1by1, strides, padding)# + fb1by1
    
    # Layer 1a: Activation.
    conv_l1by1 = tf.nn.relu(conv_l1by1)
    
    # Layer 1a: Pooling. Input = 28x28xop_depth_conv1. Output = 14x14xop_depth_conv1.
    conv_l1by1 = tf.nn.max_pool(conv_l1by1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')    
    print('conv_L1by1 convolution shape = ',conv_l1by1.shape)
    

    # Layer 2: Convolutional. Output = 10x10xop_depth_conv2.
    fw2 = tf.Variable(tf.truncated_normal((5, 5, op_depth_conv1, op_depth_conv2), mean = mu, stddev = sigma)) # (height, width, input_depth, output_depth)
    fb2 = tf.Variable(tf.zeros(op_depth_conv2))
    strides = [1, 1, 1, 1] # (batch, height, width, depth)
    padding = 'VALID'
    conv_l2 = tf.nn.conv2d(conv_l1, fw2, strides, padding) + fb2
    
    # Layer 2: Activation.
    conv_l2 = tf.nn.relu(conv_l2)
    
    
    print('conv_L2 shape before pooling = ',conv_l2.shape)
    
    # Layer 2: Pooling. Input = 10x10xop_depth_conv2. Output = 5x5xop_depth_conv2.
    conv_l2 = tf.nn.max_pool(conv_l2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
    print('conv_L2 shape = ',conv_l2.shape)
    
    # Flatten. Input = 5x5xop_depth_conv2. Output = 1200.
    flat = flatten(conv_l2)
    print('flat_shape = ',flat.shape)
    
    
    # Layer 3: Fully Connected. Input = 1200. Output = 280.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5*5*op_depth_conv2, 280), mean = mu, stddev = sigma)) # 64,280
    fc1_b = tf.Variable(tf.zeros(280))
    fc1   = tf.matmul(flat, fc1_W) + fc1_b
    
    # Layer 3: Activation.
    fc1 = tf.nn.relu(fc1)
    print('fully connected 1 shape = ',fc1.shape)
    
    # Layer 4: Fully Connected. Input = 280. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(280, 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Layer 4: Activation.
    fc2 = tf.nn.relu(fc2)
    
    print('fully connected 2 shape = ',fc2.shape)
    
    # Layer 5: Fully Connected. Input = 84. Output = n_classes = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits   = tf.matmul(fc2, fc3_W) + fc3_b
    print('fully connected 3 shape FINAL = ',logits.shape)
    
    return logits


# ### Features and Labels placeholer definitions

# In[8]:


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)


# ### Training Pipeline

# In[9]:


logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# ### MODEL EVALUATION PIPELINE

# In[10]:


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[11]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

#  Split the data into training/validation/testing sets here.
#from sklearn.model_selection import train_test_split
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# NORMALIZE THE IMAGE DATA
X_train = normalize3(X_train)
X_test = normalize3(X_test) 
X_valid = normalize3(X_valid)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            #print(batch_y.shape)
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


# In[12]:


## TEST THE MODEL - ONLY RUN ONCE

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[13]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import glob, os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

new_test_images = [cv2.imread(file) for file in glob.glob('New_images_for_test/*.*')]
new_images_classification = ([os.path.basename(file) for file in glob.glob('New_images_for_test/*.*')])

plt.figure(figsize=(12,35))
new_images = []

for index,img_value in enumerate(new_test_images):
    
    img_value = cv2.cvtColor(img_value,cv2.COLOR_BGR2RGB)    #Convert from BGR to RGB
    resize_img = cv2.resize(img_value,(32,32))         # Resize the image 
    new_images.append(resize_img)      # append images to a stack
    # remove the file extensions from the file name as the first part is the traffic sign number
    new_images_classification[index] = new_images_classification[index].split('.', 1)[0]
    
    plt.subplot(7,3,index+1)
    plt.imshow(resize_img)
    plt.title(new_images_classification[index])

new_images = normalize3(new_images)    
#print(np.array(new_images).shape)
#print(new_images_classification)


# ### Predict the Sign Type for Each Image

# In[14]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

#new_images = normalize3(new_images)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    # PREDICTION OF LABELS
    signs_pred_id = sess.run(logits, feed_dict={x: new_images})
    predicted_labels = np.argmax(signs_pred_id, axis=1)
    
    # PREDICTION ACCURACY
    test_accuracy = evaluate(new_images, new_images_classification)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    


# ### Analyze Performance

# In[15]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# ### Print out the top five softmax probabilities for random images from THE TEST DATA. 

# In[16]:


X_test = normalize3(X_test)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    softmax = sess.run(tf.nn.softmax(logits), feed_dict={x: X_test})
    top5_prob = sess.run(tf.nn.top_k(softmax, k=5)) #, feed_dict={x: new_images})

k=1
plt.figure(figsize=(20, 50))
for i in range(0,10):
    rand = np.random.randint(0,n_classes)
    plt.subplot(10, 2, k)
    plt.imshow(np.array(X_test)[rand]) 
    plt.title(sign_name[class_id == np.int(y_test[rand])]) 
    plt.axis('off')
    k+=1
    plt.subplot(10, 2, k)
    plt.barh(np.arange(1, 6, 1), top5_prob.values[rand, :])
    labs=[sign_name[j] for j in top5_prob.indices[rand]]
    plt.yticks(np.arange(1, 6, 1), labs)
    k+=1
plt.show()


# ### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 

# In[17]:


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    softmax = sess.run(tf.nn.softmax(logits), feed_dict={x: new_images})
    top5_prob = sess.run(tf.nn.top_k(softmax, k=5)) #, feed_dict={x: new_images})
    
k=1
plt.figure(figsize=(20, 50))
for i in range(len(new_images)):
    plt.subplot(len(new_images), 2, k)
    plt.imshow(np.array(new_images)[i]) 
    plt.title(sign_name[class_id == np.int(new_images_classification[i])]) 
    plt.axis('off')
    k+=1
    plt.subplot(len(new_images), 2, k)
    plt.barh(np.arange(1, 6, 1), top5_prob.values[i, :])
    labs=[sign_name[j] for j in top5_prob.indices[i]]
    plt.yticks(np.arange(1, 6, 1), labs)
    k+=1
plt.show()


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[18]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


# In[19]:


#outputFeatureMap(new_images[1],tf_ativation = fc1)


# In[21]:


import tensorflow as tf

import glob, os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    ix = int(np.random.random() * X_test.shape[0])
    random_image = np.expand_dims(X_test[ix], axis=0) 
    print('Feature maps for', sign_name[y_test[ix]]) 
    plt.imshow(X_test[ix]) 
    plt.show() 
    print('First convolutional layer') 
    outputFeatureMap(random_image, LeNet(normalize3(random_image)).conv_l1, plt_num=1) 
    print('Second convolutional layer') 
    outputFeatureMap(random_image, LeNet(normalize3(random_image)).conv_l2, plt_num=2)


# In[ ]:




