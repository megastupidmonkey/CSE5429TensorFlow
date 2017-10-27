#special Thanks to Hvass-Labs on github for providing an excellent tutorial and code
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.python.saved_model import builder as saved_model_builder

#Convolutional Layer 1
filter_size1 = 5
num_filters1 = 16

#convolutional Layer 2
filter_size2 = 5
num_filters2 = 36

#fully-connected layer
fc_size = 128

#importing MNIST for example purposes
#change this when we get actual data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot = True)

#printlines for data set sizes:
print("SIZE of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28 #change this when we get real data
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

#number of color channels in the image
num_channels = 1 #change this to 3 (RGB), MNIST is B/W

#number of classes for our output
num_classes = 10 #change this too, we only need 2 classes (selfie and no selfie)

#creates random weights for initialization
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
#creates random bias values for initialization
def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, 	        #Input is the previous Conv Layer
		   num_input_channels,  #Number of channels in previous layer
		   filter_size,         #Width/height of filter
		   num_filters,         #number of filters
		   use_pooling=True):   #use pooling (2x2)
	
	#shape of filter-weights for convolution (4d tensor)
	#format is determined by tensforflow itself
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	#create new weights based on filter shape
	weights = new_weights(shape=shape)
	#create a new bias for each filter
	biases = new_biases(length=num_filters)

	# Create the TensorFlow operation for convolution.
    	# Note the strides are set to 1 in all dimensions.
    	# The first and last stride must always be 1,
    	# because the first is for the image-number and
    	# the last is for the input-channel.
    	# But e.g. strides=[1, 2, 2, 1] would mean that the filter
    	# is moved 2 pixels across the x- and y-axis of the image.
    	# The padding is set to 'SAME' which means the input image
    	# is padded with zeroes so the size of the output is the same.
    	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

	#add biases to the convolution
	layer += biases

	#use pooling to down-sample image resolution?
	if use_pooling:
		#uses 2x2 max pooling
		#consider 2x2 windows and select largest value
		#then stride 2x2 instead of 1x1
		layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
	#RELU activation function:
	#sets negative values to zero, keeps positive same
	layer = tf.nn.relu(layer)

	#return results of the layer and the weights
	#weights are not necessary to return in practice
	return layer, weights

#flatten the 4d tensor into a 2d tensor so we can add it as an input to the FC layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

#create fully connected layer
def new_fc_layer(input,		 #the previous layer 
		 num_inputs,     #number of inputs from previous layer
		 num_outputs,    #number of outputs
		 use_relu=True): #use RELU?

	#create new weights and biases
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)
	
	#calsulate the layer as the matrix multiplication of 
	#the input and weights, then add biases
	layer = tf.matmul(input, weights) + biases
	
	#use relu?
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer	

#tensor placeholder for our images
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
#tensor placeholder for our conv layer (must be 4d)
x_img = tf.reshape(x, [-1, img_size, img_size, num_channels])
#placeholder for true labels of our images
y_true = tf.placeholder(tf.float32, shape=[None, 10], name = 'y_true')
#placeholder for class-number of images
y_true_cls = tf.argmax(y_true, dimension=1)

#create first convolutional layer
layer_conv1, weights_conv1 = \
	new_conv_layer(input=x_img,
		       num_input_channels = num_channels,
		       filter_size = filter_size1,
		       num_filters=num_filters1,
		       use_pooling=True)

#for managerial purposes
print "Layer 1:"
print layer_conv1 #prints the type of unit (relu in this case), and the shape 

#create second convolutional layer
layer_conv2, weights_conv2 = \
	new_conv_layer(input=layer_conv1,
		       num_input_channels=num_filters1,
		       filter_size = filter_size2,
		       num_filters=num_filters2,
		       use_pooling=True)
print "Layer 2:"
print layer_conv2 #prints the type of unit (relu again), and shape

#create flat layer
layer_flat, num_features = flatten_layer(layer_conv2)

print "Flat Layer:"
print layer_flat #prints the type and shape of flat layer

#create first FC layer
layer_fc1 = new_fc_layer(input=layer_flat,
			 num_inputs=num_features,
			 num_outputs=fc_size,
			 use_relu=True)

print "FC Layer 1:"
print layer_fc1 #prints the type and shape of fully connected layer 1

#create second FC layer
layer_fc2 = new_fc_layer(input=layer_fc1,
			 num_inputs=fc_size,
			 num_outputs=num_classes,
			 use_relu=False)

print "FC Layer 2:"
print layer_fc2

#squash outputs using softmax
y_pred = tf.nn.softmax(layer_fc2)
#set class as nearest integer to the largest element, this is the result of the network
y_pred_cls = tf.argmax(y_pred, dimension=1, name = "output_tensor")

#cost function: Cross_entropy
#use cross_entropy with logits to estimate error
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

#finds the total cost by averaging the cross_entropy of all images
cost = tf.reduce_mean(cross_entropy) 

#uses AdanOptimizer (better version of gradient descent) to minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#test correctness
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#number of accurate predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#create a tensorflow session to execute our graph
session = tf.Session()

#saves model to directory /model
builder = tf.saved_model.builder.SavedModelBuilder("./model")

#initialize all graph variables
session.run(tf.initialize_all_variables())

#train based on 64 images per batch
train_batch_size = 64 

total_iterations = 0
def optimize(num_iterations):
	global total_iterations
	start_time = time.time()
	for i in range(total_iterations, total_iterations + num_iterations):
		#get a batch of training examples
		#x_batch is the images
		#y_batch is the labels (actual class of the image)
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		
		#put batch into a dictionary with the correct names
		#basically just couple the above variables into a structure
		feed_dict_train = {x: x_batch, y_true: y_true_batch}
		
		#run the optimizer using our current batch for training
		session.run(optimizer, feed_dict=feed_dict_train)
		
		#Not Necessary for our purposes
		#prints updates every 100 iterations
		if i % 100 == 0:
			acc = session.run(accuracy, feed_dict = feed_dict_train)
			msg = "Optimization Iteration: {0:>6}, Training Accuracy {1:>6.1%}"
			print(msg.format(i + 1, acc))
		
		total_iterations += num_iterations
		end_time = time.time()
		time_dif = end_time - start_time

		#print time taken
		print("time usage: " + str(timedelta(seconds=int(round(time_dif)))))
		
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

#run the model a number of times
optimize(num_iterations=10000)

#save model (courtesy of daniel persson)
builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING])
builder.save(True)

#print test accuracy
print_test_accuracy()
