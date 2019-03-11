import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import time, operator
from os import listdir
from os.path import isfile, join
import shutil


test_data = os.listdir('spai_dataset/test')

for i in range(len(test_data)):
    test_data[i] = test_data[i][:-4]


#test_data = sorted(test_data)

infile = open('classifications.csv', 'r')
infile.readline() # skips first line

for line in infile:
    words = line.split(',')
    if words[1] in test_data:
        test_labels.append(int(words[0]))

# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_path=sys.argv[1]
#filename = dir_path +'/' +image_path
image_size=256
num_channels=3
results = []
pics = []

## Let us restore the saved model
with tf.Session(graph=tf.Graph()) as sess:
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('spai_model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Reading the image using OpenCV
    for filename in sorted(os.listdir('spai_dataset/test')):
        image = cv2.imread(os.path.join('spai_dataset/test',filename))
        if image is not None:
            images = []
            # Resizing the image to our desired size and preprocessing will be done exactly as done during training
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            images.append(image)
            images = np.array(images, dtype=np.uint8)
            images = images.astype('float32')
            images = np.multiply(images, 1.0/255.0)
            #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
            x_batch = images.reshape(1, image_size,image_size,num_channels)



            # Accessing the default graph which we have restored
            graph = tf.get_default_graph()

            # Now, let's get hold of the op that we can be processed to get the output.
            # In the original network y_pred is the tensor that is the prediction of the network
            y_pred = graph.get_tensor_by_name ="y_pred:0"

            ## Let's feed the images to the input placeholders
            x= graph.get_tensor_by_name = "x:0"
            y_true = graph.get_tensor_by_name ="y_true:0"
            y_test_images = np.zeros((1, len(os.listdir('spai_dataset/images'))))


            ### Creating the feed_dict that is required to be fed to calculate y_pred
            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result=sess.run(y_pred, feed_dict=feed_dict_testing)
            # result is of this format [probabiliy_of_index1 probability_of_index2 etc.]
            results.append(result.argmax())

            res = result[0]
            max_value = max(res)
            max_index = np.where(res==max_value)
            max_index = max_index[0][0]

    con= tf.confusion_matrix(np.array(test_labels), np.array(results), num_classes=5)
    con2 = sess.run(con)
    success = con2[0][0]+con2[1][1]
    accuracy = success/np.sum(con2)