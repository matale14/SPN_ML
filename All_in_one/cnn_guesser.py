import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import time, operator
from os import listdir
from os.path import isfile, join
import shutil
import time

def cnn_guesser(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]


    for i in range(0, len(onlyfiles)):
        j = onlyfiles[i]
        onlyfiles[i] = os.path.join(folder, j)


    files = []
    for f in onlyfiles:
        files.append(f)


    image_size=256
    num_channels=3
    results = {}


    ## Let us restore the saved model
    with tf.Session(graph=tf.Graph()) as sess:
        # Step-1: Recreate the network graph. At this step only graph is created.
        saver = tf.train.import_meta_graph('model/spai_model.meta')
        # Step-2: Now let's load the weights saved using the restore method.
        #saver.restore(sess, tf.train.latest_checkpoint('./'))
        tf.initialize_all_variables().run()
        # Reading the image using OpenCV
        for f in files:
            image = cv2.imread(f)
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
                graph = tf.get_default_graph()
                x = graph.get_tensor_by_name = "x:0"
                y_pred = graph.get_tensor_by_name ="y_pred:0"

                result=sess.run(y_pred, feed_dict={x: x_batch})
                res = result[0].tolist()
                #max_value = max(res)
                #max_index = np.where(res==max_value)
                #max_index = max_index[0][0]
                results[f] = res

    return(results)       
