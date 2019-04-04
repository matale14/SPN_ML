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

    #filter_path = os.path.join(folder, "Photos", "Filtered")
    filter_path = folder
    onlyfiles = [f for f in listdir(filter_path) if isfile(join(filter_path, f))]


    for i in range(0, len(onlyfiles)):
        j = onlyfiles[i]
        onlyfiles[i] = os.path.join(filter_path, j)


    files = []
    for f in onlyfiles:
        files.append(f)

    image_size=256
    num_channels=3
    results = []
    combiner = []
    pics = []


    ## Let us restore the saved model
    with tf.Session(graph=tf.Graph()) as sess:
        # Step-1: Recreate the network graph. At this step only graph is created.
        saver = tf.train.import_meta_graph('model/spai_model.meta')
        # Step-2: Now let's load the weights saved using the restore method.
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        #tf.initialize_all_variables().run()
        # Reading the image using OpenCV
        for filename in sorted(os.listdir(filter_path)):
            image = cv2.imread(os.path.join(filter_path,filename))
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
                res = result[0]
                combiner = [os.path.join(filter_path,filename), res]
                results.append(combiner)


    return(results)      


if __name__ == "__main__":
    a_index = 1
    g_index = 4
    b_index = 0
    w_index = 2
    m_index = 3
    alex_list = []
    gabbi_list = []
    bjarke_list = []
    wenche_list = []
    monica_list = []
    accuracy_manual = [0, 0]
    accuracy_per = [0, 0, 0, 0, 0]
    accuracy_count = [0, 0, 0, 0, 0]

    cnn_results = cnn_guesser("F:\\Dropbox\\Dropbox\\spai_dataset\\test")
    print(cnn_results)
    for key in cnn_results:
        filename = key[0]
        res = key[1]
        max_value = max(res)
        max_index = np.where(res==max_value)
        max_index = max_index[0][0]
        if filename[0] == "A":
            accuracy_count[a_index] += 1
        elif filename[0] == "G":
            accuracy_count[g_index] += 1
        elif filename[0] == "B":
            accuracy_count[b_index] += 1
        elif filename[0] == "M":
            accuracy_count[m_index] += 1
        elif filename[0] == "W":
            accuracy_count[w_index] += 1

        if max_index == a_index and filename[0] == "A":
            accuracy_manual[0] += 1
            accuracy_per[a_index] += 1
        elif max_index == g_index and filename[0] == "G":
            accuracy_manual[0] += 1
            accuracy_per[g_index] += 1
        elif max_index == w_index and filename[0] == "W":
            accuracy_manual[0] += 1
            accuracy_per[w_index] += 1
        elif max_index == b_index and filename[0] == "B":
            accuracy_manual[0] += 1
            accuracy_per[b_index] += 1
        elif max_index == m_index and filename[0] == "M":
            accuracy_manual[0] += 1
            accuracy_per[m_index] += 1
        else:
            accuracy_manual[1] += 1


        if max_index == a_index:
            alex_list.append(filename)
        elif max_index == g_index:
            gabbi_list.append(filename)
        elif max_index == w_index:
            wenche_list.append(filename)
        elif max_index == b_index:
            bjarke_list.append(filename)
        elif max_index == m_index:
            monica_list.append(filename)



    total = accuracy_manual[0]+accuracy_manual[1]
    alex_pic = [0, 0, 0, 0, 0]
    gabbi_pic = [0, 0, 0, 0, 0]
    wenche_pic = [0, 0, 0, 0, 0]
    bjarke_pic = [0, 0, 0, 0, 0]
    monica_pic = [0, 0, 0, 0, 0]

    for i in alex_list:
      if "A" in i:
        alex_pic[a_index] += 1
      if "B" in i:
        alex_pic[b_index] += 1
      if "W" in i:
        alex_pic[w_index] += 1
      if "G" in i:
        alex_pic[g_index] += 1
      if "M" in i:
        alex_pic[m_index] += 1
        
    for i in gabbi_list:
      if "A" in i:
        gabbi_pic[a_index] += 1
      if "B" in i:
        gabbi_pic[b_index] += 1
      if "W" in i:
        gabbi_pic[w_index] += 1
      if "G" in i:
        gabbi_pic[g_index] += 1
      if "M" in i:
        gabbi_pic[m_index] += 1
      
    for i in wenche_list:
      if "A" in i:
        wenche_pic[a_index] += 1
      if "B" in i:
        wenche_pic[b_index] += 1
      if "W" in i:
        wenche_pic[w_index] += 1
      if "G" in i:
        wenche_pic[g_index] += 1
      if "M" in i:
        wenche_pic[m_index] += 1
        
    for i in bjarke_list:
      if "A" in i:
        bjarke_pic[a_index] += 1
      if "B" in i:
        bjarke_pic[b_index] += 1
      if "W" in i:
        bjarke_pic[w_index] += 1
      if "G" in i:
        bjarke_pic[g_index] += 1
      if "M" in i:
        bjarke_pic[m_index] += 1

    for i in monica_list:
      if "A" in i:
        monica_pic[a_index] += 1
      if "B" in i:
        monica_pic[b_index] += 1
      if "W" in i:
        monica_pic[w_index] += 1
      if "G" in i:
        monica_pic[g_index] += 1
      if "M" in i:
        monica_pic[m_index] += 1


    print("Correct: {} Wrong: {}".format(accuracy_manual[0], accuracy_manual[1]))
    print("Accuracy:", accuracy_manual[0]/total, "\n")
    print("Alex: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[a_index], (accuracy_count[a_index]-accuracy_per[a_index]), (accuracy_per[a_index]/accuracy_count[a_index])))
    print("Gabbi: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[g_index], (accuracy_count[g_index]-accuracy_per[g_index]), (accuracy_per[g_index]/accuracy_count[g_index])))
    print("Bjarke: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[b_index], (accuracy_count[b_index]-accuracy_per[b_index]), (accuracy_per[b_index]/accuracy_count[b_index])))
    print("Monica: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[m_index], (accuracy_count[m_index]-accuracy_per[m_index]), (accuracy_per[m_index]/accuracy_count[m_index])))
    print("Wenche: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[w_index], (accuracy_count[w_index]-accuracy_per[w_index]), (accuracy_per[w_index]/accuracy_count[w_index])))
    print()
    print("Alex pic results: A:{} B:{} G:{} M:{} W:{}".format(alex_pic[a_index], alex_pic[b_index], alex_pic[g_index], alex_pic[m_index], alex_pic[w_index]))
    print("Gabbi pic results: A:{} B:{} G:{} M:{} W:{}".format(gabbi_pic[a_index], gabbi_pic[b_index], gabbi_pic[g_index], gabbi_pic[m_index], gabbi_pic[w_index]))
    print("Bjarke pic results: A:{} B:{} G:{} M:{} W:{}".format(bjarke_pic[a_index], bjarke_pic[b_index], bjarke_pic[g_index], bjarke_pic[m_index], bjarke_pic[w_index]))
    print("Monica pic results: A:{} B:{} G:{} M:{} W:{}".format(monica_pic[a_index], monica_pic[b_index], monica_pic[g_index], monica_pic[m_index], monica_pic[w_index]))
    print("Wenche pic results: A:{} B:{} G:{} M:{} W:{}".format(wenche_pic[a_index], wenche_pic[b_index], wenche_pic[g_index], wenche_pic[m_index], wenche_pic[w_index]))
