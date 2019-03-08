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


            if max_index == 2:
                alex_list.append(filename)
            elif max_index == 3:
                gabbi_list.append(filename)
            elif max_index == 4:
                wenche_list.append(filename)
            elif max_index == 0:
                bjarke_list.append(filename)
            elif max_index == 1:
                monica_list.append(filename)

    con= tf.confusion_matrix(np.array(test_labels), np.array(results), num_classes=5)
    con2 = sess.run(con)
    success = con2[0][0]+con2[1][1]
    accuracy = success/np.sum(con2)