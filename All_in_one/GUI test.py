from __future__ import print_function
from glob import glob
from os.path import join, basename
from os import mkdir, walk
from PIL import Image as pimage
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.factory import Factory as F
from image_class_test import Image as Filter
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import cv2


"""
Created by: Bjarke Larsen, Monica Romset & Alexander Mackenzie-Low
Version 3
"""


class SPAI(App):
    """
    The GUI as a class. Builds the interface, and handles all functions.
    """

    def build(self):
        """
        Build function creates the interface, and handles the widgets inside the interface.
        Returns:
            The root "window" to Kivy.App, that creates the window.
        """
        global scroll
        global layout
        global curdir
        global root
        global mp_queue

        #Creates global variables to be used by other functions.
        curdir = " "
        mp_queue = mp.Queue()

        # Creating the basic layouts.
        root = FloatLayout()
        scroll = ScrollView(pos_hint={"x": 0.12, "top": 0.92}, size_hint=(0.9, 1))
        layout = GridLayout(cols=5, padding=0, spacing=5)
        layout.bind(minimum_height=layout.setter("height"))

        # Create the ActionBar with buttons.
        actionbar = F.ActionBar(pos_hint={'top': 1})
        av = F.ActionView()
        av.add_widget(F.ActionPrevious(title='SPAI', with_previous=False))
        av.add_widget(F.ActionOverflow())
        av.add_widget(F.ActionButton(text='Import'.format(), on_press=self._pop))
        av.add_widget(F.ActionButton(text='Report'.format()))

        actionbar.add_widget(av)
        av.use_separator = False

        # Adding the layouts to the root layout
        root.add_widget(actionbar)
        root.add_widget(self._sidepanel())
        root.add_widget(scroll)
        scroll.add_widget(layout)

        return root


    def _update_scroll(self, path):
        """
        Function to update "showphotos" layout, when scrolling.
        Args:
            path: The path to the photos shown, in "showphotos" layout.
        """
        global scroll
        global layout

        #Removes the widgets in the scroll layout, if there is any.
        scroll.remove_widget(layout)
        #Loads the new updated layout, and updates the showphotos layout.
        layout = self._showphotos(path)
        scroll.add_widget(layout)
        layout.do_layout()


    def _sidepanel(self):
        """
        Function to create the sidepanel in the root layout. It reads all the folders from "curdir", and
        creates a button for each folder in "curdir". The sidepanel layout is then updated to show the buttons.
        Returns:
            Returs the sidepanel layout to the root layout.
        """
        global curdir
        global sidepanel_layout
        global root

        #Create the sidepanel layout.
        sidepanel_layout = BoxLayout(orientation="vertical", pos_hint={"x": 0.0, "top": 0.92}, size_hint=(0.1, 0.92))

        #If "curdir" contains folders, a button is created for each, and bind the button to update the
        # showphotos layout.
        if curdir == " ":
            return sidepanel_layout
        else:
            root.remove_widget(sidepanel_layout)
            for folders in sorted(glob(join(curdir, "thumb", "*"))):
                name = basename(folders)
                btn = Button(text=name, on_press=lambda n=name: self._update_scroll(n.text))
                sidepanel_layout.add_widget(btn)
            root.add_widget(sidepanel_layout)
            sidepanel_layout.do_layout()


    def _validate(self, fileChooser):
        """
        Function to add the path chosen by user to "curdir" and initiate functions that needs to be run.

        Args:
            fileChooser: Takes the path chosen by the user.

        Returns:
            None, but initiates several other functions.

        """
        global curdir
        global mp_queue

        mp.freeze_support()
        curdir = fileChooser.path

        #Initiates functions.
        self._queue_photos()
        self._multiprocessing(self._handle_photos, mp_queue)
        self._sidepanel()


    def _pop(self, obj):
        """
        Function that creates a pop-up window, where the user choses the path of the pictures to be imported.
        Args:
            obj: Is needed by the FileChooser class.

        Returns:
            A string containing the path chosen by the user.
        """

        # Creates the layouts.
        fileChooser = FileChooserIconView(size_hint_y=None)
        content = BoxLayout(orientation='vertical', spacing=7)
        scrollView = ScrollView()

        # Binds the chosen path to the "validate" function.
        fileChooser.bind(on_submit=self._validate)
        fileChooser.height = 500

        scrollView.add_widget(fileChooser)

        # Adding the layouts together.
        content.add_widget(Widget(size_hint_y=None, height=5))
        content.add_widget(scrollView)
        content.add_widget(Widget(size_hint_y=None, height=5))

        popup = Popup(title='Choose Directory',
                      content=content,
                      size_hint=(0.6, 0.6))

        # Creates two buttons to sumbit or cancel.
        btnlayout = BoxLayout(size_hint_y=None, height=50, spacing=5)
        btn = Button(text='Ok')
        btn.bind(on_release=lambda x: self._validate(fileChooser))
        btn.bind(on_release=popup.dismiss)
        btnlayout.add_widget(btn)

        btn = Button(text='Cancel')
        btn.bind(on_release=popup.dismiss)
        btnlayout.add_widget(btn)
        content.add_widget(btnlayout)

        popup.open()


    def _showphotos(self, btn):
        """
        Function to load photos and show them in the layout.
        Args:
            btn: String, name of the folder, containing the photos to be shown.

        Returns:
            A GridLayout containing the pictures in the path provided.
        """
        global layout
        global curdir

        # Create the layouts.
        layout = GridLayout(cols=5, padding=0, spacing=0, size_hint=(1, None))
        layout.bind(minimum_height=layout.setter("height"))

        foldername = btn

        # Args is combined with "curdir" to load the thumbnails, and add them to the Gridlayout.
        if foldername == "":
            pass
        else:
            for filename in sorted(glob(join(curdir, "thumb", foldername, "*"))):
                try:
                    canvas = BoxLayout(size_hint=(1, None))
                    im = Image(source=filename)
                    canvas.add_widget(im)
                    layout.add_widget(canvas)

                except Exception:
                    print("Pictures: Unable to load <%s>" % filename)

        return layout


    def _queue_photos(self):
        """
        Function to add photos to the queue of the multiprocessing function.

        Returns:
            Adds a list containing strings to the queue. Strings of paths to the picture and folder, and
            name of the picture.
        """
        global curdir
        global mp_queue

        for root, dirs, files in walk(curdir):
            for file in files:
                if dirs == "thumb" or dirs == "filtered":
                    pass
                else:
                    file_path = join(root, file)
                    data = [file_path, curdir]
                    mp_queue.put(data)
                    print("Queued:", file_path)

        try:
            mkdir(join(curdir, "thumb"))
        except FileExistsError:
            pass
        try:
            mkdir(join(curdir, "thumb", "Alexander"))
        except FileExistsError:
            pass
        try:
            mkdir(join(curdir, "thumb", "Bjarke"))
        except FileExistsError:
            pass
        try:
            mkdir(join(curdir, "thumb", "Gabrielle"))
        except FileExistsError:
            pass
        try:
            mkdir(join(curdir, "thumb", "Monica"))
        except FileExistsError:
            pass
        try:
            mkdir(join(curdir, "thumb", "Wenche"))
        except FileExistsError:
            pass
        try:
            mkdir(join(curdir, "filtered"))
        except FileExistsError:
            pass

    @staticmethod
    def _handle_photos(queue):
        """
        Handles all actions of each picture. Creating a thumbnail, and starts the filtering of each picture.
        Args:
            queue: Multiprocessing.queue is given, containing a list of strings, with the path to
            the picture, the folder and name of the picture.

        Returns:
            Saves a thumbnail and the filtered picture in separate folders.
        """

        # Retrieves one list from the queue and splits the list.
        data = queue.get()
        picture = data[0]
        curdir = data[1]

        picture_name = basename(picture)

        # Filters the image.
        Filter(picture, join(curdir, "filtered"))

        # CNN
        try:
            f = join(curdir, "filtered", picture_name)
            image_size = 256
            num_channels = 3

            ## Let us restore the saved model
            with tf.Session(graph=tf.Graph()) as sess:
                # Step-1: Recreate the network graph. At this step only graph is created.
                saver = tf.train.import_meta_graph('model/spai_model.meta')
                # Step-2: Now let's load the weights saved using the restore method.
                # saver.restore(sess, tf.train.latest_checkpoint('./'))
                tf.global_variables_initializer().run()
                # Reading the image using OpenCV
                image = cv2.imread(f)
                if image is not None:
                    images = []
                    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
                    images.append(image)
                    images = np.array(images, dtype=np.uint8)
                    images = images.astype('float32')
                    images = np.multiply(images, 1.0 / 255.0)
                    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                    x_batch = images.reshape(1, image_size, image_size, num_channels)
                    graph = tf.get_default_graph()
                    x = graph.get_tensor_by_name = "x:0"
                    y_pred = graph.get_tensor_by_name = "y_pred:0"

                    result = sess.run(y_pred, feed_dict={x: x_batch})
                    res = result[0].tolist()
                    # max_value = max(res)
                    # max_index = np.where(res==max_value)
                    # max_index = max_index[0][0]
        except:
            print("Error on CNN")
            pass


        # Saves a thumb of the picture in a folder depending on the values from the CNN
        try:
            size_thumb = 128, 128
            thumb = pimage.open(picture)
            thumb.thumbnail(size_thumb)

            values = (res)
            highest_value = 0
            print("Values for picture", picture)
            for x in range(len(values)):
                print("Current highest value:", highest_value)
                if values[x] > highest_value:
                    highest_value = values[x]
                    print("Changing highest value to:", highest_value)
            group = values.index(highest_value)
            if group == 1:
                print("Pictures belongs to Alexander")
                thumb.save(join(curdir, "thumb", "Alexander", picture_name), "JPEG")
            elif group == 0:
                print("Pictures belongs to Bjarke")
                thumb.save(join(curdir, "thumb", "Bjarke", picture_name), "JPEG")
            elif group == 4:
                print("Pictures belongs to Gabrielle")
                thumb.save(join(curdir, "thumb", "Gabrielle", picture_name), "JPEG")
            elif group == 3:
                print("Pictures belongs to Monica")
                thumb.save(join(curdir, "thumb", "Monica", picture_name), "JPEG")
            elif group == 2:
                print("Pictures belongs to Wenche")
                thumb.save(join(curdir, "thumb", "Wenche", picture_name), "JPEG")
        except:
            print("Error on sorting image")
            pass


    def _multiprocessing(self, function, queue):
        """
        Function to use multiprocessing for creating thumbnails and filter images.
        Args:
            function: The current function to run on each processor.
            queue: multiprossessing.queue format queue, to get data to process from.

        Returns:
            Runs a function on each processor/thread of the cpu, to speed up processing time.
        """
        # Counts number of threads on the processor.
        cpu_count = mp.cpu_count()
        print("CPU-count", cpu_count)
        # Initiate a process on each thread.
        try:
            for i in range(cpu_count):
                mp.Process(target=function, args=(queue,)).start()
        except EOFError:
            pass

        # When queue is empty, "STOP" is sent to the queue to stop all threads, and release them.
        #try:
         #   for i in range(cpu_count):
          #      queue.put("STOP")
        #except EOFError:
         #   pass


if __name__ == "__main__":
    SPAI().run()