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
from cnn_guesser import cnn_guesser
from report import createMultiPage
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import cv2
import os



"""
Created by: Bjarke Larsen, Alexander Mackenzie-Low & Monica Romset
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
        av.add_widget(F.ActionButton(text='Check class'.format(), on_press=self._cnn))
        av.add_widget(F.ActionButton(text='Report'.format(), on_press=self._report))

        actionbar.add_widget(av)
        av.use_separator = False

        # Adding the layouts to the root layout
        root.add_widget(actionbar)
        root.add_widget(self._sidepanel())
        root.add_widget(scroll)
        scroll.add_widget(layout)

        return root

    def _report(self, obj):
        """
        Function to create input values to the report.
        Args:
            obj: passes from the button to activate this function. Not used.

        Returns:
            Creates a report using the imported "createMultiPage" report funcion.
        """
        reportname = "Report test"
        casenmbr = "001"
        createdby = "Gang of Five"
        comparison= "Gang of Five"
        createMultiPage(reportname, casenmbr, createdby, comparison)


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
        Function to create the sidepanel in the root layout. It reads all the folders from "curdir/thumb", and
        creates a button for each folder in "curdir/thumb". The sidepanel layout is then updated to show the buttons.
        Returns:
            Returs the sidepanel layout to the root layout.
        """
        global curdir
        global sidepanel_layout
        global root
        #Create the sidepanel layout.
        sidepanel_layout = BoxLayout(orientation="vertical", pos_hint={"x": 0.0, "top": 0.92}, size_hint=(0.1, 0.92))

        #If "curdir/thumb" contains folders, a button is created for each, and bind the button to update the
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
        Function to add the path chosen by user as "curdir" and initiate the functions to queue and filter the photos.

        Args:
            fileChooser: Takes the path chosen by the user.

        Returns:
            None, but initiates several other functions.
        """
        global curdir
        global mp_queue
        curdir = fileChooser.path

        #Initiates functions.
        self._queue_photos()
        mp.freeze_support()
        self._multiprocessing(self._handle_photos, mp_queue)
        self._sidepanel()

    def _pop(self, obj):
        """
        Function that creates a pop-up window, where the user chooses the path of the pictures to be imported.
        Args:
            obj: Is passed by the button activating it, not used.

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
        Creates all working folders, as a folder for thumbnails. In that folder a folder for each class of pictures,
        that the CNN is guessing between. And a folder to store the filtered images.
        Returns:
            Adds a list containing strings to the queue. Strings of: paths of picture and curdir.
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
        Retrieves each image from a multiprocessing queue and starts the filtering process of each image.
        Args:
            queue: Multiprocessing.queue is given, containing a list of strings, with the path to
            the picture, the folder and name of the picture.

        Returns:
            Saves a thumbnail and the filtered picture in the previously created "filtered" folder.
        """
        while True:
            # Retrieves one list from the queue and splits the list.
            data = queue.get()
            picture = data[0]
            curdir = data[1]

            # Filters the image.
            Filter(picture, join(curdir, "filtered"))


    def _cnn(self, obj):
        global curdir
        cnn_list = cnn_guesser(curdir)
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

        cnn_results = cnn_list
        for key in cnn_results:
            filepath = key[0]
            filename = os.path.basename(filepath)
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

            try:
                size_thumb = 128, 128
                thumb = pimage.open(filepath)
                thumb.thumbnail(size_thumb)


                if max_index == a_index:
                    print("Pictures belongs to Alexander")
                    thumb.save(join(curdir, "thumb", "Alexander", filename), "JPEG")
                    alex_list.append(filename)
                elif max_index == b_index:
                    print("Pictures belongs to Bjarke")
                    thumb.save(join(curdir, "thumb", "Bjarke", filename), "JPEG")
                    bjarke_list.append(filename)
                elif max_index == g_index:
                    print("Pictures belongs to Gabrielle")
                    thumb.save(join(curdir, "thumb", "Gabrielle", filename), "JPEG")
                    gabbi_list.append(filename)
                elif max_index == m_index:
                    print("Pictures belongs to Monica")
                    thumb.save(join(curdir, "thumb", "Monica", filename), "JPEG")
                    monica_list.append(filename)
                elif max_index == w_index:
                    print("Pictures belongs to Wenche")
                    thumb.save(join(curdir, "thumb", "Wenche", filename), "JPEG")
                    wenche_list.append(filename)
            except Exception as e:
                print("Error on sorting image:", e)
                pass

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

        total = accuracy_manual[0]+accuracy_manual[1]        
        print("Correct: {} Wrong: {}".format(accuracy_manual[0], accuracy_manual[1]))
        print("Accuracy:", accuracy_manual[0]/total, "\n")
        try:
            print("Alex: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[a_index], (accuracy_count[a_index]-accuracy_per[a_index]), (accuracy_per[a_index]/accuracy_count[a_index])))
            print("Gabbi: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[g_index], (accuracy_count[g_index]-accuracy_per[g_index]), (accuracy_per[g_index]/accuracy_count[g_index])))
            print("Bjarke: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[b_index], (accuracy_count[b_index]-accuracy_per[b_index]), (accuracy_per[b_index]/accuracy_count[b_index])))
            print("Monica: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[m_index], (accuracy_count[m_index]-accuracy_per[m_index]), (accuracy_per[m_index]/accuracy_count[m_index])))
            print("Wenche: Correct: {} Wrong: {} Accuracy: {}".format(accuracy_per[w_index], (accuracy_count[w_index]-accuracy_per[w_index]), (accuracy_per[w_index]/accuracy_count[w_index])))
        except ZeroDivisionError:
            pass
        print()
        print("Alex pic results: A:{} B:{} G:{} M:{} W:{}".format(alex_pic[a_index], alex_pic[b_index], alex_pic[g_index], alex_pic[m_index], alex_pic[w_index]))
        print("Gabbi pic results: A:{} B:{} G:{} M:{} W:{}".format(gabbi_pic[a_index], gabbi_pic[b_index], gabbi_pic[g_index], gabbi_pic[m_index], gabbi_pic[w_index]))
        print("Bjarke pic results: A:{} B:{} G:{} M:{} W:{}".format(bjarke_pic[a_index], bjarke_pic[b_index], bjarke_pic[g_index], bjarke_pic[m_index], bjarke_pic[w_index]))
        print("Monica pic results: A:{} B:{} G:{} M:{} W:{}".format(monica_pic[a_index], monica_pic[b_index], monica_pic[g_index], monica_pic[m_index], monica_pic[w_index]))
        print("Wenche pic results: A:{} B:{} G:{} M:{} W:{}".format(wenche_pic[a_index], wenche_pic[b_index], wenche_pic[g_index], wenche_pic[m_index], wenche_pic[w_index]))


    def _multiprocessing(self, function, queue):
        """
        Function to use multiprocessing for any given function.
        Args:
            function: The current function to run on each processor.
            queue: multiprossessing.queue format queue, to get data to process in the function.

        Returns:
            Depends on the function executed.
        """
        # Counts number of threads on the processor.
        cpu_count = mp.cpu_count()
        print("CPU-count", cpu_count)
        # Initiate a process on each thread.
        try:
            for i in range(cpu_count-1):
                mp.Process(target=function, args=(queue,)).start()
        except EOFError:
            pass


if __name__ == "__main__":
    SPAI().run()