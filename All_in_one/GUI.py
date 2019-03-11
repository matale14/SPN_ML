from glob import glob
from os.path import join, basename
from os import mkdir
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
from image_class import Image as Filter
import multiprocessing as mp


"""
Created by: Bjarke Larsen, Monica Romset & Alexander Mackenzie-Low
Version 2
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
            for folders in sorted(glob(join(curdir, "*"))):
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
        global filter_queue

        curdir = fileChooser.path

        #Initiates functions.
        self._queue_photos()
        mp.freeze_support()
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
            for filename in glob(join(curdir, foldername, "thumb", "*")):
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

        # Creates a Thumb folder inside the folder containing the photos to be added.
        for folder in glob(join(curdir, "*")):
            try:
                mkdir(join(folder + "/thumb/"))
                for picture in glob(join(curdir, folder, "*")):
                    picture_name = basename(picture)
                    if picture_name == "thumb" or picture_name == "filtered":
                        pass
                    else:
                        mp_queue.put([picture, folder, picture_name])
            # If Thumb folder already exists, it checks if there is a thumbnail of all pictures. If not the
            # missing is added to the queue.
            except FileExistsError:
                thumb_pictures = []
                for thumb in glob(join(curdir, folder + "/thumb/", "*")):
                    thumb_pictures.append(basename(thumb))

                for picture in glob(join(curdir, folder, "*")):
                    picture_name = basename(picture)
                    if picture_name == "thumb" or picture_name == "filtered":
                        pass

                    elif picture_name in thumb_pictures:
                        pass

                    else:
                        mp_queue.put([picture, folder, picture_name])


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
        while True:
            try:
                # Retrieves one list from the queue and splits the list.
                data = queue.get()
                picture = data[0]
                folder = data[1]
                picture_name = data[2]

                size_thumb = 128, 128

                # Creates a thumbnail of the picture and saves it.
                im = pimage.open(picture)
                im.thumbnail(size_thumb)
                im.save(join(folder + "/thumb/" + picture_name), "JPEG")

                # Filters the image.
                Filter(picture, size)

            except:
                break


    def _multiprocessing(self, function, queue):
        """
        Function to use multiprocessing for creating thumbnails and filter images.
        Args:
            function: The current function to run on each processor.
            queue: multiprossessign.queue format queue, to get data to process from.

        Returns:
            Runs a function on each processor/thread of the cpu, to speed up processing time.
        """
        # Counts number of threads on the processor.
        cpu_count = mp.cpu_count()

        # Initiate a process on each thread.
        try:
            for i in range(cpu_count):
                mp.Process(target=function, args=(queue,)).start()
        except EOFError:
            pass

        # When queue is empty, "STOP" is sent to the queue to stop all threads, and release them.
        try:
            for i in range(cpu_count):
                queue.put("STOP")
        except EOFError:
            pass


if __name__ == "__main__":
    SPAI().run()