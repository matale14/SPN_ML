from glob import glob
from os.path import join, basename
from os import mkdir
from PIL import Image
from sys import platform
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout



"""
Created by: Bjarke Larsen

Note: I had to import glob, os, PIL and sys before all kivy modules, otherwise the photos didn't load. Haven't 
tested if it affects the creation of the thumbnails.
"""

class TestApp(App):

    def build(self):
        layout = BoxLayout(orientation="vertical", padding=0, spacing=0)
        #Specifies the path of the main folder, the first * in the "glob(join(curdir, '*', '*'))" is the subfolder
        # containing the images. An * is used to import from all subfolders, if there are several. The last * is the
        # image name, again * is used as a wild card.


        curdir = "/Users/Bjarke/Desktop/Test/"

        """Note for my self. Try implement the button and presentation in a FloatLayout, then it can be more smooth."""

        for folders in glob(join(curdir, "*")):
            try:
                name = str(folders)
                btn = Button(text=name, size=(100, 100), size_hint=(None, None))
                layout.add_widget(btn)
                presentation = GridLayout(cols=20,padding=0, spacing=10)

                for filename in glob(join(curdir, folders, "*")):
                    try:
                        # load the image
                        im = Image(source=filename)
                        presentation.add_widget(im)

                    except Exception:
                        print("Pictures: Unable to load <%s>" % filename)

                layout.add_widget(presentation)

            except:
                pass

        return layout

    def _create_thumbs(self):
        curdir = "/Users/Bjarke/Desktop/Test/"
        try:
            mkdir(curdir + "thump")
        except FileExistsError:
            pass
        size = 128, 128
        if platform == "darwin" or platform == "linux":
            for folder in glob(join(curdir, "*")):
                folder_name = basename(folder)
                if folder_name == "thump":
                    pass
                else:
                    thumbfolder_path = curdir + "thump/" + folder_name
                    try:
                        mkdir(thumbfolder_path)

                    except FileExistsError:
                        pass

                    try:
                        for picture in glob(join(curdir, folder, "*")):
                            picture_name = basename(picture)
                            im = Image.open(picture)
                            im.thumbnail(size)
                            im.save(thumbfolder_path + "/" + picture_name, "JPEG")
                    except:
                        pass
        else:
            for folder in glob(join(curdir, "*")):
                folder_name = basename(folder)
                if folder_name == "thump":
                    pass
                else:
                    thumbfolder_path = str(curdir + r"'thump\'" + folder_name)
                    try:
                        mkdir(thumbfolder_path)

                    except FileExistsError:
                        pass

                    try:
                        for picture in glob(join(curdir, folder, "*")):
                            picture_name = basename(picture)
                            im = Image.open(picture)
                            im.thumbnail(size)
                            im.save(thumbfolder_path + r"'\'" + picture_name, "JPEG")
                    except:
                        pass


if __name__ == "__main__":
    TestApp().run()