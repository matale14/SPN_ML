from glob import glob
from os.path import join, basename
from os import mkdir
from PIL import Image as pimage
from sys import platform
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView



"""
Created by: Bjarke Larsen

Note: I had to import glob, os, PIL and sys before all kivy modules, otherwise the photos didn't load. Haven't 
tested if it affects the creation of the thumbnails.
"""

class SPAI(App):
    def build(self):
        #Creating the layout
        root = FloatLayout()
        scroll = ScrollView(pos_hint={"x": 0.2, "y": 0.0})

        #Adding the layouts together
        root.add_widget(self._sidepanel())
        root.add_widget(scroll)
        scroll.add_widget(self._showphotos())

        return root


    def _sidepanel(self):
        layout = BoxLayout(orientation="vertical", pos_hint={"x": 0.0, "y": 0.0})
        layout.add_widget(Button(text="Import Photos", size_hint=(0.2,1)))
        layout.add_widget(Button(text="Create Report", size_hint=(0.2,1)))
        layout.add_widget(Button(text="Save?", size_hint=(0.2,1)))
        layout.add_widget(Button(text="Button 4", size_hint=(0.2,1)))
        return layout

    def _showphotos(self):
        #self._create_thumbs()
        layout = BoxLayout(orientation="vertical", padding=20, spacing=30, size_hint_y=None)
        layout.bind(minimum_height=layout.setter("height"))
        #Specifies the path of the main folder, the first * in the "glob(join(curdir, '*', '*'))" is the subfolder
        # containing the images. An * is used to import from all subfolders, if there are several. The last * is the
        # image name, again * is used as a wild card.

        curdir = "/Users/Bjarke/Desktop/Test/"

        for folders in glob(join(curdir, "*")):
            try:
                name = basename(folders)
                btn = Button(text=name)
                layout.add_widget(btn)
                presentation = GridLayout(cols=20, padding=0, spacing=10, size_hint_y=None)

                for filename in glob(join(curdir, folders, "thumb", "*")):
                    try:
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
        if platform == "darwin" or platform == "linux":
            for folder in glob(join(curdir, "*")):
                try:
                    mkdir(folder + "/thumb/")
                except FileExistsError:
                    print("Thumb folder already exists")
                    pass

                for picture in glob(join(curdir, folder, "*")):
                    picture_name = basename(picture)
                    if picture_name == "thumb":
                        pass
                    else:
                        try:
                            size = 128, 128
                            im = pimage.open(picture)
                            im.thumbnail(size)
                            im.save(folder + "/thumb/" + picture_name, "JPEG")
                        except FileExistsError:
                            print("Pictures already exists")
                            pass
        else:
            for folder in glob(join(curdir, "*")):
                try:
                    mkdir(folder + r"'\thumb\'")
                except FileExistsError:
                    print("Thumb folder already exists")
                    pass

                for picture in glob(join(curdir, folder, "*")):
                    picture_name = basename(picture)
                    if picture_name == "thumb":
                        pass
                    else:
                        try:
                            size = 128, 128
                            im = pimage.open(picture)
                            im.thumbnail(size)
                            im.save(folder + r"'\thumb\'" + picture_name, "JPEG")
                        except FileExistsError:
                            print("Pictures already exists")
                            pass


if __name__ == "__main__":
    SPAI().run()