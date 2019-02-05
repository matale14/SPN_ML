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
        global scroll
        global layout
        global curdir
        curdir = "/Users/Bjarke/Desktop/Test/"
        #Creating the layout
        root = FloatLayout()
        scroll = ScrollView(pos_hint={"x": 0.12, "top": 1.0}, size_hint=(0.9,1))
        layout = GridLayout(cols=5, padding=0, spacing=5)
        layout.bind(minimum_height=layout.setter("height"))

        #Adding the layouts together
        root.add_widget(self._sidepanel())
        root.add_widget(scroll)
        scroll.add_widget(layout)
        #self._create_thumbs()
        return root

    def _update_scroll(self, path):
        global scroll
        global layout
        scroll.remove_widget(layout)
        layout = self._showphotos(path)
        scroll.add_widget(layout)
        layout.do_layout()



    def _sidepanel(self):
        global curdir
        layout = BoxLayout(orientation="vertical", pos_hint={"x": 0.0, "top": 1.0}, size_hint=(0.1,1))

        for folders in glob(join(curdir, "*")):
            name = basename(folders)
            btn = Button(text=name, on_press=lambda n=name:self._update_scroll(n.text))
            layout.add_widget(btn)

        return layout


    def _showphotos(self, btn):
        global layout
        global curdir
        layout = GridLayout(cols=5, padding=0, spacing=5, size_hint=(None, None), width=600)
        layout.bind(minimum_height=layout.setter("height"))

        foldername = btn

        if foldername == "":
            pass
        else:
            #Specifies the path of the main folder, the first * in the "glob(join(curdir, '*', '*'))" is the subfolder
            # containing the images. An * is used to import from all subfolders, if there are several. The last * is the
            # image name, again * is used as a wild card.

            for filename in glob(join(curdir, foldername, "thumb", "*")):
                try:
                    canvas = BoxLayout(size_hint_y=None)
                    im = Image(source=filename)
                    canvas.add_widget(im)
                    layout.add_widget(canvas)

                except Exception:
                    print("Pictures: Unable to load <%s>" % filename)

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