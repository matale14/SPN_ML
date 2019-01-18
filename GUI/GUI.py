from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from glob import glob
from os.path import join
"""
Created by: Bjarke Larsen
"""

class TestApp(App):

    def build(self):
        layout = GridLayout(cols=3)
        #Specifies the path of the main folder, the first * in the "glob(join(curdir, '*', '*'))" is the subfolder
        # containing the images. An * is used to import from all subfolders, if there are several. The last * is the
        # image name, again * is used as a wild card.
        curdir = "/Users/Bjarke/Desktop/Test/"
        for filename in glob(join(curdir, '*', '*')):
            try:
                # load the image
                im = Image(source=filename)
                layout.add_widget(im)

            except Exception:
                print("Pictures: Unable to load <%s>" % filename)


        return layout

if __name__ == "__main__":
    TestApp().run()