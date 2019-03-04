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
from SPAI.FILTER.filter_cy import filter_main

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
        global root

        curdir = " "
        # Creating the layout
        root = FloatLayout()
        scroll = ScrollView(pos_hint={"x": 0.12, "top": 0.92}, size_hint=(0.9, 1))
        layout = GridLayout(cols=5, padding=0, spacing=5)
        layout.bind(minimum_height=layout.setter("height"))

        actionbar = F.ActionBar(pos_hint={'top': 1})
        av = F.ActionView()
        av.add_widget(F.ActionPrevious(title='SPAI', with_previous=False))
        av.add_widget(F.ActionOverflow())
        av.add_widget(F.ActionButton(text='Import'.format(), on_press=self._pop))
        av.add_widget(F.ActionButton(text='Report'.format()))
        av.add_widget(F.ActionButton(text='Save'.format()))
        av.add_widget(F.ActionButton(text='Whatever'.format()))

        actionbar.add_widget(av)
        av.use_separator = False
        root.add_widget(actionbar)

        # Adding the layouts together
        root.add_widget(self._sidepanel())
        root.add_widget(scroll)
        scroll.add_widget(layout)

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
        global sidepanel_layout
        global root
        sidepanel_layout = BoxLayout(orientation="vertical", pos_hint={"x": 0.0, "top": 0.92}, size_hint=(0.1, 0.92))
        if curdir == " ":
            return sidepanel_layout
        else:
            root.remove_widget(sidepanel_layout)
            for folders in glob(join(curdir, "*")):
                name = basename(folders)
                btn = Button(text=name, on_press=lambda n=name: self._update_scroll(n.text))
                sidepanel_layout.add_widget(btn)
            root.add_widget(sidepanel_layout)
            sidepanel_layout.do_layout()

    def _validate(self, fileChooser):
        global curdir
        curdir = fileChooser.path
        for folders in glob(join(curdir, "*")):
            if folders == "thumb":
                pass
            else:
                filter_main(folders, 16, 16)
        self._create_thumbs()
        self._sidepanel()

    def _pop(self, obj):

        fileChooser = FileChooserIconView(size_hint_y=None)
        content = BoxLayout(orientation='vertical', spacing=7)

        # first, create the scrollView
        scrollView = ScrollView()

        # then, create the fileChooser and integrate it in the scrollView

        fileChooser.bind(on_submit=self._validate)
        fileChooser.height = 500  # this is a bit ugly...

        scrollView.add_widget(fileChooser)

        # construct the content, widget are used as a spacer
        content.add_widget(Widget(size_hint_y=None, height=5))
        content.add_widget(scrollView)
        content.add_widget(Widget(size_hint_y=None, height=5))

        popup = Popup(title='Choose Directory',
                      content=content,
                      size_hint=(0.6, 0.6))

        # 2 buttons are created for accept or cancel the current value
        btnlayout = BoxLayout(size_hint_y=None, height=50, spacing=5)
        btn = Button(text='Ok')
        btn.bind(on_release=lambda x: self._validate(fileChooser))
        btn.bind(on_release=popup.dismiss)
        btnlayout.add_widget(btn)

        btn = Button(text='Cancel')
        btn.bind(on_release=popup.dismiss)
        btnlayout.add_widget(btn)
        content.add_widget(btnlayout)

        # all done, open the popup !
        popup.open()

    def _showphotos(self, btn):
        global layout
        global curdir
        layout = GridLayout(cols=5, padding=0, spacing=0, size_hint=(1, None))
        layout.bind(minimum_height=layout.setter("height"))

        foldername = btn

        if foldername == "":
            pass
        else:
            # Specifies the path of the main folder, the first * in the "glob(join(curdir, '*', '*'))" is the subfolder
            # containing the images. An * is used to import from all subfolders, if there are several. The last * is the
            # image name, again * is used as a wild card.

            for filename in glob(join(curdir, foldername, "thumb", "*")):
                try:
                    canvas = BoxLayout(size_hint=(1, None))
                    im = Image(source=filename)
                    canvas.add_widget(im)
                    layout.add_widget(canvas)

                except Exception:
                    print("Pictures: Unable to load <%s>" % filename)

        return layout

    def _create_thumbs(self):
        global curdir

        for folder in glob(join(curdir, "*")):
            try:
                mkdir(join(folder + "/thumb/"))
                for picture in glob(join(curdir, folder, "*")):
                    picture_name = basename(picture)
                    if picture_name == "thumb" or picture_name == "filtered":
                        pass
                    else:
                        size = 128, 128
                        im = pimage.open(picture)
                        im.thumbnail(size)
                        im.save(join(folder + "/thumb/" + picture_name), "JPEG")

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
                        size = 128, 128
                        im = pimage.open(picture)
                        im.thumbnail(size)
                        im.save(join(folder + "/thumb/" + picture_name), "JPEG")


if __name__ == "__main__":
    SPAI().run()