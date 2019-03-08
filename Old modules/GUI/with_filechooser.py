from kivy.app import App
from kivy.properties import ListProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Canvas
from kivy.uix.popup import *
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView



class RootWidget(FloatLayout):
    global fileChooser
    fileChooser = FileChooserIconView(size_hint_y=None)

    def __init__(self, **kwargs):

        super(RootWidget, self).__init__(**kwargs)

        sidepanel = BoxLayout(orientation="vertical", size_hint=(1, 1), pos_hint={"x":0.0, "y":0.0})
        self.add_widget(sidepanel)

        imp_photo = Button(text='Import Photos', size_hint=(0.2, 1))
        imp_photo.bind(on_press=self.pop)

        sidepanel.add_widget(imp_photo)


        sidepanel.add_widget(Button(text="Create Report", size_hint=(0.2, 1)))
        sidepanel.add_widget(Button(text="Save?", size_hint=(0.2, 1)))
        sidepanel.add_widget(Button(text="Button 4", size_hint=(0.2, 1)))

    def pop(self, obj):
        global fileChooser
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
        btn.bind(on_release=self._validate)
        btn.bind(on_release=popup.dismiss)
        btnlayout.add_widget(btn)

        btn = Button(text='Cancel')
        btn.bind(on_release=popup.dismiss)
        btnlayout.add_widget(btn)
        content.add_widget(btnlayout)

        # all done, open the popup !
        popup.open()

    def _validate(self, button):
        global fileChooser
        path = fileChooser.path

        print(path)
        return path

class TestApp(App):

    def build(self):
        rw=RootWidget()
        return RootWidget()

if __name__ == '__main__':
    TestApp().run()