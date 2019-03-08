from kivy.base import runTouchApp
from kivy.factory import Factory as F

actionbar = F.ActionBar(pos_hint={'top': 1})

av = F.ActionView()
av.add_widget(F.ActionPrevious(title='Menu', with_previous=False))
av.add_widget(F.ActionOverflow())

av.add_widget(F.ActionButton(text='Import'.format()))
av.add_widget(F.ActionButton(text='Report'.format()))
av.add_widget(F.ActionButton(text='Save'.format()))
av.add_widget(F.ActionButton(text='Whatever'.format()))


actionbar.add_widget(av)

av.use_separator = False

runTouchApp(actionbar)