from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.factory import Factory

import json


class InterfaceManager(BoxLayout):
    screen_area = ObjectProperty()

    def __init__(self, **kwargs):
        super(InterfaceManager, self).__init__(**kwargs)
        self.InputWidget = InputWidget()
    #     self.add_widget(self.InputWidget)

    def load_previous_screen(self):
        # print self.ids
        # print type(self.screen_area)
        # self.ids.l_button.text = 'changed'
        self.screen_area.clear_widgets()

    def load_next_screen(self):
        self.screen_area.add_widget(self.InputWidget)


class InputWidget(BoxLayout):
    parameters = ObjectProperty(json.load(open('../default_inputs.json')))
#
#     def generate_(self, arg):
#         pass


class MapsApp(App):

    def build(self):
        # paramters = json.load(open('../default_inputs.json'))

        # root_widget = BoxLayout(orientation='vertical')
        w = InterfaceManager()
        return w


if __name__ == '__main__':
    InputApp().run()
