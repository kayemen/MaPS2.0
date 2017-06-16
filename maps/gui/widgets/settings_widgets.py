from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, BooleanProperty


class SettingRow(BoxLayout):

    def get_setting_value(self):
        # Assuming each setting row has a text box as first child
        return self.children[0].text


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    dirselect = BooleanProperty(False)


class FileChooserWidget(BoxLayout):
    text = ObjectProperty(None)
    file_field = ObjectProperty(None)
    dirselect = BooleanProperty(False)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup, dirselect=self.dirselect)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        # with open(os.path.join(path, filename[0])) as stream:
        #     self.text_input.text = stream.read()
        # self.file_field.readonly = False
        self.file_field.text = filename[0]
        # self.file_field.readonly = True
        print 'loading path', path, 'file', filename
        self.dismiss_popup()
