# Main script. Calls all module functions

# import settings
from kivy.config import Config

from maps.gui_windows.maps_ui import MapsApp

window_size = (1000, 750)

Config.set('graphics', 'width', str(window_size[0]))
Config.set('graphics', 'height', str(window_size[1]))
MapsApp().run()
