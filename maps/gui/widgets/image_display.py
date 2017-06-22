from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty,\
    NumericProperty,\
    ObjectProperty,\
    BooleanProperty
from kivy.garden.matplotlib.backend_kivyagg import NavigationToolbar2Kivy

from maps.settings import setting

import glob
import os
import cv2
import matplotlib.pyplot as plt

from maps.helpers.gui_modules import create_image_overlay,\
    create_blank_image,\
    load_image_sequence


class FrameSelectionWidget(BoxLayout):
    frame = ObjectProperty()
    frame_selector = ObjectProperty()
    slider = ObjectProperty()
    frame_start = NumericProperty()
    frame_count = NumericProperty()
    frame_pathlist = StringProperty()
    frame_select = NumericProperty()
    overlay = BooleanProperty(False)
    display_blank = BooleanProperty(False)

    def __init__(self, frame_start=1, overlay_type='rectangle', **kwargs):
        super(FrameSelectionWidget, self).__init__(**kwargs)
        self.img_paths = []
        self.img_seq = []
        self.frame_start = frame_start
        self.frame_count = len(self.img_paths)
        self.overlay_type = overlay_type
        self.overlay_data = {}

    def update(self, *args):
        if self.display_blank:
            self.frame_select = 0
            frame = create_blank_image(400, 600)
        else:
            self.frame_select = int(
                self.frame_selector.min + self.frame_selector.value
            )
            frame = create_image_overlay(self.img_seq[self.frame_select])
        frame = cv2.flip(frame, 0)
        buf = frame.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.frame.texture = image_texture

    def load_frames(self, img_paths, frame_index_start=1):
        if img_paths is not None:
            self.img_seq = load_image_sequence(img_paths)
        else:
            self.img_seq = []
        self.frame_count = len(self.img_seq)
        self.frame_start = frame_index_start

    # def load_blank(self):
    #     pass


class PlotDisplay(BoxLayout):
    plot_area = ObjectProperty()
    plot_selection = ObjectProperty()

    figure = ObjectProperty()
    axes = ObjectProperty()

    def __init__(self, **kwargs):
        super(PlotDisplay, self).__init__(**kwargs)
        self.figure, self.axes = plt.subplots()
        print self.children

    def initialize_plots(self):
        # nav_buttons = NavigationToolbar2Kivy(self.figure.canvas)
        # self.plot_area.add_widget(nav_buttons.actionbar)
        self.plot_area.add_widget(self.figure.canvas)
        self.plotting_methods = {}
        self.plot_selection.values = []

        def show_plot(spinner, text):
            print text

        self.plot_selection.bind(text=show_plot)
        self.plot_selection.bind(text=self.update_plot)

    def update_plot(self, spinner_instance, method_name):
        self.axes.clear()

        if method_name == 'Select plot':
            pass
        elif method_name not in self.plotting_methods.keys():
            print 'Unknown plot type'
        elif self.plotting_methods[method_name]['data'] is None:
            print 'No data to plot'
        else:
            method = self.plotting_methods[method_name]['method']
            data = self.plotting_methods[method_name]['data']
            method(self.axes, data)

        self.figure.canvas.draw()

    def add_plotting_method(self, plot_name, plot_method):
        new_plot = {'method': plot_method, 'data': None}
        self.plotting_methods[plot_name] = new_plot
        self.plot_selection.values.append(plot_name)

    def update_plot_data(self, plot_name, plot_data=None):
        self.plotting_methods[plot_name]['data'] = plot_data

    def clear_plots(self):
        self.plotting_methods = {}
        self.plot_selection.values = []
        # for _, obj in self.plotting_methods.iteritems():
        #     obj['data'] = None
