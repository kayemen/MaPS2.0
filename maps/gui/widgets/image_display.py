from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty, ObjectProperty, BooleanProperty

from maps.settings import setting

import glob
import os
import cv2

from maps.helpers.gui_modules import create_image_overlay, create_blank_image, load_image_sequence


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
    # plot = ObjectProperty()

    def __init__(self, **kwargs):
        super(PlotDisplay, self).__init__(**kwargs)
        # self.plot = get_plot()

    def update(self, dt):
        pass
        # from maps.core.z_stamping import get_plot
        # frame = get_plot()
        # frame = cv2.flip(frame, 0)
        # print frame.shape
        # buf = frame.tostring()
        # # print buf
        # image_texture = Texture.create(
        #     size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        # image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        # self.texture = image_texture
        # print 'updateing'
        # TODO: Draw rectangle from self.parent.selected_region here
