from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty

from maps.settings import setting

import glob
import os
import cv2

from maps.helpers.gui_modules import create_image_overlay, load_image_sequence


class FrameDisplay(Image):
    path = StringProperty('')
    frame_count = NumericProperty(50)

    def __init__(self, **kwargs):
        super(FrameDisplay, self).__init__(**kwargs)

    def load_frames(self):
        print setting
        self.path = setting['bf_path']
        self.frame_count = setting['fphb']
        self.path_list = glob.glob(os.path.join(self.path, '*.tif'))[:self.frame_count]
        self.selection_window = [0, 0]
        self.img_seq = load_image_sequence(self.path_list)

    def update(self, dt):
        frame = create_image_overlay(self.img_seq[int(self.parent.parent.img_select)])
        frame = cv2.flip(frame, 0)
        buf = frame.tostring()
        # print buf
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture
        # TODO: Draw rectangle from self.parent.selected_region here


class PlotDisplay(Image):
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
