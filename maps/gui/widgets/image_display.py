from kivy.uix.image import Image
from kivy.graphics.texture import Texture

import glob
import cv2

from maps.helpers.gui_modules import create_image_overlay, load_image_sequence


class FrameDisplay(Image):

    def __init__(self, path='../Data sets/Phase_Bidi/*.tif', frame_count=50, **kwargs):
        super(FrameDisplay, self).__init__(**kwargs)
        self.frame_count = frame_count
        self.path_list = glob.glob(path)[:self.frame_count]
        self.selection_window = [0, 0]
        self.img_seq = load_image_sequence(self.path_list)

    def update(self, dt):
        frame = create_image_overlay(self.img_seq[int(self.parent.parent.img_select)])
        frame = cv2.flip(frame, 0)
        buf = frame.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture
        # TODO: Draw rectangle from self.parent.selected_region here
