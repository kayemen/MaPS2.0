from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.graphics import Rectangle, Color, Line

from kivy.graphics.texture import Texture
from kivy.uix.image import Image

# from ..gui_modules import load_image_sequence, create_image_overlay
from maps.helpers.gui_modules import load_image_sequence, create_image_overlay, rectangular_select, max_heartsize_frame, get_rect_params

import json
import os
import glob
import cv2
import threading


class InterfaceManager(BoxLayout):
    screen_area = ObjectProperty()

    scr1_btn = ObjectProperty()
    scr2_btn = ObjectProperty()
    scr3_btn = ObjectProperty()
    # scr4_btn = ObjectProperty()
    # scr5_btn = ObjectProperty()
    # scr6_btn = ObjectProperty()
    # scr7_btn = ObjectProperty()

    prev_btn = ObjectProperty()
    next_btn = ObjectProperty()

    current_view = NumericProperty(0)
    # progress_bar = ObjectProperty()

    def __init__(self, **kwargs):
        super(InterfaceManager, self).__init__(**kwargs)
        self.InputParametersView = InputParametersView()
        self.FrameSelectionView = FrameSelectionView()
        self.ZookPruningView = ZookPruningView()
        self.view_order = [
            self.InputParametersView,
            self.FrameSelectionView,
            self.ZookPruningView
        ]
        self.current_view = 0
        self.screen_area.add_widget(self.view_order[self.current_view])
        # self.progress_bar.value_normalized = float(self.current_view) / len(self.view_order)

    #     self.add_widget(self.InputParametersView)

    def load_previous_screen(self):
        if self.current_view > 0:
            self.load_screen((self.current_view - 1))
        # self.progress_bar.value_normalized = float(self.current_view) / len(self.view_order)

    def load_next_screen(self):
        if self.current_view < len(self.view_order) - 1:
            self.load_screen((self.current_view + 1))
            # self.progress_bar.value_normalized = float(self.current_view) / len(self.view_order)

    def load_screen(self, view_id):
        if view_id >= 1 and (not all([self.view_order[temp_id].validate_step() for temp_id in range(view_id)])):
            try:
                self.view_order[self.current_view].error_popup.open()
            except:
                print 'No error message set for view %s' % (repr(self.view_order[self.current_view]))
        elif view_id >= len(self.view_order):
            pass
        else:
            self.current_view = view_id
            self.screen_area.clear_widgets()
            self.screen_area.add_widget(self.view_order[self.current_view])

    def on_current_view(self, instance, value):
        for index in range(len(self.view_order)):
            btn = eval('self.scr%d_btn' % (index + 1))
            if index == value:
                if btn.disabled:
                    btn.disabled = False
                btn.color = [0.3, 0.6, 1, 1]
            else:
                btn.color = [1, 1, 1, 1]
        # eval('self.scr%d_btn' % (value + 1)).color = [0,1,0,1]
        if eval('self.scr%d_btn' % (value + 1)).disabled:
            eval('self.scr%d_btn' % (value + 1)).disabled = False

        if value >= len(self.view_order):
            self.next_btn.disabled = True
        else:
            self.next_btn.disabled = False

        if value == 0:
            self.prev_btn.disabled = True
        else:
            self.prev_btn.disabled = False


class InputParametersView(BoxLayout):
    parameters = json.load(open('maps/default_inputs.json'))
    screen = ObjectProperty()

    def __init__(self, **kwargs):
        super(InputParametersView, self).__init__(**kwargs)
        self.screen.add_widget(Label(text="testing"))

        popup_content = BoxLayout(orientation='vertical')
        popup_content.add_widget(Label(text='Please fix the errors in the data entered'))
        cls_btn = Button(text='Close', size_hint=(1.0, 0.1))
        popup_content.add_widget(cls_btn)
        self.error_popup = Popup(title='Errors', content=popup_content, size_hint=(0.6, 0.6))
        cls_btn.bind(on_release=self.error_popup.dismiss)

    def validate_step(self):
        return True
#     def generate_(self, arg):
#         pass


class FrameDisplay(Image):

    def __init__(self, path='../Data sets/Phase_Bidi/*.tif', frame_count=50, **kwargs):
        super(FrameDisplay, self).__init__(**kwargs)
        self.frame_count = frame_count
        self.path_list = glob.glob(path)[:self.frame_count]
        self.selection_window = [0, 0]
        self.img_seq = load_image_sequence(self.path_list)
        # print self.path_list
        # self.img = load_image_sequence(path_list[0])[0]

    def update(self, dt):
        # print self.path_list[0]
        # print img_seq[0]
        frame = create_image_overlay(self.img_seq[int(self.parent.parent.img_select)])
        frame = cv2.flip(frame, 0)
        buf = frame.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture


class FrameSelectionView(BoxLayout):
    img_frame = ObjectProperty()
    slider = ObjectProperty()
    img_select = NumericProperty()
    frame_select = ObjectProperty()
    region_select = ObjectProperty()

    def __init__(self, fps=30, **kwargs):
        super(FrameSelectionView, self).__init__(**kwargs)
        Clock.schedule_interval(self.img_frame.update, 1.0 / fps)
        self.selected_frame = None
        self.selected_region = None

    def mark_frame(self):
        self.selected_frame = int(self.img_select)
        self.selected_region = None
        threading.Thread(target=max_heartsize_frame, args=(self.img_frame.img_seq[self.selected_frame], )).start()
        self.region_select.disabled = False
        # print params

    def mark_region(self):
        cv2.destroyAllWindows()
        self.selected_region = get_rect_params()
        print 'Frame:', self.selected_frame
        print 'Region:', self.selected_region

    def on_img_select(self, instance, value):
        # Change in value of slider
        # Disabling the region selection button till frame is finalized
        self.region_select.disabled = True
        # TODO: Should the old values immediately be discarded
        # self.selected_frame = None
        # self.selected_region = None

    # def on_touch_down(self, touch):
    #     if self.collide_point(*touch.pos):
    #         print 'box touch down detected'
    #         touch.push()
    #         touch.apply_transform_2d(self.to_local)
    #         super(FrameSelectionView, self).on_touch_down(touch)
    #         touch.pop()
    #
    # def on_touch_up(self, touch):
    #     if self.collide_point(*touch.pos):
    #         print 'box touch up detected'
    #         touch.push()
    #         touch.apply_transform_2d(self.to_local)
    #         super(FrameSelectionView, self).on_touch_up(touch)
    #         touch.pop()
    #
    # def on_touch_move(self, touch):
    #     if self.collide_point(*touch.pos):
    #         print 'box touch move detected'
    #         touch.push()
    #         touch.apply_transform_2d(self.to_local)
    #         super(FrameSelectionView, self).on_touch_move(touch)
    #         touch.pop()

    def validate_step(self):
        if self.selected_frame is None or self.selected_region is None:
            return False
        return True



class ZookPruningView(BoxLayout):

    def validate_step(self):
        return True


class MapsApp(App):

    def build(self):
        # paramters = json.load(open('../default_inputs.json'))

        # root_widget = BoxLayout(orientation='vertical')
        w = InterfaceManager()
        return w


if __name__ == '__main__':
    MapsApp().run()
