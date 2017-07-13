from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.treeview import TreeView, TreeViewNode, TreeViewLabel
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.graphics import Rectangle, Color, Line
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.core.window import Window

# from ..gui_modules import load_image_sequence, create_image_overlay
from maps import settings

from maps.core.z_stamping import z_stamping_step

from maps.helpers.gui_modules import load_image_sequence, create_image_overlay, rectangular_select, max_heartsize_frame, get_rect_params
from maps.helpers.misc import pickle_object
from maps.helpers.plotting_module import line_plot
from maps.gui.widgets.image_display import FrameSelectionWidget
from maps.gui.widgets.settings_widgets import FileChooserWidget, SettingRow

import json
import os
import glob
import cv2
import threading
import traceback
import pprint

error_message = ''
DEBUG = True


class InterfaceManager(BoxLayout):
    screen_area = ObjectProperty()

    scr1_btn = ObjectProperty()
    scr2_btn = ObjectProperty()
    scr3_btn = ObjectProperty()
    scr4_btn = ObjectProperty()
    scr5_btn = ObjectProperty()
    scr6_btn = ObjectProperty()
    scr7_btn = ObjectProperty()

    prev_btn = ObjectProperty()
    next_btn = ObjectProperty()

    current_view = NumericProperty(0)
    # progress_bar = ObjectProperty()

    def __init__(self, **kwargs):
        super(InterfaceManager, self).__init__(**kwargs)
        self.InputParametersView = InputParametersView()
        self.view_order = [
            self.InputParametersView,
        ]
        self.current_view = 0
        settings.read_setting_from_json()
        self.screen_area.add_widget(self.view_order[self.current_view])
        # self.progress_bar.value_normalized = float(self.current_view) / len(self.view_order)
        popup_content = BoxLayout(orientation='vertical')
        self.error_content = Label(text=error_message)
        popup_content.add_widget(self.error_content)
        cls_btn = Button(text='Close', size_hint=(1.0, 0.1))
        popup_content.add_widget(cls_btn)
        self.error_popup = Popup(title='Error', content=popup_content, size_hint=(0.6, 0.6))
        cls_btn.bind(on_release=self.error_popup.dismiss)

    #     self.add_widget(self.InputParametersView)

    def load_widgets(self):
        self.ReferenceFrameSelectionView = ReferenceFrameSelectionView()
        self.ZookPruningView = ZookPruningView()

        self.view_order = [
            self.InputParametersView,
            self.ReferenceFrameSelectionView,
            self.ZookPruningView
        ]

        self.current_view = 0

    def load_previous_screen(self):
        if self.current_view > 0:
            self.load_screen((self.current_view - 1))
        # self.progress_bar.value_normalized = float(self.current_view) / len(self.view_order)

    def load_next_screen(self):
        if self.current_view == 0:
            self.load_widgets()
        if self.current_view < len(self.view_order) - 1:
            self.load_screen((self.current_view + 1))
            # self.progress_bar.value_normalized = float(self.current_view) / len(self.view_order)

    def load_screen(self, view_id):
        global error_message
        error_message = ''
        if view_id >= 1 and (not all([self.view_order[temp_id].validate_step() for temp_id in range(view_id)])):
            try:
                self.error_content.text = error_message
                self.error_popup.open()
            except:
                self.error_content.text = 'Unknown error. Please view error logs'
                self.error_popup.open()
                # print 'No error message set for view %s' % (repr(self.view_order[self.current_view]))
        elif view_id >= len(self.view_order):
            pass
        else:
            self.current_view = view_id
            self.screen_area.children[0].widget_inactive()
            self.screen_area.clear_widgets()
            self.screen_area.add_widget(self.view_order[self.current_view])
            self.screen_area.children[0].widget_active()

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

    def reload_settings(self, new_parameters):
        # TODO
        print 'reloading settings'
        settings.read_setting_from_json()
        self.load_widgets()


class InputParametersView(BoxLayout):
    # parameters = json.load(open('maps/default_inputs.json'))
    parameters = ObjectProperty()
    json_path = StringProperty(os.path.join(settings.BASE_DIR, 'current_inputs.json'))
    setting_path = os.path.join(settings.BASE_DIR, 'current_inputs.json')
    screen = ObjectProperty()

    def __init__(self, **kwargs):
        super(InputParametersView, self).__init__(**kwargs)

        self.load_settings_json()

        # Used for scrollview.
        # Sets the height of children inside the gridlayout to 35 pixels
        self.screen.height = len(self.parameters) * 35

    def widget_active(self):
        pass

    def widget_inactive(self):
        pass

    def load_settings_json(self):
        try:
            self.parameters = json.load(open(self.json_path))
            self.screen.clear_widgets()
            self.setting_list = {}
            self.load_setting_widgets()
        except:
            print 'Unable to load json file'
            traceback.print_exc()

    def json_load_popup(self):

        def get_json_path(instance):
            self.json_path = self.json_pop.content.children[1].text
            try:
                self.load_settings_json()
                return False
            except:
                return True

        pop_content = BoxLayout(orientation='vertical')
        pop_content.add_widget(FileChooserWidget(dirselect=False))

        self.json_pop = Popup(
            title="Load new settings from json file",
            content=pop_content,
            size_hint=(0.5, 0.2)
        )
        pop_content.add_widget(
            Button(text='OK', on_release=self.json_pop.dismiss)
        )
        self.json_pop.bind(on_dismiss=get_json_path)
        self.json_pop.open()

    def load_setting_widgets(self):
        for setting_obj in self.parameters:
            if not setting_obj['hidden']:
                new_setting_layout = self.create_setting_layout(**setting_obj)
                self.setting_list[setting_obj['varname']] = new_setting_layout
                self.screen.add_widget(self.setting_list[setting_obj['varname']])

    def create_setting_layout(self, varname, description, type, helptext=None, value=None, hidden=False):
        setting_layout = SettingRow(size_hint=(1, 0.06))
        setting_layout.add_widget(Label(text=description, size_hint_x=0.4))
        if type == 'path':
            setting_layout.add_widget(FileChooserWidget(
                init_text=value,
                size_hint_x=0.6,
                dirselect=True
            ))
        elif type == 'file':
            setting_layout.add_widget(FileChooserWidget(
                init_text=value,
                size_hint_x=0.6,
                dirselect=False
            ))
        elif type == 'str':
            setting_layout.add_widget(TextInput(
                text=str(value),
                size_hint_x=0.6
            ))
        elif type == 'int' or type == 'float':
            setting_layout.add_widget(TextInput(
                text=str(value),
                size_hint_x=0.6
            ))

        return setting_layout

    def dump_new_settings(self):
        # lambda function to return {} object from inside [] of parameters having field 'varname' as varname
        get_param_obj = lambda varname: self.parameters[[var['varname'] for var in self.parameters].index(varname)]

        for varname in self.setting_list.keys():
            obj = get_param_obj(varname)
            new_val = self.setting_list[varname].get_setting_value()
            obj['value'] = new_val

        # pprint.pprint(self.parameters, width=1)
        json.dump(self.parameters, open(self.setting_path, 'w'))
        self.json_path = self.setting_path[:]

    def reload_settings(self):
        print self.setting_list
        # print self.setting_list['km_path'].get_setting_value()
        # print dir(self.setting_list['km_path'])
        print 'creating new settings'
        new_settings = self.dump_new_settings()
        self.parent.parent.reload_settings(new_settings)

    def validate_step(self):
        global error_message
        # TODO
        error_message = 'Error in data entered. Please check entered data'
        # Return false if error found
        return True


class ReferenceFrameSelectionView(BoxLayout):
    frame_window = ObjectProperty()
    img_select = NumericProperty()
    frame_select = ObjectProperty()
    region_select = ObjectProperty()

    def __init__(self, **kwargs):
        super(ReferenceFrameSelectionView, self).__init__(**kwargs)
        # self.fps = fps
        self.selected_frame = None
        self.selected_region = None
        self.frame_window.frame_count = settings.setting['fphb']
        img_paths = glob.glob(
            os.path.join(settings.setting['bf_path'], '*.tif')
        )[:self.frame_window.frame_count]
        self.frame_window.load_frames(img_paths)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if not len(modifiers):
            if keycode[1] == 'right':
                if self.frame_window.frame_selector.value < self.frame_window.frame_selector.max:
                    self.frame_window.frame_selector.value = self.frame_window.frame_selector.value + 1
            elif keycode[1] == 'left':
                if self.frame_window.frame_selector.value > self.frame_window.frame_selector.min:
                    self.frame_window.frame_selector.value -= 1
        return True

    def widget_active(self):
        # self.refresh = Clock.schedule_interval(self.frame_window.update, 1.0 / self.fps)
        self.frame_window.refresh = True
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def widget_inactive(self):
        self.frame_window.refresh = False
        # self.refresh.cancel()

    def mark_frame(self):
        self.selected_frame = int(self.frame_window.frame_select)
        self.selected_region = None
        sel_frame = self.frame_window.img_seq[self.selected_frame]
        threading.Thread(
            target=max_heartsize_frame,
            args=(sel_frame, )
        ).start()
        self.region_select.disabled = False

    def mark_region(self):
        cv2.destroyAllWindows()
        self.selected_region = get_rect_params()
        data = [
            ('frame', self.selected_frame),
            ('x_end', self.selected_region['x_end']),
            ('height', self.selected_region['height']),
            ('y_end', self.selected_region['y_end']),
            ('width', self.selected_region['width']),
        ]
        settings.setting['ref_frame'] = self.selected_frame
        pickle_object(data, file_name='corr_window.csv', dumptype='csv')
        print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])
        self.frame_window.overlay_data = self.selected_region
        self.frame_window.overlay = True

    def on_img_select(self, instance, value):
        # Disabling the region selection button till frame is finalized
        self.region_select.disabled = True
        self.frame_window.overlay_data = {}
        self.frame_window.overlay = False

    def validate_step(self):
        global error_message
        if DEBUG:
            return True
        if self.selected_frame is None or self.selected_region is None:
            error_message = 'Please select a frame and a region within the frame'
            return False
        return True


class ZookPruningView(BoxLayout):
    img_frame = ObjectProperty()
    plot_window = ObjectProperty()
    kymo_sel = ObjectProperty()
    zook_sel = ObjectProperty()
    selected_zook = NumericProperty(-1)
    selected_frame = NumericProperty(-1)

    def __init__(self, fps=20, **kwargs):
        super(ZookPruningView, self).__init__(**kwargs)
        # No idea why this is needed
        self.plot_window.initialize_plots()

        self.fps = fps
        self.img_frame.load_frames(None)
        kymo_paths = glob.glob(
            os.path.join(
                settings.setting['km_path'],
                '*.tif'
            )
        )
        self.kymo_names = {os.path.basename(filepath): False for filepath in kymo_paths}
        self.kymo_paths = {os.path.basename(filepath): filepath for filepath in kymo_paths}
        kymo_names = self.kymo_names.keys()
        kymo_names.sort()
        self.kymo_sel.values = kymo_names
        self.selected_kymo = ''
        self.img_frame.frame_count = settings.setting['ZookZikPeriod']
        self.img_frame.display_blank = True
        self.bad_zooks = []
        self.bad_zooks_list = []
        self.discarded_zooks_list = []

        def show_display(spinner, text):
            if text != self.selected_kymo:
                self.clear_zook_tree()
            self.selected_kymo = text

        self.kymo_sel.bind(text=show_display)
        self.bind(selected_zook=self.load_zook_frames, selected_frame=self.move_slider_to_frame)

        self.zook_sel.bind(minimum_height=self.zook_sel.setter('height'))

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if not len(modifiers):
            if keycode[1] == 'right':
                if self.img_frame.frame_selector.value < self.img_frame.frame_selector.max:
                    self.img_frame.frame_selector.value = self.img_frame.frame_selector.value + 1
            elif keycode[1] == 'left':
                if self.img_frame.frame_selector.value > self.img_frame.frame_selector.min:
                    self.img_frame.frame_selector.value -= 1
        return True

    def widget_active(self):
        self.img_frame.refresh = True
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def widget_inactive(self):
        self.img_frame.refresh = False

    def process_kymograph(self):
        if self.selected_kymo != '':
            self.img_frame.refresh = False
            # self.refresh.cancel()
            if self.kymo_names[self.selected_kymo]:
                print 'Already processed this kymo'
                self.z_stamps, _, residues, self.bad_zooks = z_stamping_step(
                    kymo_path=self.kymo_paths[self.selected_kymo],
                    frame_count=settings.setting['bf_framecount'],
                    phase_img_path=settings.setting['bf_path'],
                    use_old=True,
                    datafile_name='z_stamp_opt_%s.pkl' % (self.selected_kymo[:-4])
                )
            else:
                self.z_stamps, _, residues, self.bad_zooks = z_stamping_step(
                    kymo_path=self.kymo_paths[self.selected_kymo],
                    frame_count=settings.setting['bf_framecount'],
                    phase_img_path=settings.setting['bf_path'],
                    use_old=True,
                    # use_old=settings.setting['use_pkl_zstamp'],
                    datafile_name='z_stamp_opt_%s.pkl' % (self.selected_kymo[:-4])
                )

            self.selected_zook = -1
            self.selected_frame = -1
            self.discarded_zooks_list = []
            self.plot_window.clear_plots()

            self.kymo_names[self.selected_kymo] = True
            self.bad_zooks_to_tree()

            self.plot_window.add_plotting_method('Residues', line_plot)
            self.plot_window.add_plotting_method('ZStamps', line_plot)
            # self.plot_window.add_plotting_method('Bad zook locations', scatter_plot)

            self.plot_window.update_plot_data('Residues', residues)
            self.plot_window.update_plot_data('ZStamps', self.z_stamps)

            self.img_frame.refresh = True

    # FrameSelectionWidget modification functions
    def load_zook_frames(self, widget, zook_no):
        if zook_no == -1:
            # Load blank frame
            self.img_frame.display_blank = True
            self.plot_window.plot_selection.text = 'Select plot'
        else:
            self.img_frame.display_blank = False
            start_frame = zook_no * settings.setting['ZookZikPeriod']
            end_frame = start_frame + settings.setting['ZookZikPeriod']
            img_paths = glob.glob(
                os.path.join(settings.setting['bf_path'], '*.tif')
            )[start_frame: end_frame]
            self.img_frame.load_frames(img_paths)
            self.img_frame.frame_start = start_frame
            if self.selected_frame == -1:
                self.selected_frame = 0

            self.plot_window.plot_selection.text = 'Zook#%d' % zook_no

    def move_slider_to_frame(self, widget, frame_no):
        if frame_no == -1:
            # Do nothing
            pass
        else:
            self.img_frame.frame_selector.value = frame_no % settings.setting['ZookZikPeriod']

    # Zook tree view methods
    def clear_zook_tree(self):
        self.bad_zooks_list = []

        for node in self.zook_sel.root.nodes[:]:
            self.zook_sel.remove_node(node)

    def bad_zooks_to_tree(self):
        self.clear_zook_tree()
        for bad_zook in self.bad_zooks:
            zook_no = bad_zook[0]
            self.bad_zooks_list.append(zook_no)
            zook_node = self.zook_sel.add_node(
                TreeViewLabel(text='Zook#%d' % (zook_no))
            )
            zook_node.bind(is_selected=self.zook_node_callback)
            self.plot_window.add_plotting_method('Zook#%d' % (zook_no), line_plot)

            zook_start = max((zook_no) * settings.setting['ZookZikPeriod'], 0)
            zook_end = min((zook_no + 1) * settings.setting['ZookZikPeriod'], len(self.z_stamps))

            self.plot_window.update_plot_data('Zook#%d' % (zook_no), self.z_stamps[zook_start:zook_end])

            for bad_frame in bad_zook[2]:
                frame_node = self.zook_sel.add_node(
                    TreeViewLabel(text='Frame#%d' % (bad_frame)),
                    zook_node
                )
                frame_node.bind(is_selected=self.frame_node_callback)

        if len(self.zook_sel.root.nodes) > 0:
            self.zook_sel.select_node(self.zook_sel.root.nodes[0])
        else:
            self.selected_zook = -1
            self.process_discarded_zooks()

    # Tree view callbacks
    def zook_node_callback(self, zook_node, selected):
        if selected:
            self.selected_zook = int(zook_node.text.replace('Zook#', ''))

    def frame_node_callback(self, frame_node, selected):
        if selected:
            zook_no = int(frame_node.parent_node.text.replace('Zook#', ''))
            if self.selected_zook != zook_no:
                self.selected_zook = zook_no
            self.selected_frame = int(frame_node.text.replace('Frame#', ''))

    # Button callbacks
    def prev_zook_callback(self):
        prev_zook = (self.bad_zooks_list.index(self.selected_zook) - 1) % len(self.bad_zooks_list)
        self.zook_sel.deselect_node()
        self.zook_sel.select_node(self.zook_sel.root.nodes[prev_zook])

    def next_zook_callback(self):
        next_zook = (self.bad_zooks_list.index(self.selected_zook) + 1) % len(self.bad_zooks_list)
        self.zook_sel.deselect_node()
        self.zook_sel.select_node(self.zook_sel.root.nodes[next_zook])

    def keep_zook_callback(self):
        curr_zook = self.bad_zooks_list.index(self.selected_zook)
        self.bad_zooks.pop(curr_zook)
        self.bad_zooks_list.pop(curr_zook)
        self.bad_zooks_to_tree()

    def discard_zook_callback(self):
        curr_zook = self.bad_zooks_list.index(self.selected_zook)
        self.discarded_zooks_list.append(self.bad_zooks.pop(curr_zook))
        # self.bad_zooks.pop(curr_zook)
        self.bad_zooks_list.pop(curr_zook)
        self.bad_zooks_to_tree()

    def discard_all_callback(self):
        self.discarded_zooks_list.extend(self.bad_zooks)
        self.bad_zooks = []
        self.bad_zooks_list = []
        self.bad_zooks_to_tree()

    def process_discarded_zooks(self):
        # Write the final z stamp after discarding bad zooks
        print 'Discarded zooks-', self.discarded_zooks_list
        # self.passed_zstamps = []
        # for zook in

    def validate_step(self):
        return True


class CroppingWindowView(BoxLayout):
    # frame_window = ObjectProperty()
    # img_select = NumericProperty()
    crop_frame = ObjectProperty()
    mask_frame = ObjectProperty()
    crop_region = ObjectProperty()
    mask_region = ObjectProperty()

    def __init__(self, **kwargs):
        super(CroppingWindowView, self).__init__(**kwargs)
        # self.fps = fps
        self.selected_frame = None
        self.crop_region = None
        self.mask_region = None

        self.crop_frame.frame_selector.disabled = True
        self.mask_frame.frame_selector.disabled = True
        # img_paths = glob.glob(
        #     os.path.join(settings.setting['bf_path'], '*.tif')
        # )[:self.frame_window.frame_count]
        # self.frame_window.load_frames(img_paths)

    def widget_active(self):
        self.selected_frame = settings.setting['ref_frame']
        img_paths = glob.glob(
            os.path.join(settings.setting['bf_path'], '*.tif')
        )[self.selected_frame]
        self.frame_window.refresh = True
        self.frame_window.load_frames(img_paths)

    def widget_inactive(self):
        self.frame_window.refresh = False
        # self.refresh.cancel()

    def mark_cropping(self):
        self.selected_frame = int(self.frame_window.frame_select)
        self.selected_region = None
        sel_frame = self.frame_window.img_seq[self.selected_frame]
        threading.Thread(
            target=max_heartsize_frame,
            args=(sel_frame, )
        ).start()
        self.region_select.disabled = False

    def mark_mask(self):
        cv2.destroyAllWindows()
        self.selected_region = get_rect_params()
        data = [
            ('frame', self.selected_frame),
            ('x_end', self.selected_region['x_end']),
            ('height', self.selected_region['height']),
            ('y_end', self.selected_region['y_end']),
            ('width', self.selected_region['width']),
        ]
        pickle_object(data, file_name='corr_window.csv', dumptype='csv')
        print '\n'.join(['%s:%d' % (i[0], i[1]) for i in data])
        self.frame_window.overlay_data = self.selected_region
        self.frame_window.overlay = True

    def on_img_select(self, instance, value):
        # Disabling the region selection button till frame is finalized
        self.region_select.disabled = True
        self.frame_window.overlay_data = {}
        self.frame_window.overlay = False

    def validate_step(self):
        global error_message
        if DEBUG:
            return True
        if self.selected_frame is None or self.selected_region is None:
            error_message = 'Please select a frame and a region within the frame'
            return False
        return True


class MapsApp(App):

    def build(self):
        return InterfaceManager()


if __name__ == '__main__':
    MapsApp().run()
