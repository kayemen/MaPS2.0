import cv2
import numpy as np
import glob
import time

rect_data = {'pt1': (0, 0), 'pt2': (0, 0), 'color': (0, 65535, 0), 'thickness': 2}
freeform_data = {'ptarray': [], 'color': (0, 65535, 0), 'thickness': 2}
selecting = False


class GUIException(Exception):
    pass


def nothing(arg):
    pass


def rectangular_select(event, x, y, flags, param):
    '''
    Callback for rectangular selection window
    '''
    global rect_data, selecting
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        rect_data['pt1'] = (x, y)
        rect_data['pt2'] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        rect_data['pt2'] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        rect_data['pt2'] = (x, y)


def lock_selection(*arg, **kwargs):
    pass


def draw_image_overlay(window_name, img, overlay_type=None, overlay_data=None, normalize=True):
    # frame = img.copy()
    frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if normalize:
        frame = cv2.normalize(frame, None, 0, 65535, cv2.NORM_MINMAX)
    if overlay_type == 'rectangle':
        if overlay_data is None:
            raise GUIException('Rectangle parameters not passed')
        if set(['pt1', 'pt2', 'color']) > set(overlay_data.keys()):
            raise GUIException('Required rectangle parameters not passed - %s' % (set(overlay_data.keys()) - set(['pt1', 'pt2', 'color'])))
        cv2.rectangle(frame, **overlay_data)
    elif overlay_type == 'freeform':
        if overlay_data is None:
            raise GUIException('Polygon data not provided')
        if set(['pts', 'color', 'lineType']) > set(overlay_data.keys()):
            raise GUIException('Required polygon parameters not passed - %s' % (set(overlay_data.keys()) - set()))
    else:
        raise GUIException('Unknown overlay type')
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF


def load_image_sequence(img_path_list):
    img_seq = []
    try:
        for img_path in img_path_list:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_seq.append(image)
    except:
        import traceback
        traceback.print_exc()

    return img_seq


def max_heartsize_frame_selection(img_seq):
    cv2.namedWindow("MaxHeartframe")
    cv2.setMouseCallback("MaxHeartframe", rectangular_select)
    rect_params = {'color': (0, 65535, 0), 'thickness': 3}
    key = draw_image_overlay("MaxHeartframe", img_seq[0], overlay_type='rectangle', overlay_data=rect_data)
    selected_frame = 0
    cv2.createTrackbar('img_id', 'MaxHeartframe', selected_frame, len(img_seq) - 1, nothing)
    while key != ord('q'):
        # print cv2.getWindowProperty('MaxHeartframe', 0)
        key = draw_image_overlay("MaxHeartframe", img_seq[cv2.getTrackbarPos('img_id', 'MaxHeartframe')], overlay_type='rectangle',
                                 overlay_data=rect_data)


if __name__ == '__main__':
    path_list = glob.glob('../Data sets/Phase_Bidi/*.tif')
    img_seq = load_image_sequence(path_list[:100])
    max_heartsize_frame_selection(img_seq)
