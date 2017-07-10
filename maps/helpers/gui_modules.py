import cv2
import numpy as np
import glob
import time

cv2_methods = {'ccorr_norm': cv2.TM_CCORR_NORMED, 'ccoeff_norm': cv2.TM_CCOEFF_NORMED, 'mse_norm': cv2.TM_SQDIFF_NORMED}

rect_data = {'pt1': (0, 0), 'pt2': (0, 0), 'color': (0, 65535, 0), 'thickness': 2}
freeform_data = {'ptarray': [], 'color': (0, 65535, 0), 'thickness': 2}
selecting = False


class GUIException(Exception):
    pass


def nothing(arg):
    pass


def generate_mask(frame):
    print 'Generating mask'
    ht, wt, _ = frame.shape
    mask = np.zeros((ht, wt), dtype=np.uint8)
    mask_array = np.array([freeform_data['ptarray']], dtype=np.int32)
    cv2.fillPoly(mask, mask_array, 255)

    # cv2.imshow('mask', mask)
    return mask


def freeform_select(event, x, y, flags, param):
    '''
    Callback for free-form selection window
    '''
    global freeform_data, selecting
    if event == cv2.EVENT_LBUTTONDOWN:
        freeform_data['ptarray'] = [(x, y)]
        selecting = True

    elif event == cv2.EVENT_LBUTTONUP:
        freeform_data['ptarray'].append((x, y))
        selecting = False

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        freeform_data['ptarray'].append((x, y))


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


def create_blank_image(height, width, imgtype='white'):
    # TODO Check if more efficient to create seoarate frame for each case
    frame = np.ones((height, width, 3), np.uint8)

    if imgtype == 'white':
        frame = frame * 255
    else:
        frame = frame * 0

    return frame


def create_image_overlay(img, overlay_type=None, overlay_data=None, normalize=True):
    # frame = img.copy()
    frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if normalize:
        frame = cv2.normalize(frame, None, 0, 65535, cv2.NORM_MINMAX)

    if overlay_type is None:
        pass

    elif overlay_type == 'rectangle':
        if overlay_data is None:
            raise GUIException('Rectangle parameters not passed')
        if set(['pt1', 'pt2', 'color']) > set(overlay_data.keys()):
            raise GUIException('Required rectangle parameters not passed - %s' % (set(overlay_data.keys()) - set(['pt1', 'pt2', 'color'])))
        # Draw rectangle on frame
        cv2.rectangle(frame, **overlay_data)

    elif overlay_type == 'freeform':
        if overlay_data is None:
            raise GUIException('Polygon data not provided')
        if set(['ptarray', 'color', 'lineType']) > set(overlay_data.keys()):
            raise GUIException('Required polygon parameters not passed - %s' % (set(overlay_data.keys()) - set()))
        # Draw polygon on image_texture
        for ptindex in range(len(overlay_data['ptarray'])):
            cv2.line(frame, overlay_data['ptarray'][ptindex], overlay_data['ptarray'][(ptindex + 1) % len(overlay_data['ptarray'])], overlay_data['color'], overlay_data['thickness'])

    else:
        raise GUIException('Unknown overlay type - %s' % overlay_type)

    # print type(frame[0,0,0])
    frame = (frame / 256).astype('uint8')
    return frame


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


def max_heartsize_frame(img):
    cv2.namedWindow("MaxHeartframe")
    cv2.setMouseCallback("MaxHeartframe", rectangular_select)

    while cv2.getWindowProperty('MaxHeartframe', 0) >= 0:
        # print cv2.getWindowProperty('MaxHeartframe', 0)
        frame = create_image_overlay(
            img,
            overlay_type='rectangle',
            overlay_data=rect_data
        )
        cv2.imshow("MaxHeartframe", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break

    cv2.destroyAllWindows()


def masking_window_frame(img):
    cv2.namedWindow("CropFrame")
    cv2.setMouseCallback("CropFrame", rectangular_select)

    print img.shape
    while cv2.getWindowProperty('CropFrame', 0) >= 0:
        frame = create_image_overlay(
            img,
            overlay_type='rectangle',
            overlay_data=rect_data
        )
        cv2.imshow("CropFrame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break

    cv2.namedWindow("MaskFrame")
    cv2.setMouseCallback("MaskFrame", freeform_select)

    pt1 = rect_data['pt1']
    pt2 = rect_data['pt2']
    crop_region = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    print crop_region.shape
    while cv2.getWindowProperty('MaskFrame', 0) >= 0:
        mask_region = create_image_overlay(
            crop_region,
            overlay_type='freeform',
            overlay_data=freeform_data
        )
        cv2.imshow("MaskFrame", mask_region)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break

    mask_frame = generate_mask(mask_region)
    cv2.destroyAllWindows()
    cv2.imshow('mask', mask_frame)
    cv2.waitKey(0)


def get_rect_params():
    # return [rect_data['pt1'], rect_data['pt2']]
    return {
        'x_end': max(rect_data['pt1'][1], rect_data['pt2'][1]),
        'y_end': max(rect_data['pt1'][0], rect_data['pt2'][0]),
        'height': abs(rect_data['pt2'][1] - rect_data['pt1'][1]),
        'width': abs(rect_data['pt2'][0] - rect_data['pt1'][0]),
    }


def put_rect_params(rect_params, color=(0, 65535, 0), thickness=2):
    return {
        'pt1': (
            rect_params['y_end'] - rect_params['width'],
            rect_params['x_end'] - rect_params['height']
        ),
        'pt2': (
            rect_params['y_end'],
            rect_params['x_end']
        ),
        'color': color,
        'thickness': thickness
    }

if __name__ == '__main__':
    path_list = glob.glob('../Data sets/Phase_Bidi/*.tif')
    img_seq = load_image_sequence(path_list[:100])
    max_heartsize_frame_selection(img_seq)
