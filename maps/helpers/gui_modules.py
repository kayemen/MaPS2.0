from skimage.transform import resize

from maps.helpers.tiffseriesimport import importtiff, writetiff
from maps.settings import setting

import cv2
import numpy as np
import glob
import time
from matplotlib import pyplot as plt

# Custom heatmap
cdict = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.001, 0.0, 0.01),
        (1.0, 1.0, 1.0)
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.001, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (0.999, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    ],
    'blue': [
        (0.0, 0.0, 0.0),
        (0.001, 0.0, 1.0),
        (1.0, 0.0, 0.0)
    ]
}
plt.register_cmap(name='custom_heatmap', data=cdict)


cv2_methods = {
    'ccorr_norm': cv2.TM_CCORR_NORMED,
    'ccoeff_norm': cv2.TM_CCOEFF_NORMED,
    'mse_norm': cv2.TM_SQDIFF_NORMED
}

rect_data = {
    'pt1': (0, 0),
    'pt2': (0, 0),
    'color': (0, 65535, 0),
    'thickness': 2
}
freeform_data = {
    'ptarray': [],
    'color': (0, 65535, 0),
    'thickness': 2
}

selecting = False


class GUIException(Exception):
    pass


def nothing(arg):
    pass


def generate_mask(frame):
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
            raise GUIException('Required rectangle parameters not passed - %s' %
                               (set(overlay_data.keys()) - set(['pt1', 'pt2', 'color'])))
        # Draw rectangle on frame
        cv2.rectangle(frame, **overlay_data)

    elif overlay_type == 'freeform':
        if overlay_data is None:
            raise GUIException('Polygon data not provided')
        if set(['ptarray', 'color', 'lineType']) > set(overlay_data.keys()):
            raise GUIException(
                'Required polygon parameters not passed - %s' % (set(overlay_data.keys()) - set()))
        # Draw polygon on image_texture
        for ptindex in range(len(overlay_data['ptarray'])):
            cv2.line(frame, overlay_data['ptarray'][ptindex], overlay_data['ptarray'][
                     (ptindex + 1) % len(overlay_data['ptarray'])], overlay_data['color'], overlay_data['thickness'])

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


def max_heartsize_frame(img_stack, startval=0):
    def draw_frame(img):
        frame = create_image_overlay(
            img,
            overlay_type='rectangle',
            overlay_data=rect_data
        )
        cv2.imshow("MaxHeartframe", frame)

    slider = False
    if len(img_stack.shape) > 2:
        slider = True
        slider_lim = img_stack.shape[2] - 1
    #     img = img_stack[:, :, 0]
    # else:
    #     img = img_stack

    cv2.namedWindow("MaxHeartframe")
    cv2.setMouseCallback("MaxHeartframe", rectangular_select)

    if slider:
        cv2.createTrackbar(
            'Frame', 'MaxHeartframe', 0, slider_lim,
            # lambda frame_id: draw_frame(img_stack[:, :, frame_id])
            lambda x: None
        )

    while cv2.getWindowProperty('MaxHeartframe', 0) >= 0:
        # print cv2.getWindowProperty('MaxHeartframe', 0)
        img = img_stack[:, :, cv2.getTrackbarPos(
            'Frame', 'MaxHeartframe')] if slider else img_stack
        draw_frame(img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break

    if slider:
        ref_frame = startval + cv2.getTrackbarPos('Frame', 'MaxHeartframe')
    else:
        ref_frame = None

    cv2.destroyAllWindows()

    return ref_frame


def masking_window_frame(img_stack, crop_selection=True, mask_selection=False, startval=0):
    global mask_array

    def draw_crop_frame(img):
        frame = create_image_overlay(
            img,
            overlay_type='rectangle',
            overlay_data=rect_data
        )
        cv2.imshow('CropFrame', frame)

    def draw_mask_frame(img):
        mask_region = create_image_overlay(
            img,
            overlay_type='freeform',
            overlay_data=freeform_data
        )
        cv2.imshow("MaskFrame", mask_region)

    slider = False
    if len(img_stack.shape) > 2:
        slider = True
        slider_lim = img_stack.shape[2] - 1
    #     img = img_stack[:, :, 0]
    # else:
    #     img = img_stack

    if crop_selection:
        cv2.namedWindow("CropFrame")
        cv2.setMouseCallback("CropFrame", rectangular_select)

        if slider:
            cv2.createTrackbar(
                'Frame', 'CropFrame', 0, slider_lim,
                lambda x: None
            )

        while cv2.getWindowProperty('CropFrame', 0) >= 0:
            img = img_stack[:, :, cv2.getTrackbarPos(
                'Frame', 'CropFrame')] if slider else img_stack
            draw_crop_frame(img)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break

        pt1 = rect_data['pt1']
        pt2 = rect_data['pt2']
        t11, t21 = (min(pt1[1], pt2[1]), max(pt1[1], pt2[1]))
        t10, t20 = (min(pt1[0], pt2[0]), max(pt1[0], pt2[0]))
        pt1 = (t10, t11)
        pt2 = (t20, t21)
        crop_region = img[pt1[1]:pt2[1] + 1, pt1[0]:pt2[0] + 1]

        if slider:
            frame_no = startval + cv2.getTrackbarPos('Frame', 'CropFrame')
        else:
            frame_no = None

    else:
        crop_region = img_stack
        frame_no = None

    slider = False
    if len(crop_region.shape) > 2:
        slider = True
        slider_lim = crop_region.shape[2] - 1
    #     img = crop_region[:, :, 0]
    # else:
    #     img = crop_region

    if mask_selection:
        cv2.namedWindow("MaskFrame")
        cv2.setMouseCallback("MaskFrame", freeform_select)

        if slider:
            cv2.createTrackbar(
                'Frame', 'MaskFrame', 0, slider_lim,
                lambda x: None
            )

        while cv2.getWindowProperty('MaskFrame', 0) >= 0:
            img = crop_region[:, :, cv2.getTrackbarPos(
                'Frame', 'MaskFrame')] if slider else crop_region
            draw_mask_frame(img)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break

        mask_array = generate_mask(
            create_image_overlay(
                img,
                overlay_type='freeform',
                overlay_data=freeform_data
            )
        )

        cv2.namedWindow("mask")
        while cv2.getWindowProperty('mask', 0) >= 0:
            cv2.imshow('mask', mask_array)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break

        if slider:
            frame_no = startval + cv2.getTrackbarPos('Frame', 'MaskFrame')
        # else:
        #     frame_no = None

        cv2.destroyAllWindows()

        return frame_no, mask_array

    cv2.destroyAllWindows()
    return frame_no
# def masking_window_frame(img):
#     cv2.namedWindow("CropFrame")
#     cv2.setMouseCallback("CropFrame", rectangular_select)

#     print img.shape
#     while cv2.getWindowProperty('CropFrame', 0) >= 0:
#         frame = create_image_overlay(
#             img,
#             overlay_type='rectangle',
#             overlay_data=rect_data
#         )
#         cv2.imshow("CropFrame", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 13:
#             break

#     cv2.namedWindow("MaskFrame")
#     cv2.setMouseCallback("MaskFrame", freeform_select)

#     pt1 = rect_data['pt1']
#     pt2 = rect_data['pt2']
#     t11, t21 = (min(pt1[1], pt2[1]), max(pt1[1], pt2[1]))
#     t10, t20 = (min(pt1[0], pt2[0]), max(pt1[0], pt2[0]))
#     pt1 = (t10, t11)
#     pt2 = (t20, t21)
#     # pt1[1], pt2[1] = (min(pt1[1], pt2[1]), max(pt1[1], pt2[1]))
#     # pt1[0], pt2[0] = (min(pt1[0], pt2[0]), max(pt1[0], pt2[0]))

#     crop_region = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

#     print crop_region.shape
#     while cv2.getWindowProperty('MaskFrame', 0) >= 0:
#         mask_region = create_image_overlay(
#             crop_region,
#             overlay_type='freeform',
#             overlay_data=freeform_data
#         )
#         cv2.imshow("MaskFrame", mask_region)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 13:
#             break

#     mask_frame = generate_mask(mask_region)
#     cv2.destroyAllWindows()
#     cv2.imshow('mask', mask_frame)
#     cv2.waitKey(0)


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


def extract_window(frame, x_start, x_end, y_start, y_end):
    '''
    Return the ROI from a frame within the bounding box specified by x_start, x_end, y_start, y_end
    '''
    frame_win = frame[int(x_start): int(x_end) + 1,
                      int(y_start): int(y_end) + 1]
    return frame_win


def load_frame(img_path, frame_no, upsample=True, crop=False, cropParams=(), index_start_number=None, prefix=None, num_digits=None):
    '''
    Load the tiff file of the frame, resize (upsample by resampling factor if needed), crop using cropParams if needed and return image array.
    '''
    if index_start_number is None:
        index_start_number = setting['index_start_at']
    if prefix is None:
        prefix = setting['image_prefix']
    if num_digits is None:
        num_digits = setting['num_digits']
    img = importtiff(img_path, frame_no, prefix=prefix,
                     index_start_number=index_start_number, num_digits=num_digits)
    if upsample:
        img = resize(
            img,
            (
                img.shape[0] * setting['resampling_factor'],
                img.shape[1] * setting['resampling_factor']
            ),
            preserve_range=True
        )
    if crop:
        img = extract_window(img, *cropParams)

    return img


def apply_mask(frame, mask):
    if not frame.shape == mask.shape:
        raise Exception('Size of frame and mask not matching')

    return frame[np.where(mask != 0)]


def apply_unmask(masked_frame, mask):
    unmasked_frame = np.zeros(mask.shape).astype(masked_frame.dtype)
    unmasked_frame[np.where(mask != 0)] = masked_frame
    return unmasked_frame


if __name__ == '__main__':
    path_list = glob.glob('../Data sets/Phase_Bidi/*.tif')
    img_seq = load_image_sequence(path_list[:100])
    max_heartsize_frame_selection(img_seq)
