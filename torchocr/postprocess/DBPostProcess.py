import numpy as np
import cv2
from pyclipper import PyclipperOffset
import pyclipper
from shapely.geometry import Polygon

class DBPostProcess:
    def __init__(self, thresh=0.3, unclip_ratio=1.5, box_thresh=0.6):
        self.min_size = 3
        self.thresh = thresh
        self.bbox_scale_ratio = unclip_ratio
        self.shortest_length = 5

    def __call__(self, _predict_score, _ori_img_shape):
        instance_score = _predict_score.squeeze()
        h, w = instance_score.shape[:2]
        height, width = _ori_img_shape[0]
        available_region = np.zeros_like(instance_score, dtype=np.float32)
        np.putmask(available_region, instance_score > self.thresh, instance_score)
        to_return_boxes = []
        to_return_scores = []
        mask_region = (available_region > 0).astype(np.uint8) * 255
        structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        refined_mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, structure_element)
        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours(refined_mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        elif cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(refined_mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise NotImplementedError(f'opencv {cv2.__version__} not support')
        tmp_points = []
        tmp_socre = []
        for m_contour in contours:
            if len(m_contour) < 4 and cv2.contourArea(m_contour) < 16:
                continue
            m_rotated_box = get_min_area_bbox(refined_mask_region, m_contour, self.bbox_scale_ratio)
            if m_rotated_box is None:
                continue
            m_box_width = m_rotated_box['box_width']
            m_box_height = m_rotated_box['box_height']
            if min(m_box_width * w, m_box_height * h) < self.shortest_length:
                continue
            rotated_points = get_coordinates_of_rotated_box(m_rotated_box, height, width)
            #print("points:",rotated_points)
            # tmp_points.append(rotated_points)

            m_available_mask = np.zeros_like(available_region, dtype=np.uint8)
            cv2.drawContours(m_available_mask, [m_contour, ], 0, 255, thickness=-1)
            m_region_mask = cv2.bitwise_and(available_region, available_region, mask=m_available_mask)
            m_mask_count = np.count_nonzero(m_available_mask)
            # ???????????????????????????????????????0.89
            if (float(np.sum(m_region_mask) / m_mask_count))>0.89:
                # print('self.thresh:',self.thresh)
                # print('tmp_socre: ',float(np.sum(m_region_mask) / m_mask_count))
                tmp_socre.append(float(np.sum(m_region_mask) / m_mask_count))
                tmp_points.append(rotated_points)
                # print('tmp_socre:',tmp_socre)
                # print('tmp_points:',tmp_points)
            # tmp_socre.append(float(np.sum(m_region_mask) / m_mask_count))

        to_return_boxes.append(tmp_points)
        to_return_scores.append(tmp_socre)

        return to_return_boxes, to_return_scores


def rotate_points(_points, _degree=0, _center=(0, 0)):
    """
    ?????????????????????????????????
    Args:
        _points:    ??????????????????
        _degree:    ??????
        _center:    ?????????
    Returns:    ???????????????
    """
    angle = np.deg2rad(_degree)
    rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    center = np.atleast_2d(_center)
    points = np.atleast_2d(_points)
    return np.squeeze((rotate_matrix @ (points.T - center.T) + center.T).T)


def get_coordinates_of_rotated_box(_rotated_box, _height, _width):
    """
    ???????????????????????????????????????????????????
    Args:
        _image:     ???????????????
        _rotated_box:   ???????????????
    Returns:    ????????????????????????????????????
    """
    center_x = _rotated_box['center_x']
    center_y = _rotated_box['center_y']
    half_box_width = _rotated_box['box_width'] / 2
    half_box_height = _rotated_box['box_height'] / 2
    # ???????????????????????????????????????????????????(x1-5,y1-5),(x2+5,y2-5),(x3+5,y3+10),(x4-5,y4+10)
    raw_points = np.array([
        [center_x - half_box_width, center_y - half_box_height],
        [center_x + half_box_width, center_y - half_box_height],
        [center_x + half_box_width, center_y + half_box_height],
        [center_x - half_box_width, center_y + half_box_height]
    ]) * (_width, _height) + np.array([[-5.0,-5.0],[5.0,-5.0],[5.0,10.0],[-5.0,10.0]])
    # print("raw_points",raw_points)
    
    rotated_points = rotate_points(raw_points, _rotated_box['degree'], (center_x * _width, center_y * _height))
    rotated_points[:, 0] = np.clip(rotated_points[:, 0], a_min=0, a_max=_width)
    rotated_points[:, 1] = np.clip(rotated_points[:, 1], a_min=0, a_max=_height)
    return rotated_points.astype(np.int32)


def get_min_area_bbox(_image, _contour, _scale_ratio=1.0):
    """
    ????????????contour???????????????????????????
    note:????????????????????????????????????????????????
    Args:
        _image:     bbox????????????
        _contour:   ??????
        _scale_ratio:      ????????????
    Returns:    ?????????????????????????????????
    """
    h, w = _image.shape[:2]
    if abs(_scale_ratio -1) > 0.001:
        reshaped_contour = _contour.reshape(-1, 2)
        current_polygon = Polygon(reshaped_contour)
        distance = current_polygon.area * _scale_ratio / current_polygon.length
        offset = PyclipperOffset()
        offset.AddPath(reshaped_contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        box = offset.Execute(distance)
        if len(box) == 0 or len(box) > 1:
            return None
        scaled_contour = np.array(box).reshape(-1, 1, 2)
    else:
        scaled_contour = _contour
    try:
        rotated_box = cv2.minAreaRect(scaled_contour)
    except Exception:
        return None
    if -90 <= rotated_box[2] <= -45:
        to_rotate_degree = rotated_box[2] + 90
        bbox_height, bbox_width = rotated_box[1]
    else:
        to_rotate_degree = rotated_box[2]
        bbox_width, bbox_height = rotated_box[1]
    # ???????????????????????????????????????????????????????????????????????????
    to_return_rotated_box = {
        'degree': int(to_rotate_degree),
        'center_x': rotated_box[0][0] / w,
        'center_y': rotated_box[0][1] / h,
        'box_height': bbox_height / h,
        'box_width': bbox_width / w,
    }
    return to_return_rotated_box
