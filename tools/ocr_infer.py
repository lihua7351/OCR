from det_infer import DetInfer
from rec_infer import RecInfer
import argparse
import time
from line_profiler import LineProfiler
from memory_profiler import profile
from torchocr.utils.vis import draw_ocr_box_txt
import numpy as np
import os
import cv2

def get_files(img_dir):
    imgs = list_files(img_dir)
    return imgs

def list_files(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
    return img_files


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    points = points.astype(np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class OCRInfer(object):
    def __init__(self, det_path, rec_path, rec_batch_size=16, time_profile=False, mem_profile=False ,**kwargs):
        super().__init__()
        self.det_model = DetInfer(det_path)
        self.rec_model = RecInfer(rec_path, rec_batch_size)
        assert not(time_profile and mem_profile),"can not profile memory and time at the same time"
        self.line_profiler = None
        if time_profile:
            self.line_profiler = LineProfiler()
            self.predict = self.predict_time_profile
        if mem_profile:
            self.predict = self.predict_mem_profile

    def do_predict(self, img):
        # started = time.time()
        box_list, score_list = self.det_model.predict(img)
        # finished = time.time()
        # print('det_inference time: {0}'.format(finished - started))
        if len(box_list) == 0:
            return [], [], img
        draw_box_list = [tuple(map(tuple, box)) for box in box_list]
        imgs =[get_rotate_crop_image(img, box) for box in box_list]
        # started = time.time()
        texts = self.rec_model.predict(imgs)
        # finished = time.time()
        # print('rec inference time: {0}'.format(finished - started))
        # print(texts)
        texts = [txt[0][0] for txt in texts]
        # print(texts)
        debug_img = draw_ocr_box_txt(img, draw_box_list, texts)
        return box_list, score_list, debug_img

    def predict(self, img):
        return self.do_predict(img)

    def predict_mem_profile(self, img):
        wapper = profile(self.do_predict)
        return wapper(img)

    def predict_time_profile(self, img):
        # run multi time
        for i in range(8):
            print("*********** {} profile time *************".format(i))
            lp = LineProfiler()
            lp_wrapper = lp(self.do_predict)
            ret = lp_wrapper(img)
            lp.print_stats()
        return ret


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='OCR infer')
    parser.add_argument('--det_path', required=True, type=str, help='det model path')
    parser.add_argument('--rec_path', required=True, type=str, help='rec model path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    parser.add_argument('--rec_batch_size', type=int, help='rec batch_size', default=16)
    parser.add_argument('-time_profile', action='store_true', help='enable time profile mode')
    parser.add_argument('-mem_profile', action='store_true', help='enable memory profile mode')
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    # import cv2
    # args = init_args()
    # img = cv2.imread(args['img_path'])
    # model = OCRInfer(**args)
    # txts, boxes, debug_img = model.predict(img)

## 批量处理
    args = init_args()
    image_list = get_files(args['img_path'])
    for k, image_path in enumerate(image_list):
        started = time.time()
        img = cv2.imread(image_path)
        res_path = image_path[:-4] +"{}.jpg".format(k)
        model = OCRInfer(**args)
        # started = time.time()
        txts, boxes, debug_img = model.predict(img)
        finished = time.time()
        print('ocr_inference time: {0}'.format(finished - started))
        h,w,_, = debug_img.shape
        raido = 1.0
        # if w > 1200:
        #     raido = 600.0/w
        debug_img = cv2.resize(debug_img, (int(w*raido), int(h*raido)))
        # debug_img = cv2.resize(debug_img, w, h)

        if not(args['mem_profile'] or args['time_profile']):
            cv2.imwrite(res_path, debug_img)

    # out = model.predict(img)
    # print(out)
    # h,w,_, = debug_img.shape
    # raido = 1
    # if w > 1200:
    #     raido = 600.0/w
    # debug_img = cv2.resize(debug_img, (int(w*raido), int(h*raido)))
    # if not(args['mem_profile'] or args['time_profile']):
    #     cv2.imshow("debug", debug_img)
    #     cv2.waitKey()

