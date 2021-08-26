# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib
import time

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchvision import transforms
from torchocr.networks import build_model
from torchocr.datasets.det_modules import ResizeFixedSize
from torchocr.postprocess import build_post_process


class DetInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.resize = ResizeFixedSize(224, False)
        self.post_process = build_post_process(cfg['post_process'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['train']['dataset']['mean'], std=cfg['dataset']['train']['dataset']['std'])
        ])

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        out = out.cpu().numpy()
        box_list, score_list = self.post_process(out, data['shape'])
        # print(box_list, score_list)
        box_list, score_list = box_list[0], score_list[0]
        # print(box_list)
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    from torchocr.utils import draw_ocr_box_txt, draw_bbox

    args = init_args()
    img = cv2.imread(args.img_path)
    model = DetInfer(args.model_path)
    started = time.time()
    box_list, score_list = model.predict(img, is_output_polygon=False)
    finished = time.time()
    print('det inference time: {0}'.format(finished - started))
    # print(box_list)
    raw_path = args.img_path
    # res_path = raw_path[:-3] + "jpg"
    # print(res_path)
    for i in range(len(box_list)):
        print(box_list[i])
        res_path = raw_path[:-4] +"{}.jpg".format(i)
        print(res_path)
        print(box_list[i][0])
        x1 = box_list[i][0][0] - 5
        y1 = box_list[i][0][1] - 5
        x2 = box_list[i][2][0] + 5
        y2 = box_list[i][2][1] + 5
        cv2.imwrite(res_path,img[y1:y2,x1:x2])
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = draw_bbox(img, box_list)
    # plt.imshow(img)
    # plt.show()
