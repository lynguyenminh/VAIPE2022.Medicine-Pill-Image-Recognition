import cv2
import os
import argparse
from tqdm import tqdm

from predict import load_model_vietocr, predict_text

import sys
sys.path.append('../../../src')
from util.clean_string import post_process_text


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_yolo_txt_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--predict_vietocr_path', type=str, required=True)
    args = parser.parse_args()
    return args


def predict_vietocr_prescription(label_yolo_txt_name, img_name):
    # 1, Read info from predict yolo
    list_bbox = []
    with open(label_yolo_txt_name, 'r') as f: 
        list_bbox = [i for i in f.readlines() if i[0] == '1']

    # 2. read image
    img = cv2.imread(img_name)
    h, w, _ = img.shape

    # 3. open new txt file, predict and write result
    new_file = open('/'.join([predict_vietocr_path, label_yolo_txt_name.split('/')[-1]]), 'w')
    for i in list_bbox:
        bbox = i.split(' ')
        x_center, y_center = int(float(bbox[1]) * w), int(float(bbox[2]) * h)
        sub_w, sub_h = int(float(bbox[3]) * w), int(float(bbox[4]) * h)
        # 3.1. Crop image
        sub_img = img[y_center - int(sub_h/2):y_center + int(sub_h/2), x_center - int(sub_w /2):x_center+int(sub_w/2)]
        # 3.2. predict and post_processing text
        result_text = post_process_text(predict_text(sub_img, detector))
        # 3.3. Write into file
        if result_text == '':
            continue
        new_file.writelines(result_text + '\n')

    new_file.close()

if __name__=="__main__":
    args = get_args()
    label_yolo_txt_path = args.label_yolo_txt_path
    img_path = args.img_path
    predict_vietocr_path = args.predict_vietocr_path

    # Load model vietocr
    detector = load_model_vietocr()

    # create folder save prescription info
    if not os.path.isdir(predict_vietocr_path):
        os.makedirs(predict_vietocr_path)

    list_file = os.listdir(label_yolo_txt_path)
    list_file.sort()

    for i in tqdm(list_file):
        txt_name = label_yolo_txt_path + '/' + i
        img_name = img_path + '/' + i.replace('txt', 'png')
        predict_vietocr_prescription(txt_name, img_name)
