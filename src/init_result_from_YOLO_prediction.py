import cv2
import numpy as np
import os
import pandas as pd
import argparse
from tqdm import tqdm
from time import time



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--crop_img_path', type=str, required=True)
    args = parser.parse_args()
    return args

def read_label_from_yolo(txt_path):
    '''
    This function read and convert info from label that predict by YOLo.
    '''
    with open(txt_path, 'r') as file: 
        lines = file.readlines()
        lines = [line[:-1].split(' ') for line in lines]
        lines = [list(map(float, line)) for line in lines]
    return lines

def main(final_matrix, txt_path, img_path):
    # 1. Read label yolo
    txt_value = read_label_from_yolo(txt_path)

    # 2. Read w, h from image
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # 3. Create folder to saving crop images
    if not os.path.isdir(crop_img_path):
        os.makedirs(crop_img_path)

    id_crop = 0 # id for crop image
    for i in txt_value: 
        name_img = img_path.split('/')[-1]
        class_id, propa = int(i[0]), i[5]
        x_center, y_center, w_object, h_object = i[1] * w, i[2] * h, i[3] * w, i[4] * h

        # Find top_left 
        x_top_left, y_top_left = int(x_center - 0.5 * w_object), int(y_center - 0.5 * h_object)

        # Find bot_right
        x_bot_right, y_bot_right = int(x_center + 0.5 * w_object), int(y_center + 0.5 * h_object)

        # if class = 107 then crop pill and re-predict
        sub_img_name = ''.join([crop_img_path, '/', str(id_crop), name_img])
        if class_id == 107: 
            sub_img = img[y_top_left:y_bot_right, x_top_left:x_bot_right]
            cv2.imwrite(sub_img_name, sub_img)

        
        final_matrix = np.concatenate((final_matrix, np.array([sub_img_name.split('/')[-1], name_img, class_id, propa, x_top_left, y_top_left, x_bot_right, y_bot_right]).reshape(1, -1)), axis=0)
        id_crop += 1
    return final_matrix

if __name__=="__main__":
    start_time = time()
    print('Starting step 1...')

    # get args
    args = get_args()
    txt_path = args.txt_path
    img_path = args.img_path
    crop_img_path = args.crop_img_path

    # Create matrix to saving all info
    final_matrix = np.arange(8).reshape(1, -1)

    # main handle
    for i in tqdm(os.listdir(txt_path)):
        sub_txt_path = ''.join([txt_path ,'/', i])
        sub_img_path = ''.join([img_path, '/', i.replace('txt', 'jpg')])
        final_matrix = main(final_matrix, sub_txt_path, sub_img_path)
    
    df = pd.DataFrame(data = final_matrix[1:, :])
    df.to_csv('results_init.csv', index=False, header=['sub_img_name', 'image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
    print('Completed predict using YOLO prediction. Total time is: %s s'%(round((time() - start_time), 2)))