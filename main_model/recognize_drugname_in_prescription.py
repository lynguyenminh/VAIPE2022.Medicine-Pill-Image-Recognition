import cv2
import os
import argparse
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from tqdm import tqdm
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_yolo_txt_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    args = parser.parse_args()
    return args


def predict_text(img):
    opencv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR to RGB
    pil_image = Image.fromarray(opencv_image_rgb) # convert numpy array to Pillow Image Object
    text, prob = detector.predict(pil_image , return_prob = True)
    return text

def post_process_text(text):
    if text == '':
        return ''

    # chuyen thanh chu viet thuong
    text = text.lower()

    # phat hien nhan dien sai
    for i in ['chẩn đoán', 'bệnh', 'chứng', 'sl:', 'lời dặn', 'bố', 'mẹ', 'sáng', 'trưa', 'tối', 'đoán', 'tái khám', 'khám lại', 'xơ vữa động mạch', 'uống trước khi ăn', 'uống sau khi ăn', 'viêm họng cấp', 'tìnhtrạng', 'tình trạng']:
        if i in text:
            return ''
    # xoa ngoac kep (dau hoac cuoi)
    text = text[1:] if text[0] == '"' else text
    text = text[:-1] if text[-1] == '"' else text

    # xoa so thu tu
    # ex: '1) GLUCOFAST 500 500mg'
    temp = text.split(' ')
    if temp[0][-1] == ')':
        text= text.replace(temp[0], '')

    # xoa cach va cac ki tu dat biet
    for i in [',', '.', '+', '-', '%', ' ', '\'', '(', ')']:
        text = text.replace(i,'')

    text = text.replace('dường', 'dưỡng')

    for i in text: 
        if i.isnumeric() or i == ')': 
            text = text[1:]
        else: 
            break

    return text

def predict_vietocr(label_yolo_txt_name, img_name):
    # 1, Read info from predict yolo
    list_bbox = []
    with open(label_yolo_txt_name, 'r') as f: 
        list_bbox = [i for i in f.readlines() if i[0] == '1']

    # 2. read image
    img = cv2.imread(img_name)
    h, w, _ = img.shape

    # 3. open new txt file, predict and write result
    new_file = open(''.join(['./prescription_info/', label_yolo_txt_name.split('/')[-1]]), 'w')
    for i in list_bbox:
        bbox = i.split(' ')
        x_center, y_center = int(float(bbox[1]) * w), int(float(bbox[2]) * h)
        sub_w, sub_h = int(float(bbox[3]) * w), int(float(bbox[4]) * h)
        proba = float(bbox[-1])
        # 3.1. Crop image
        sub_img = img[y_center - int(sub_h/2):y_center + int(sub_h/2), x_center - int(sub_w /2):x_center+int(sub_w/2)]
        # 3.2. predict and post_processing text
        result_text = post_process_text(predict_text(sub_img))
        # 3.3. Write into file
        if result_text == '':
            continue
        new_file.writelines(result_text + '\n')

    new_file.close()

if __name__=="__main__":
    args = get_args()
    label_yolo_txt_path = args.label_yolo_txt_path
    img_path = args.img_path

    # Load model vietocr
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained']=False
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['predictor']['beamsearch']=False
    detector = Predictor(config)

    # create folder save prescription info
    if not os.path.isdir('./prescription_info/'):
        os.makedirs('./prescription_info/')

    list_file = os.listdir(label_yolo_txt_path)
    list_file.sort()

    for i in tqdm(list_file):
        txt_name = label_yolo_txt_path + '/' + i
        img_name = img_path + '/' + i.replace('txt', 'png')
        predict_vietocr(txt_name, img_name)
