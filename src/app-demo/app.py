import streamlit as st
import os
import json
from pathlib import Path
import base64
import cv2
from PIL import Image
import torch


import sys
sys.path.append('../../src')
from util.clean_string import post_process_text
from model.classify_model.predict import predict_img
from model.vietocr_model.predict import load_model_vietocr, predict_text
from util.main_check import similar

THRESHOLD_DETECT_PRES = 0.9
THRESHOLD_SIMILAR_TEXT = 0.8
CLASS_NAMES = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']



# Load model detect pill and detect presctiption
model_pres = torch.hub.load('../model/detect_model', 'custom', path='../../models/detect_prescription.pt', source='local') 
model_pill = torch.hub.load('../model/detect_model', 'custom', path='../../models/detect_pill.pt', source='local') 

# Load model vietocr
detector = load_model_vietocr()


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def check_pill_in_pres(name_pill, list_pres):
    for i in name_pill: 
        list_score = [similar(i, a) for a in list_pres]
        if max(list_score) > THRESHOLD_SIMILAR_TEXT: 
            return True, i
    return False, "Undefined"

if __name__=="__main__":
    st.markdown('''
        # <center>VAIPE: Medicine Pill Image Recognition Challenge</center>
        ''', unsafe_allow_html=True
    )

    pill_uploaded_file = st.file_uploader("Upload pill image")
    pres_uploaded_file = st.file_uploader("Upload prescription image")

    button = st.button('Predict')
    if button: 
        if pill_uploaded_file is None or pres_uploaded_file is None:
            st.write('Please upload 2 image!!!')
        else:
            # 1. UPLOAD IMAGES
            upload_path = 'images'
            if not os.path.isdir(upload_path): 
                os.makedirs(upload_path)

            # 1.1. SAVE IMAGES
            with open(os.path.join("images",pill_uploaded_file.name),"wb") as f:
                f.write(pill_uploaded_file.getbuffer())
            with open(os.path.join("images",pres_uploaded_file.name),"wb") as f:
                f.write(pres_uploaded_file.getbuffer())

            # 1.2. SHOW IMAGES
            pill_image = Image.open(pill_uploaded_file)
            pres_image = Image.open(pres_uploaded_file)
            cols = st.columns(2)
            cols[0].image(pill_image, use_column_width=True, caption='Pill image')
            cols[1].image(pres_image, use_column_width=True, caption='Prescription image')


            # 2. READ INFO FROM PRESCRIPTION UISNG YOLOV5
            list_pres = []
            img_pres = os.path.join("images",pres_uploaded_file.name)
            image_matrix = cv2.imread(img_pres)
            result_pres = model_pres(img_pres)
            result_pres = result_pres.pandas().xyxy[0]

            for index, row in result_pres.iterrows():
                if row['class'] == 1 and row['confidence'] > THRESHOLD_DETECT_PRES: 
                    sub_img = image_matrix[int(row['ymin']): int(row['ymax']), int(row['xmin']):int(row['xmax'])]
                    # using VietOCR recognize text
                    text = predict_text(sub_img, detector)
                    text = post_process_text(text)
                    list_pres.append(text)

            # 3. PREDICT PILL USING YOLOV5
            img_pill = os.path.join("images",pill_uploaded_file.name)
            image_pill_matrix = cv2.imread(img_pill)
            result_pill = model_pill(img_pill)
            result_pill = result_pill.pandas().xyxy[0]

            count_sub_img = 0 # count num crop image
            list_pill = [] # save list sub_image info
            list_class = []
            for index, row in result_pill.iterrows():
                # 3.1. CROP IMAGES AND SAVE
                sub_img = image_pill_matrix[int(row['ymin']): int(row['ymax']), int(row['xmin']):int(row['xmax'])]
                img_name  = './images/sub_img_%s.jpg' % str(count_sub_img)
                count_sub_img += 1
                cv2.imwrite(img_name, sub_img)

                # 3.2. PREDICT CLASS 107 AGAIN
                row['class'] = predict_img(weight='./weights/classify_pill.pt', img=cv2.imread(img_name)) if row['class'] == 107 else row['class']

                # 3.3. WRITE INFO
                if row['class'] not in list_class:
                    dict = {}
                    dict['sub-img'] = img_name
                    dict['confidence'] = round(row['confidence'] * 100, 2)
                    dict['class'] = row['class']
                    dict['num'] = 1
                    list_pill.append(dict)
                    list_class.append(row['class'])
                else: 
                    for t, element in enumerate(list_pill): 
                        if element['class'] == row['class']:
                            list_pill[t]['num'] += 1
                

            # 4. MAPPPING RESULTS
            for i, element in enumerate(list_pill): 
                # 4.1. GET NAME OF PILL
                data = None
                with open('../../data/mapping_standard.json') as f: 
                    data = json.load(f)
                list_name_pill_possible = data[str(element['class'])]

                # MAPPING
                list_pill[i]['in'], list_pill[i]['name'] = check_pill_in_pres(list_name_pill_possible, list_pres)
                

            # 5. DISPLAY RESULT
            table = "| Pill image  | ID pill | Name pill | Num pill | Confidence (%) | In prescription |\n| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|\n"
            for element in list_pill: 
                header_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='80px' height='auto'>".format(img_to_bytes(element['sub-img']))
                table += "| %s | %s | %s | %s | %s | %s |\n" % (header_html, element['class'], element['name'], element['num'], element['confidence'], element['in'])

            st.markdown(table, unsafe_allow_html=True)
            print(list_pres)