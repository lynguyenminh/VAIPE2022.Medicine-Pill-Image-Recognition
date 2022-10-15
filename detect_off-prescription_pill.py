import json
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import argparse
import re, string
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pres_name_txt_path', type=str, required=True)
    parser.add_argument('--pill_prescription_map_path', type=str, required=True)
    parser.add_argument('--score', type=float, required=True)
    args = parser.parse_args()
    return args


def clean_drugname(drug_name):
    drug_name = drug_name.lower() #lowercase drug_name
    drug_name=drug_name.strip()  #get rid of leading/trailing whitespace 
    drug_name = re.compile('[%s]' % re.escape(string.punctuation)).sub('', drug_name)  #Replace punctuation with space. Careful since punctuation can sometime be useful
    
    drug_name=re.sub(r'[^\w\s]', '', str(drug_name).lower().strip())
    drug_name = re.sub(r'\s+',' ',drug_name) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    
    # replace 0.5g -> 500mg
    for i in range(1, 9):
        drug_name = drug_name.replace(''.join(['0', str(i), 'g']), ''.join([str(i), '00mg']))
    
    # replace 0.45g -> 450mg
    for i in range(10, 99):
        drug_name = drug_name.replace('0' + str(i) + 'g', str(i * 10) + 'mg')
    
    # replace o.5g -> 500mg
    for i in range(1, 9):
        drug_name = drug_name.replace('o' + str(i) + 'g', str(i) + '00mg')
    
    # replace 500 500mg -> 500mg
    for i in ['500mg', '1000mg', '500', '1000', '250', '125', '850', '800', '250mg', '850mg', '800mg', '300mg', '300', '20mg', '10mg', '200mg', '30mg', '30', '60mg', '125mg', '25mg', '145mg', '16mg', '4mg', '175mg', '175', '16', '4', '145', '25', '60', '200', '20', '10']:
        if drug_name.count(i) > 1: 
            drug_name = drug_name.replace(i, '', 1)

    drug_name = re.sub('\s+', ' ', drug_name)  #Remove extra space and tabs
    drug_name=re.sub(r'[^\w\s]', '', str(drug_name).lower().strip())
    drug_name = re.sub(r'\s+',' ',drug_name) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 

    return drug_name

# score of 2 strings
def similar(a, b):
    a = clean_drugname(a)
    b = clean_drugname(b)
    return SequenceMatcher(None, a, b).ratio()

# Doc thong tin results_change_label.csv
def read_results():
    df = pd.read_csv('./results_change_label.csv')
    return df

# Doc thong tin mapping id and drugname
def read_mapping_id():
    with open('./mapping_standard.json') as f: 
        data = json.load(f)
    return data

# Doc thong tin mapping pill and prescription
def read_mapping_pill():
    with open(args.pill_prescription_map_path) as f: 
        data = json.load(f)
    return data


# kiem tra 1 ten thuoc co thuoc danh sach ten thuoc trong prescription khong
def check_pill_in_pres(pill_name, name_in_prescription):
    '''
    This function check pill in prescription by calculate similar drugname.
    '''
    for i in name_in_prescription: 
        if similar(pill_name, i) > args.score: 
            return True
    return False

# kiem tra xem danh sach ten thuoc co thuoc prescription ko
def decision_in_out(name_in_pill, name_in_prescription):
    count = [check_pill_in_pres(i, name_in_prescription) for i in name_in_pill]
    return 0 if count == 0 else 1

# print(decision_in_out(['amoxicilin500mg500mg', 'fabamox500500mg', 'novoxim50005g'], ['novoxim50005gtru', 'kavasdin55mg', 'mypara500500mg']))

if __name__=="__main__":
    args = get_args()
    print('Start using prescription info...')
    # 1. Doc thong tin co ban
    results = read_results()
    id_name_mapping = read_mapping_id()
    pill_prescription_mapping = read_mapping_pill()

    # 2. Tao final_array save final data
    final_array = np.arange(7).reshape(1, -1)

    # 3. Check tung anh xem pill co thuoc prescription ko
    for prescription in tqdm(pill_prescription_mapping): 
        # 3.1. Doc path cua prescription va image tuong ung
        prescription_path = prescription.get('pres')
        pill_path = prescription.get('pill')

        # 3.2. Doc thong tin don thuoc ma vietocr da predict
        with open(''.join([args.pres_name_txt_path, '/', prescription_path, '.txt'])) as f:
            vietocr_prescription = [i[:-1] for i in f.readlines()]

        # 3.3. Fix error pill class
        for i in pill_path: 
            # 3.3.1. lay ra thong tin cac anh thuoc prescription dang xet
            sub_df = results[results['image_name'] == i + '.jpg'].values

            # 3.3.2. Kiem tra thong tin va sua doi
            for line in sub_df:
                # 3.3.2.1. Xac dinh cac ten thuoc co the co voi id
                id = line[1]
                name_drug = id_name_mapping.get(str(id))

                # 3.3.2.2.Danh gia xem thuoc co thuoc don ko
                state = decision_in_out(name_drug, vietocr_prescription)
                if state == 0: # Neu ko thuoc
                    line[1] = 107

                final_array = np.concatenate((final_array, line.reshape(1, -1)), axis=0)

    # 4. Save result to csv
    df = pd.DataFrame(data = final_array[1:, :])
    df.to_csv('results.csv', index=False, header=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])

    print('Complete.')