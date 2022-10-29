import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from util.main_check import decision_in_out

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pres_name_txt_path', type=str, required=True)
    parser.add_argument('--pill_prescription_map_path', type=str, required=True)
    parser.add_argument('--score', type=float, required=True)
    args = parser.parse_args()
    return args


# Doc thong tin results_change_label.csv
def read_results():
    df = pd.read_csv('./results_change_label.csv')
    return df

# Doc thong tin mapping id and drugname
def read_mapping_id():
    with open('../data/mapping_standard.json') as f: 
        data = json.load(f)
    return data

# Doc thong tin mapping pill and prescription
def read_mapping_pill():
    with open(args.pill_prescription_map_path) as f: 
        data = json.load(f)
    return data


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
                state = decision_in_out(name_drug, vietocr_prescription, args.score)
                if state == 0: # Neu ko thuoc
                    line[1] = 107

                final_array = np.concatenate((final_array, line.reshape(1, -1)), axis=0)

    # 4. Save result to csv
    df = pd.DataFrame(data = final_array[1:, :])
    df.to_csv('results.csv', index=False, header=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])

    print('Complete.')