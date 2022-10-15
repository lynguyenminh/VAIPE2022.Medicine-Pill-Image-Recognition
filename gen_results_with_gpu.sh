#!/bin/sh
# 1. Using yolov5 to predict
cd main_model
rm -rf ./runs/detect

rm -rf ./prescription_info

# running on prescription
CUDA_VISIBLE_DEVICES="1,2,3" \
python3.8 detect.py \
        --data data/data.yaml\
        --source ../public_test/prescription/image \
        --weights ../weights/detect_prescription.pt \
        --conf 0.5 \
        --save-txt \
        --save-conf

# using vietocr
python3.8 -W ignore recognize_drugname_in_prescription.py \
        --label_yolo_txt_path   ./runs/detect/exp/labels \
        --img_path              ../public_test/prescription/image

rm -rf ./runs/detect

# running on pill
CUDA_VISIBLE_DEVICES="1,2,3" \
python3.8 detect.py \
        --data data/data.yaml\
        --source ../public_test/pill/image \
        --weights ../weights/detect_pill.pt \
        --conf 0.35 \
        --save-txt \
        --save-conf


# 2. post-process and write result
cd ..

rm -rf predict
echo "Remove crop image successfully"

# write result from yolo result
rm results_init.csv
python3.8 init_result_from_YOLO_prediction.py \
        --txt_path ./main_model/runs/detect/exp/labels \
        --img_path ./public_test/pill/image

# re-predict pill using EfficientNetB7

rm results_change_label.csv
CUDA_VISIBLE_DEVICES="1,2,3" \
python3.8 -W ignore change_class.py \
        --weights   ./weights/classify_pill.pt \
        --batchsize 8 \
        --source    '.'

# using description to find off-prescription and write final result
rm results.csv
python3.8 detect_off-prescription_pill.py\
        --pres_name_txt_path ./main_model/prescription_info\
        --pill_prescription_map_path ./public_test/pill_pres_map.json\
        --score 0.88
