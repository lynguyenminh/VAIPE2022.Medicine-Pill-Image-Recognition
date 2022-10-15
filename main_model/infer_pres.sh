#!/bin/sh

rm -rf ./runs/detect

rm -rf ./prescription_info



python3.8 detect.py \
        --data data/data.yaml\
        --source ../public_test/prescription/image \
        --weights ../weights/detect_prescription.pt \
        --conf 0.5 \
        --save-txt \
        --save-conf


python3.8 -W ignore recognize_drugname_in_prescription.py \
        --label_yolo_txt_path   ./runs/detect/exp/labels \
        --img_path              ../public_test/prescription/image

rm -rf ./runs/detect

# using yolov5 to detect pill
python3.8 detect.py \
        --data data/data.yaml\
        --source ../public_test/pill/image \
        --weights ../weights/detect_pill.pt \
        --conf 0.35 \
        --save-txt \
        --save-conf