#!/bin/sh
GPU_FREE=6

rm -rf ./runs/detect

rm -rf ./prescription_info

using yolov5 to predict prescription
CUDA_VISIBLE_DEVICES=$GPU_FREE \
	python3.8 detect.py \
        --data data/data.yaml\
        --source ../public_test/prescription/image \
        --weights ../weights/detect_prescription.pt \
        --conf 0.5 \
        --save-txt \
        --save-conf



# using vietocr to reg name drug in prescription
CUDA_VISIBLE_DEVICES=$GPU_FREE \
    python3.8 -W ignore recognize_drugname_in_prescription.py \
        --label_yolo_txt_path   ./runs/detect/exp/labels \
        --img_path              ../public_test/prescription/image



rm -rf ./runs/detect

# using yolov5 to detect pill
CUDA_VISIBLE_DEVICES=$GPU_FREE \
	python3.8 detect.py \
        --data data/data.yaml\
        --source ../public_test/pill/image \
        --weights ../weights/detect_pill.pt \
        --conf 0.35 \
        --save-txt \
        --save-conf

