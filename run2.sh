cd ./src

rm -rf ../results/crop_107_class

# # write result from yolo result
python3.8 init_result_from_YOLO_prediction.py \
        --txt_path ../results/detect_pill/labels \
        --img_path ../data/testset/pill/image \
        --crop_img_path ../results/crop_107_class/1


# re-predict pill using EfficientNetB7
python3.8 -W ignore change_class.py \
        --weights   ../models/classify_pill.pt \
        --batchsize 8 \
        --crop_img_path    ../results


python3.8 detect_off-prescription_pill.py\
        --pres_name_txt_path ../results/prescription_info \
        --pill_prescription_map_path ../data/testset/pill_pres_map.json\
        --score 0.88



mv results.csv ../results
rm *.csv