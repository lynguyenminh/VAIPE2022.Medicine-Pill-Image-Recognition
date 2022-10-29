rm -rf results/*

cd ./src/model/detect_model


# # xoa kq predict cu
rm -rf ./runs

# # detect pill
python3.8 detect.py \
        --data data/data.yaml\
        --source ../../../data/testset/pill/image \
        --weights ../../../models/detect_pill.pt \
        --conf 0.35 \
        --save-txt \
        --save-conf

# move result ra cho khac
mv ./runs/detect/exp ../../../results
mv ../../../results/exp ../../../results/detect_pill

# xoa kq predict cu
rm -rf ./runs


#  detect pres
python3.8 detect.py \
        --data data/data.yaml\
        --source ../../../data/testset/prescription/image \
        --weights ../../../models/detect_prescription.pt \
        --conf 0.5 \
        --save-txt \
        --save-conf


# move result ra cho khac
mv ./runs/detect/exp ../../../results
mv ../../../results/exp ../../../results/detect_pres


# uisng vietocr
cd ../vietocr_model
python3.8 -W ignore recognize_drugname_in_prescription.py \
        --label_yolo_txt_path   ../../../results/detect_pres/labels \
        --img_path              ../../../data/testset/prescription/image \
        --predict_vietocr_path  ../../../results/prescription_info

