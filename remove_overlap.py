import os
import numpy as np
import cv2
import os 

class box:
    def __init__(self, id, x_left, x_right, y_top, y_bot, proba):
        self.id = id
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bot = y_bot
        self.proba = proba


# scale coor
def scale_bbox(list_bbox, w, h):
    '''
    This function is convert from yolo format to box object
    '''
    scale_list_box = []
    for bbox in list_bbox: 
        x_center, y_center, w_obj, h_obj = bbox[1] * w, bbox[2] * h, bbox[3] * w, bbox[4] * h
        scale_list_box.append(box(id=int(bbox[0]), x_left=int(x_center - w_obj/2), x_right = int(x_center + w_obj/2), y_top=int(y_center - h_obj/2), y_bot=int(y_center + h_obj/2), proba=bbox[-1]))
    return scale_list_box

def rescale_bbox(list_bbox, w, h):
    '''
    This function is convert from box object to yolo format
    '''
    lines = []
    for i in list_bbox: 
        x_center, y_center = str(float((i.x_left + i.x_right)/2/w)), str(float((i.y_top + i.y_bot)/2/h))
        w_obj, h_obj = str(float((i.x_right - i.x_left)/w)), str(float((i.y_bot - i.y_top)/h))
        line = str(i.id) + ' ' + x_center + ' ' + y_center + ' ' + w_obj + ' ' + h_obj + ' ' + str(i.proba) + '\n'
        lines.append(line)
    return lines

def iou(boxA, boxB):
    # determine intersection rectangle
    xA = max(boxA.x_left, boxB.x_left)
    yA = max(boxA.y_top, boxB.y_top)
    xB = min(boxA.x_right, boxB.x_right)
    yB = min(boxA.y_bot, boxB.y_bot)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = abs((boxA.x_right - boxA.x_left) * (boxA.y_bot - boxA.y_top))
    boxBArea = abs((boxB.x_right - boxB.x_left) * (boxB.y_bot - boxB.y_top))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_score = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou_score


def find_overlap(list_box, thresh):
    box1 = list_box[0]
    list_box = list_box[1:]
    # tim nhung box overlab voi box1
    index_overlab = []
    for i in range(len(list_box)):
        if iou(box1, list_box[i]) > thresh: 
            index_overlab.append(i)
    # lua chon box co proba cao nhat
    max = box1.proba
    max_index = 0
    final_box = None
    for i in index_overlab: 
        if list_box[i].proba > max: 
            max_index = i
            max = list_box[i].proba
   
    final_box = list_box[max_index]
    # xoa nhung box overlab voi box1 ra khoi list
    index_overlab.sort(reverse=True)
    for i in index_overlab: 
        del list_box[i]
    return list_box, final_box

def write_box(lines, txt_name):
    file = open(txt_name, 'w')
    for i in lines: 
        file.writelines(i)
    file.close()

def remove_overlap(txt_path, img_path):
    # read_txt
    txt_file = open(txt_path, 'r')
    lines = txt_file.readlines()
    lines = [list(map(float, i[:-1].split(' '))) for i in lines]
    txt_file.close()

    # read images
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # scale box
    list_box = scale_bbox(lines, w, h)

    # remove overlap
    final_list = []
    while list_box:
        list_box, final_box = find_overlap(list_box, thresh)
        final_list.append(final_box)

    # rescale_bbox
    lines = rescale_bbox(final_list, w, h)

    # write finalbox
    write_box(lines, txt_path)

thresh = 0.6
txt_path = './yolov5/runs/detect/predict_pill/labels'
img_path = './YOLOv6/public_test/pill/image'

count = 0
for i in os.listdir(txt_path):
    txt_name = txt_path + '/' + i
    img_name = img_path + '/' + i.replace('txt', 'jpg')
    print(count)
    count += 1
    remove_overlap(txt_name, img_name)