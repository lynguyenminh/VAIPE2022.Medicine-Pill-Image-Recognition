import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from operator import itemgetter


cudnn.benchmark = True
plt.ion()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--batchsize', type=int, required=True)
    args = parser.parse_args()
    return args

args = get_args()


# ================================================================================================
# load data for classify
data_transforms = {
    'predict': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = args.source
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['predict']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchsize,
                                             shuffle=False, num_workers=4)
              for x in ['predict']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on %s'%device)

# class_names = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '23', '24', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
# class_names = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
class_names = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
print('There are %d classname!'%len(class_names))
# ============================================================================================


# Load model efficientnetb7
def load_model_classify_efficientnetb7(weight):
    # 1. init model
    model = models.efficientnet_b7(pretrained=False)
    num_class = 107
    model.classifier = nn.Linear(2560, num_class)

    for param in model.parameters():
        param.requires_grad = False

    # 2. Load weight for model
    checkpoint = torch.load(weight, map_location=torch.device('cpu')) # if using cpu
    # checkpoint = torch.load(weight) # if using gpu
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# predict data da chuan bi phia tren
def predict(model):
    '''
    This function is using to predict pills which are predicted into class 107 by YOLO.
    '''
    phase = 'predict'
    model.eval()

    predict_list = []
    print('Start using classifier re-predict pill in class 107...')
    for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        predict_list += [class_names[i] for i in list(preds.cpu().detach().numpy())]
    return predict_list

def format_results(weight):
    # predict
    predict_list = predict(weight)

    # get all name images
    list_name_img = list(map(itemgetter(0), image_datasets['predict'].imgs))

    # convert data into (name, class)
    final_info = [(list_name_img[i].split('/')[-1], predict_list[i]) for i in range(len(predict_list))]
    return final_info

def main(csv_file, weight):
    '''
    This function is re-predict class 107 in result_init.csv and change class for it.
    '''
    # read result_init.csv
    df = pd.read_csv(csv_file)
    data = df.values

    # predict pills which are predicted into class 107 by YOLO
    list_predict = format_results(weight)

    # Thay the class 107
    for i in list_predict: 
        # Tim vi tri can thay the
        index = np.where(data == i[0])[0]
        
        # thay the
        if len(index) != 0: 
            data[index[0], 2] = i[1]


    df = pd.DataFrame(data = data[:, 1:])
    df.to_csv('results_change_label.csv', index=False, header=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])

if __name__=="__main__":
    weight = args.weights

    # load model
    model = load_model_classify_efficientnetb7(weight)
    model = model.to(device)
    print('Model is loaded successfully.')

    # change class
    main('./results_init.csv', model)
    # print('Change class successfully.')
