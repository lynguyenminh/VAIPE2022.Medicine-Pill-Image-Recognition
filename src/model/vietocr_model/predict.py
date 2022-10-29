import cv2
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch


def load_model_vietocr():
     # Load model vietocr
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained']=False
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['predictor']['beamsearch']=False
    detector = Predictor(config)
    return detector

def predict_text(img, detector):
    opencv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR to RGB
    pil_image = Image.fromarray(opencv_image_rgb) # convert numpy array to Pillow Image Object
    text, prob = detector.predict(pil_image , return_prob = True)
    return text