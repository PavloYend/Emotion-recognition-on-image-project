import torch
import torchvision.transforms as transforms
import numpy as np
from model import *


class Predictor():

    def __init__(self, model_path):
        self.model = EmotionClassifier_v1()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()


    def __call__(self, picture):
        pic = self.transform(picture)

        # print(np.shape(pic))
        
        with torch.no_grad():
            pred = self.model(pic)

        pic = transforms.ToPILImage()(pic)
        result = dict(zip(self.model.emotions, [float(_) for _ in pred]))

        return (result, pic)


    def transform(self, picture):
        pic = picture
        pic = transforms.CenterCrop(min(np.shape(pic)[0], np.shape(pic)[0]))(pic)
        pic = transforms.Resize(self.model.pic_size)(pic)
        pic = transforms.Grayscale()(pic)
        pic = transforms.ToTensor()(pic)
        # pic = transforms.Normalize(mean=[0.5], std=[0.9])(pic)

        return pic
