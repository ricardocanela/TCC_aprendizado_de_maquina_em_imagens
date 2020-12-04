import os, sys
import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')
from fastai import *
from fastai.vision import *
import torchvision.transforms as T
import pickle
from numpy import asarray
from PIL import Image as pil_image
import time

class ResNetPredictor:
    
    def __init__(self, image_input_dir, model_resnet_dir):
        
        self.image_input_dir = image_input_dir
        self.model_resnet_dir =  model_resnet_dir
            
        self.resnet_model = self.load_resnet_model()

    def load_resnet_model(self):
        print('Loading RESNET model...')
        return load_learner(self.model_resnet_dir)


    def predict_images(self):
        for filename in os.listdir(self.image_input_dir):            
            

            try:
                file = self.image_input_dir + filename
                image = pil_image.open(file)

            
                # ResNet
                img_tensor = T.ToTensor()(image)
                img = Image(img_tensor)
                x = self.resnet_model.predict(img)
                if x[1].item() == 0:
                    pass
                else:
                    pass
                    
            except Exception as e:
                print('erro: '+str(e))


                
if __name__ == '__main__':

    print('Executando predict resnet')

    start_time = time.time()

    image_input_dir = '../teste/processo/'
    model_resnet_dir = './'
    
    collector = ResNetPredictor(image_input_dir, model_resnet_dir)

    collector.predict_images()


    print("--- %s seconds ---" % (time.time() - start_time))
