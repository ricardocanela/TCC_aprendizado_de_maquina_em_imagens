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

class VGGPredictor:
    
    def __init__(self, image_input_dir, model_dir):
        
        self.image_input_dir = image_input_dir
        self.model_dir =  model_dir 
            
        self.vgg_model = self.load_vgg_model()

    def load_vgg_model(self):
        print('Loading VGG16 model...')
        return load_learner(self.model_dir)


    def predict_images(self):
        for filename in os.listdir(self.image_input_dir):            
            

            try:
                file = self.image_input_dir + filename
                image = pil_image.open(file)

            
                # ResNet
                img_tensor = T.ToTensor()(image)
                img = Image(img_tensor)
                x = self.vgg_model.predict(img)
                if x[1].item() == 0:
                    pass
                else:
                    pass
                    
            except Exception as e:
                print('erro: '+str(e))


                
if __name__ == '__main__':

    print('Executando predict vgg16')

    start_time = time.time()

    image_input_dir = '../teste/processo/'
    model_dir = './'
    
    collector = VGGPredictor(image_input_dir, model_dir)

    collector.predict_images()


    print("--- %s seconds ---" % (time.time() - start_time))
