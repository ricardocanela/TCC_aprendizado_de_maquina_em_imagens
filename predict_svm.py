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

class SVMPredictor:
    
    def __init__(self, image_input_dir, model_svm_dir):
        
        self.image_input_dir = image_input_dir
        self.model_svm_dir =  model_svm_dir
            
        self.svm_model = self.load_svm_model()

    def load_svm_model(self):
        print('Loading SVM model...')
        return pickle.load(open(self.model_svm_dir+'svm_model.pkl', 'rb'))


    def predict_images(self):
        for filename in os.listdir(self.image_input_dir):            
            
            try:
                file = self.image_input_dir + filename
                image = pil_image.open(file)

            
                image_resized = image.resize((100,100)).convert('L')
                X = np.ravel(asarray(image_resized))
                svm_predict = self.svm_model.predict_proba(X.reshape(1, -1))
                if svm_predict[0][1] >= 0.5:
                    pass
                else:
                    if svm_predict[0][1] >= 0.3:
                        pass
                    
            except Exception as e:
                print('erro: '+str(e))


                
if __name__ == '__main__':

    print('Executando predict svm')

    start_time = time.time()

    image_input_dir = '../teste/processo/'
    model_svm_dir = './'
    
    collector = SVMPredictor(image_input_dir, model_svm_dir)

    collector.predict_images()

    print("--- %s seconds ---" % (time.time() - start_time))