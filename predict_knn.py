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

class KNNPredictor:
    
    def __init__(self, image_input_dir, model_knn_dir):
        
        self.image_input_dir = image_input_dir
        self.model_knn_dir =  model_knn_dir
            
        self.knn_model = self.load_knn_model()

    def load_knn_model(self):
        print('Loading KNN model...')
        return pickle.load(open(self.model_knn_dir+'knn_model.pkl', 'rb'))


    def predict_images(self):
        for filename in os.listdir(self.image_input_dir):            
            
            try:
                file = self.image_input_dir + filename
                image = pil_image.open(file)

            
                image_resized = image.resize((100,100)).convert('L')
                X = np.ravel(asarray(image_resized))
                knn_predict = self.knn_model.predict_proba(X.reshape(1, -1))
                if knn_predict[0][1] >= 0.5:
                    pass
                else:
                    if knn_predict[0][1] >= 0.3:
                        pass
                    
            except Exception as e:
                print('erro: '+str(e))


                
if __name__ == '__main__':

    print('Executando predict knn')

    start_time = time.time()

    image_input_dir = '../teste/processo/'
    model_knn_dir = './'
    
    collector = KNNPredictor(image_input_dir, model_knn_dir)

    collector.predict_images()

    print("--- %s seconds ---" % (time.time() - start_time))