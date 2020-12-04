# from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
import cv2
import numpy as np

from os import listdir
from numpy import asarray
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

boletos_dir = '../teste/boletos/'
processos_dir = '../teste/processo/'

# Carrega imagens de boleto
X_boleto_labeled = list()
Y_boleto_labeled = list()
for filename in listdir(boletos_dir):
    
    image = Image.open(boletos_dir + filename)
    image_resized = image.resize((100,100)).convert('L')
    X = asarray(image_resized)
    
    # load image
    #X = image.imread('../resnet50/dataset/boletos/' + filename)
    Y = 1
    # store loaded image
    X_boleto_labeled.append(np.ravel(X))
    Y_boleto_labeled.append(Y)


# Carrega imagens de processo
X_processo_labeled = list()
Y_processo_labeled = list()
for filename in listdir(processos_dir):
    
    image = Image.open(processos_dir + filename)
    image_resized = image.resize((100,100)).convert('L')
    X = asarray(image_resized)
    
    Y = 0
    X_processo_labeled.append(np.ravel(X))
    Y_processo_labeled.append(Y)


X_train_processo, X_test_processo, y_train_processo, y_test_processo = train_test_split(X_processo_labeled, Y_processo_labeled, test_size=0.2, random_state=42)

X_train_boleto, X_test_boleto, y_train_boleto, y_test_boleto = train_test_split(X_boleto_labeled, Y_boleto_labeled, test_size=0.2, random_state=42)

X_train = X_train_processo+X_train_boleto
X_train = np.array(X_train).astype(int)
y_train = y_train_processo+y_train_boleto
y_train = np.array(y_train).astype(int)

X_test = X_test_processo+X_test_boleto
X_test = np.array(X_test).astype(int)
y_test = y_test_processo+y_test_boleto
y_test = np.array(y_test).astype(int)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

import pickle
filename = 'knn_model.pkl'
pickle.dump(model, open(filename, 'wb'))
