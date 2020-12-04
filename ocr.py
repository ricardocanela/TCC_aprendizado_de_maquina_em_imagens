import pytesseract
from PIL import Image
import os, re
import shutil
import cv2
import jellyfish as jf

def rotate(image, center = None, scale = 1.0):
    angle=360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def word_distance(text):
    for word in text.split(' '):
        if jf.levenshtein_distance(word, 'extraordinario:') <= 4:
            print(word)
            return True
    return False


config = '--psm 6'

for page in os.listdir('./boletos'):
    image = cv2.imread('./boletos/'+page)
    text = pytesseract.image_to_string(image).lower()
    text_config = pytesseract.image_to_string(image, config=config).lower()
    if word_distance(text):
        print('found: '+page)
        shutil.copy2('./boletos/'+page,'./RE/'+page)
    elif word_distance(text_config):
        print('found config: '+page)
        shutil.copy2('./boletos/'+page,'./RE/'+page)
    else:
        image_rotated = rotate(image)
        text_rotated = pytesseract.image_to_string(image_rotated).lower()
        text_rotated_config = pytesseract.image_to_string(image_rotated, config=config).lower()
        if word_distance(text_rotated) or word_distance(text_rotated_config):
            print('found rotated: '+page)
            shutil.copy2('./boletos/'+page,'./RE/'+page)
        else:
            print('undefined: '+page)
            shutil.copy2('./boletos/'+page,'./UNDEFINED/'+page)

