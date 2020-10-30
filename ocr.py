import pytesseract
from PIL import Image
import os
import shutil

RE_list = ['recursos','interpostos','instancia','inferior','recurso','extraordinario']
AUTOS_list = ['remessa','retorno','autos','folhas']

x = 0
for page in os.listdir('./boletos'):
    text = pytesseract.image_to_string(Image.open('./boletos/'+page)).lower()
    if any(x in text for x in RE_list):
        shutil.copy2('./boletos/'+page,'./RE/'+page)
    else:
        if any(x in text for x in AUTOS_list):
            shutil.copy2('./boletos/'+page,'./AUTOS/'+page)
        else:
            shutil.copy2('./boletos/'+page,'./UNDEFINED/'+page)

