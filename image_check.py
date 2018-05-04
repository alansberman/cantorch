import os
from PIL import Image
from glob import glob

data_dir = 'D:\\WikiArt\\wiki\\'

def get_data(data_dir):
    data = glob(data_dir+"/*/*", recursive=True)
    return data

bad_files = []

data = get_data(data_dir)
for f in data:
    if f.endswith('.jpg'):
        try:
            img  = Image.open(f) # open the image file
            w, h = img.size
            if w != 64:
                print("width bad", f, w)
            if h != 64:
                print("height bad",f, h)
            
            img.verify() # verify that it is, in fact an image
        except (IOError,SyntaxError) as e:
            bad_files.append(f)

print(bad_files)

        
# for filename in listdir():
#   if filename.endswith('.png'):
#     try:
#       img = Image.open('./'+filename) # open the image file
#       img.verify() # verify that it is, in fact an image
#     except (IOError, SyntaxError) as e:
#       print('Bad file:', filename) # print out the names of corrupt files