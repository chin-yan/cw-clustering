from PIL import Image
import os


fin = r'C:\Users\VIPLAB\Desktop\Yan\ANNProject#2\cartoon-gan-master3.0\datasets\trainA'   
fout = r'C:\Users\VIPLAB\Desktop\Yan\ANNProject#2\cartoon-gan-master3.0\datasets\trainA'    

for file in os.listdir(fin):
    file_fullname = fin + '/' +file
    print(file_fullname)                           
    img = Image.open(file_fullname)
    if img.mode == "CMYK":
        img = img.convert('RGB')
    img = img.convert('RGB')
    #im_resized = img.resize((256, 256))             
    out_path = fout + '/' + file
    #im_resized.save(out_path)         
                  

