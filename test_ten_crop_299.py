from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Flatten
from keras.models import Sequential,load_model
from keras.layers import Activation
import cv2
import gc
import numpy as np
import sys

img_path = sys.argv[1]
test_csv_path = sys.argv[2]
model_path = sys.argv[3]
output_path = sys.argv[4]


def load_data_from_file(file, batch_size = 32):
    cnt = 0
    X = []
    img_name = []
    with open(file) as f:
        line_num = 0
  
        for line in f:
            if line_num > 0 :
                x,path = process_line(line)
                
                X.append(ten_crop(x))
                img_name.append(path)
              
                cnt += 1
                if cnt == batch_size or line_num>=33652 :
                    
                    X = np.array(X)
                    yield ( X , img_name)
                    cnt = 0
                    del X
                    del x
                    gc.collect()
                    X = []
                    img_name = []

            line_num+=1

def ten_crop(img):
    img1 = random_crop(img,(1, 13))
    img2 = random_crop(img,(11, 32))
    img3 = random_crop(img,(18, 29))
    img4 = random_crop(img,(25, 26))
    img5 = random_crop(img,(26, 22))
    
    img6 = random_crop(img,(31, 3))
    img7 = random_crop(img,(15, 14))
    img8 = random_crop(img,(27, 3))
    img9 = random_crop(img,(23, 28))
    img10 = random_crop(img,(5, 16))

    imgs = [ img1, img2, img3, img4, img5 ,img6,img7, img8, img9, img10]
    
    return np.array(imgs)

def random_crop(img, xy):
    # Note: image_data_format is 'channel_last'
    random_crop_size = (299,299)
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = xy[0]
    y = xy[1]
    return img[y:(y+dy), x:(x+dx), :]

def process_line(line):
    path = line[:line.find('\n')]
    #print(path)
    img = cv2.imread(img_path+'/'+str(path))
    img = ( cv2.resize(img, (331, 331))  ).astype(np.float32)  /255.0
    return img, path


model = load_model(model_path)
model.summary()
gen = load_data_from_file(test_csv_path, 500)

ans = []
pathes = []
cnt = 0
x = []
while True:
    try:
        del x
        gc.collect()
        
        x, path = next(gen)
        print(x.shape)
        
        y1 = model.predict(x[:, 0, :, :,:], verbose = 1)
        y2 = model.predict(x[:, 1, :, :,:], verbose = 1)
        y3 = model.predict(x[:, 2, :, :,:], verbose = 1)
        y4 = model.predict(x[:, 3, :, :,:], verbose = 1)
        y5 = model.predict(x[:, 4, :, :,:], verbose = 1)
        
        y6 = model.predict(x[:, 5, :, :,:], verbose = 1)
        y7 = model.predict(x[:, 6, :, :,:], verbose = 1)
        y8 = model.predict(x[:, 7, :, :,:], verbose = 1)
        y9 = model.predict(x[:, 8, :, :,:], verbose = 1)
        y10 = model.predict(x[:, 9, :, :,:], verbose = 1)
        
        
        #y = model.predict(x, verbose = 1)
        
        y = (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10)/10.0
        ans.append(y)
        pathes.append(path)
        print(cnt)
        cnt = cnt +1
    except StopIteration:
        break


print('hi')

import csv
def write_ans(pathes,pred_y,fname):
    file = open( fname, 'w')
    writer = csv.writer(file)
    csvHeader = ['Id','Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']
    writer.writerow(csvHeader)
    
    
    print(len(pred_y) )
    

    for i in range( len(pred_y)):
        for j in range( pred_y[i].shape[0] ):
            
            row = []
            row.append(pathes[i][j])
            for k in range(14):
                row.append( str(pred_y[i][j][k]) )
            writer.writerow(row)


    return

write_ans(pathes, ans, output_path)
