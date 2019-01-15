import numpy as np
import sys
import os
num_ensemble = 10
single_ans_dir = sys.argv[1] + '/'
output_dir = sys.argv[2] +'/'

def process_line(line):
    path = line[:line.find(',')]
    label = line[-28:]
    label = np.array( list( map( int ,label.split(',') ) ) )
    return path,label


def process_ans_line(line):
    path = line[:line.find(',')]
    line = line[line.find(',')+1:]
    label = np.array( list( map( float ,line.split(',') ) ) )

    return path, label

def get_y_y_hat(file):
    img_name = []
    y = []
    y_hat = []
    pathes = []
    with open(file) as f:
        line_num = 0
  
        for line in f:
            if line_num > 0 :
                path,label = process_ans_line(line)
                y_hat.append(label)
                pathes.append(path)
            line_num+=1
        return np.array(y_hat), np.array(pathes)


files = os.listdir(single_ans_dir)

y_hats = []



for cnt in range(num_ensemble):
    w = [1,1,2,1,   1,1,1,1,1,  1,3,2,   1,1,2]
    w = np.array(w)
    np.random.shuffle(w)
    y_hat ,pathes = get_y_y_hat(single_ans_dir+files[0])
    print('start ensembling ', cnt )
    print(w[0], files[0])
    for i in range(1, len(files)):
        y_hat_tmp ,pathes = get_y_y_hat(single_ans_dir+files[i])
        y_hat += (y_hat_tmp* w[i])
        print(w[i], files[i])


    y_hat /= np.sum(w[0:len(files)])
    y_hats.append(y_hat)

import csv
def write_ans(pathes,pred_y,fname):
    file = open( fname, 'w')
    writer = csv.writer(file)
    csvHeader = ['Id','Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']
    writer.writerow(csvHeader)
    
    
    print(len(pred_y) )
    
    for i in range( len(pred_y)):
            row = []
            row.append(pathes[i])
            for j in range(14):
                row.append( str(pred_y[i][j]) )
            writer.writerow(row)


    return
for i in range(num_ensemble):
    idx = i+1
    write_ans(pathes, y_hats[i], output_dir+'ensemble_'+str(idx) + '.csv')
print('ok')
