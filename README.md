ML2018 - Final Small Data Training for Medical Images
=============
<p align="right">組名: 泱泱溫妮琳妲凡凡千千提莫</p>

## Requirement
```bash
tensorflow-gpu==1.10.0
Keras==2.1.6
opencv-contrib-python==3.4.3.18
numpy==1.15.4
Pillow==5.3.0
imageio==2.4.1
Shapely==1.6.4.post2
scikit-image==0.14.1
matplotlib==3.0.0
scipy==1.1.0
six==1.11.0
imgaug==0.2.7
sklearn==0.0
pandas==0.23.4
```

## 安裝所需套件

```bash
pip install -r requirement.txt
```

## 前置作業
請先將Chest X-Ray Dataset所有影像放置到同一個資料夾當中。

## How to reproduce result

### Step1. 執行 train.sh
train.sh 會先在當前目錄創建名為'model'的資料夾，並且訓練模型(model_1.h5~ model_13.h5)存到該資料夾中。

train.sh 需要兩個參數 [image_dir_path] 和 [train.csv]。

```bash
Usage: bash train.sh [image_dir_path] [train.csv]

[image_dir_path] : Chest X-Ray Dataset資料夾的路徑。
[train.csv] : htc 提供的train.csv路徑。
```


例如:

```bash
bash train.sh ../final/images ../final/train.csv
```

用 Geforce 1080Ti 大約要跑12小時，請耐心等候。

跑完後，可以去"model"資料夾中檢查是否有13個 model (model_1.h5~ model_13.h5)

其中兩個model_3.h5 和model_4.h5 可能不會被生出來，因為我有設定epoch大於8 且loss、auroc超過門檻值才能被生出來。

如果都有被生出來請忽略以下指令。

如果其中一個沒被生出來，請幫我手動輸入以下指令(注意您的影像、train.csv路徑可能與以下不同):

```bash
python3 train_3.py ../final/images ../final/train.csv model_3.h5
或
python3 train_3.py ../final/images ../final/train.csv model_4.h5

(model_3.h5 和 model_4.h5 都是同一個 train_3.py 生的)
```


### Step2. 執行 test.sh

請先確定model資料夾有 model_1.h5~ model_13.h5 13個model。

若都有請執行以下指令(注意您的影像、test.csv路徑可能與以下不同):

```bash
bash test.sh ../final/images ../final/test.csv
```

此指令會讀入 'model' 資料夾裡的13個model (model_1.h5~ model_13.h5)，並輸出各自的答案到"single_ans"資料夾(會由script 自動產生)，共生成13個答案(ans_1.csv ans_13.csv)。

用 Geforce 1080Ti 大約要跑10小時，請耐心等候。
(還有下一步喔)


### Step3. 執行 ensemble.sh

最後一步很簡單，請確認 single_ans 有13個答案(ans_1.csv ans_13.csv)。

```bash
bash ensemble.sh
```
這個指令會先創建ensemble_ans這個資料夾，


同時指令會用不同的權重將single_ans裡的答案加權平均，總共用十種不同的權重排列組合，

會在生成ensemble_1.csv to ensemble_10.csv十個答案。



