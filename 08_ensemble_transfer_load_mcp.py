import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# 1. data

path = '_save/_NPY/color/'
# path = '_save/_NPY/gray/'

x_train_MW = np.load(path+'MW_x_train.npy')
x_test_MW = np.load(path+'MW_x_test.npy')
y_train_MW = np.load(path+'MW_y_train.npy')
y_test_MW = np.load(path+'MW_y_test.npy')

x_train_JOB = np.load(path+'JOB_x_train.npy')
x_test_JOB = np.load(path+'JOB_x_test.npy')
y_train_JOB = np.load(path+'JOB_y_train.npy')
y_test_JOB = np.load(path+'JOB_y_test.npy')

x_train_TYPE = np.load(path+'TYPE_x_train.npy')
x_test_TYPE = np.load(path+'TYPE_x_test.npy')
y_train_TYPE = np.load(path+'TYPE_y_train.npy')
y_test_TYPE = np.load(path+'TYPE_y_test.npy')

x_pred = np.load(path+'pred_x_train.npy')

# # 2. model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import concatenate

print('=================load model=================')
model = load_model('_save/_MCP/MCP_store/pj01_0804_1052_0091_3.413314.hdf5')

loss = model.evaluate(
    [x_test_MW, x_test_JOB, x_test_TYPE],
    [y_test_MW, y_test_JOB, y_test_TYPE],
    )

y_predict = model.predict([x_pred, x_pred, x_pred])
res11 = np.array([np.argmax(y_predict[0])])
res22 = np.array([np.argmax(y_predict[1])])
res33 = np.array([np.argmax(y_predict[2])])

for i in res11 :
    if i == 0:
        res1 = '남자'
    if i == 1:
        res1 = '여자'

for i in res22 :
    if i == 0:
        res2 = '배우'
    if i == 1:
        res2 = '가수'

for i in res33:
    if i == 0:
        res3 = '강아지'
    if i == 1:
        res3 = '고양이'
    if i == 2:
        res3 = '토끼'
    if i == 3:
        res3 = '여우'
    if i == 4:
        res3 = '공룡'
    if i == 5:
        res3 = '개구리'
    if i == 6:
        res3 = '뱀'
    if i == 7:
        res3 = '꼬북이'
    if i == 8:
        res3 = '곰'
    if i == 9:
        res3 = '쥐'
    if i == 10:
        res3 = '호랑이'

result = "".join(['당신은 ',res3,'상의 ',res1,' ',res2,'같아요!'])

from PIL import Image
plt.rc('font', family='GULIM')

path_pred = '_data/sample/sam/'
file_name = '84132187.jpg'
image_pil = Image.open(path_pred+file_name)
image = np.array(image_pil)

plt.title(result)
plt.imshow(image_pil)
plt.show()