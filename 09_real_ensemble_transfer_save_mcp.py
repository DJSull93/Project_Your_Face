import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.ops.gen_array_ops import Reshape
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

# make 2 cate to 11 -> y_train.shape = (N, 3, 11)
# (27240, 2) (2160, 2)
tail1 = np.zeros((47240, 9))
tail2 = np.zeros((2160, 9))

# print(tail1.shape, tail1[0])

y_train_MW = np.concatenate((y_train_MW, tail1), axis=1)
y_test_MW = np.concatenate((y_test_MW, tail2), axis=1)
y_train_JOB = np.concatenate((y_train_JOB, tail1), axis=1)
y_test_JOB = np.concatenate((y_test_JOB, tail2), axis=1)

# print(x_train_MW.shape) # (17240, 100, 100, 3)
# print(x_test_MW.shape) # (2160, 100, 100, 3)

Y_train = np.concatenate((y_train_MW, y_train_JOB, y_train_TYPE), axis=1)
Y_test = np.concatenate((y_test_MW, y_test_JOB, y_test_TYPE), axis=1)

# print(Y_train.shape)

Y_train = Y_train.reshape(47240, 3, 11)
Y_test = Y_test.reshape(2160, 3, 11)

print(Y_train.shape)
print(Y_test.shape)
# print(Y_train[0])
####################################################

# # 2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, Reshape, BatchNormalization
from tensorflow.keras.layers import concatenate

from tensorflow.keras.applications import ResNet50V2, VGG19, EfficientNetB0

base_size = 120
color = 3
img_size = (base_size, base_size, color)

# 2-1. model1
in1 = ResNet50V2(weights='imagenet',
            include_top=False,
            input_shape=img_size,
            )
in1.trainable = False
xx = in1.output
out1 = GlobalAveragePooling2D()(xx)


# 2-2. model2
in2 = EfficientNetB0(weights='imagenet',
            include_top=False,
            input_shape=img_size,
            )
in2.trainable = False
xx = in2.output
out2 = GlobalAveragePooling2D()(xx)


# 2-3. model3
in3 = VGG19(weights='imagenet',
            include_top=False,
            input_shape=img_size,
            )
in3.trainable = False
xx = in3.output
out3 = GlobalAveragePooling2D()(xx)


# 2-5. model 1, 2, 3 merge
from tensorflow.keras.layers import concatenate

merge1 = concatenate([out1, out2, out3]) 
xx = Dense(64, activation='relu')(merge1)
# xx = BatchNormalization()(xx)
# xx = Dense(128, activation='relu')(xx)
# xx = BatchNormalization()(xx)
# xx = Dense(64, activation='relu')(xx)
# xx = BatchNormalization()(xx)
xx = Dense(33, activation='relu')(xx)
xx = Reshape([3, 11])(xx)
l_out = Dense(11, activation='softmax', name='all')(xx)

model = Model(inputs=[in1.input, in2.input, in3.input], 
        outputs=l_out)

# model.summary()

# 3. compile train
from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',
            optimizer=Adam(3e-5), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=4, mode='auto', 
        verbose=2, restore_best_weights=True)

###################################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = '_save/_MCP/'
filename = '{epoch:04d}_{val_loss:4f}.hdf5'
modelpath = "".join([filepath, "pj01_", date_time, "_", filename])
###################################################################
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
            save_best_only=True, 
            # filepath='./_save/ModelCheckPoint/keras49_MCP.h5')
            filepath= modelpath)

import time 

start_time = time.time()
hist = model.fit(
    [x_train_MW, x_train_JOB, x_train_TYPE],
    Y_train,
    epochs=200, verbose=1,
    validation_split=0.1, callbacks=[es, mcp], 
    batch_size=256)
end_time = time.time() - start_time


# 4. predict eval 

loss = model.evaluate(
    [x_test_MW, x_test_JOB, x_test_TYPE],
    Y_test
    )

print('loss : ', hist.history['loss'][-4])
print('val_loss : ', hist.history['val_loss'][-4])
print('acc : ', hist.history['acc'][-4])
print('val_acc : ', hist.history['val_acc'][-4])

y_predict = model.predict([x_pred, x_pred, x_pred])
res11 = np.array([np.argmax(y_predict[0][0])])
res22 = np.array([np.argmax(y_predict[0][1])])
res33 = np.array([np.argmax(y_predict[0][2])])

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

print(result)

import matplotlib.pyplot as plt

acc = hist.history['acc']
val_acc = hist.history['val_acc']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

'''
loss :  1.0977648496627808
val_loss :  1.2829519510269165
acc :  0.5254538059234619
val_acc :  0.3711008131504059

loss :  1.1729713678359985
val_loss :  1.2309563159942627
acc :  0.48152947425842285
val_acc :  0.42494696378707886
'''