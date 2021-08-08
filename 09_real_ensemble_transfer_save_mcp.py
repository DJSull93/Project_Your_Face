import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.merge import concatenate
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
tail1 = np.zeros((17240, 9))
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

Y_train = Y_train.reshape(17240, 3, 11)
Y_test = Y_test.reshape(17240, 3, 11)

# print(Y_train.shape)
# print(Y_train[0])
####################################################

# # 2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import concatenate

from tensorflow.keras.applications import ResNet50V2, InceptionResNetV2, VGG19

base_size = 100
color = 3
img_size = (base_size, base_size, color)

# 2-1. model1
in1 = ResNet50V2(weights='imagenet',
            include_top=False,
            input_shape=img_size,
            )
in1.trainable = False
xx = in1.output
xx = GlobalAveragePooling2D()(xx)
xx = Flatten()(xx)
out1 = Dense(64, activation='relu')(xx)

# 2-2. model2
in2 = VGG19(weights='imagenet',
            include_top=False,
            input_shape=img_size,
            )
in2.trainable = False
xx = in2.output
xx = GlobalAveragePooling2D()(xx)
xx = Flatten()(xx)
out2 = Dense(64, activation='relu')(xx)

# 2-3. model3
in3 = InceptionResNetV2(weights='imagenet',
            include_top=False,
            input_shape=img_size,
            )
in3.trainable = False
xx = in3.output
xx = GlobalAveragePooling2D()(xx)
xx = Flatten()(xx)
out3 = Dense(256, activation='relu')(xx)

# 2-5. model 1, 2, 3 merge
from tensorflow.keras.layers import concatenate

merge1 = concatenate([out1, out2, out3]) 
xx = Dense(256)(merge1)

out21 = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
l_out1 = Dense(2, activation='softmax', name='MW')(out21)

out22 = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
l_out2 = Dense(2, activation='softmax', name='JOB')(out22)

out33 = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
l_out3 = Dense(11, activation='softmax', name='Type')(xx)

model = Model(inputs=[in1.input, in2.input, in3.input], 
        outputs=[l_out1, l_out2, l_out3])

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
    [Y_train],
    epochs=200, verbose=1,
    validation_split=0.2, callbacks=[es, mcp], 
    batch_size=256)
end_time = time.time() - start_time


# 4. predict eval 

loss = model.evaluate(
    [x_test_MW, x_test_JOB, x_test_TYPE],
    [Y_test],
    )

print('loss : ', hist.history['loss'][-10])
print('val_loss : ', hist.history['val_loss'][-10])
# print('acc : ', hist.history['accuracy'])
# print('val_acc : ', hist.history['val_accuracy'])

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

print(result)

'''

'''