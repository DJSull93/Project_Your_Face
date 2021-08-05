import numpy as np
import tensorflow as tf
# 1. data

# path = '_save/_NPY/color/'
path = '_save/_NPY/gray/'

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

# print(x_train_MW.shape, #
#     x_train_JOB.shape, #
#     x_train_TYPE.shape, #
#     )

# # 2. model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import concatenate
'''
img_size = (100, 100, 1)

# 2-1. model1
in1 = Input(shape=img_size)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(in1)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu')(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Flatten()(xx)
out1 = Dense(5, activation='relu')(xx)

# 2-2. model2
in2 = Input(shape=img_size)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(in2)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu')(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Flatten()(xx)
out2 = Dense(5, activation='relu')(xx)

# 2-3. model3
in3 = Input(shape=img_size)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(in3)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Flatten()(xx)
out3 = Dense(5, activation='relu')(xx)

# 2-5. model 1, 2, 3, 4 merge
from tensorflow.keras.layers import concatenate

merge1 = concatenate([out1, out2, out3]) 

xx = Dense(20)(merge1)

out21 = Dense(4, activation='relu')(xx)
l_out1 = Dense(2, activation='softmax')(out21)

out22 = Dense(4, activation='relu')(xx)
l_out2 = Dense(2, activation='softmax')(out22)

out33 = Dense(20, activation='relu')(xx)
l_out3 = Dense(11, activation='softmax')(out33)

model = Model(inputs=[in1, in2, in3], 
        outputs=[l_out1, l_out2, l_out3])

# model.summary()

# 3. compile train
model.compile(loss='categorical_crossentropy', 
            optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', 
        verbose=2, restore_best_weights=True)

###################################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = '_save/_MCP/'
filename = '{epoch:04d}_{val_loss:4f}.hdf5'
modelpath = "".join([filepath, "k47_", date_time, "_", filename])
###################################################################
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
            save_best_only=True, 
            # filepath='./_save/ModelCheckPoint/keras49_MCP.h5')
            filepath= modelpath)

import time 

start_time = time.time()
hist = model.fit(
    [x_train_MW, x_train_JOB, x_train_TYPE],
    [y_train_MW, y_train_JOB, y_train_TYPE],
    epochs=20, verbose=1,
    validation_split=0.01, callbacks=[es, mcp], 
    batch_size=100)
end_time = time.time() - start_time


# 4. predict eval -> no need to
'''

print('=================2. load model=================')
model = load_model('_save/_MCP/k47_0804_0113_0005_3.619629.hdf5')


loss = model.evaluate(
    [x_test_MW, x_test_JOB, x_test_TYPE],
    [y_test_MW, y_test_JOB, y_test_TYPE],
    )

# print('loss : ', hist.history['loss'])
# print('val_loss : ', hist.history['val_loss'])
# print('acc : ', hist.history['acc'])
# print('val_acc : ', hist.history['val_acc'])

y_predict = model.predict([x_pred, x_pred, x_pred])
res1 = (np.argmax(y_predict[0]))
res2 = (np.argmax(y_predict[1]))
res3 = (np.argmax(y_predict[2]))

print('남자0 여자1 : ',res1)
print('배우0 가수1 : ',res2)
print('닮은꼴 코드 : ',res3)


