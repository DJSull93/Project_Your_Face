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

# # 2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import concatenate

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
out1 = Dense(64, activation='relu')(xx)

# 2-2. model2
in2 = Input(shape=img_size)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(in2)
xx = Conv2D(32, kernel_size=(2,2), activation='relu')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu')(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Flatten()(xx)
out2 = Dense(64, activation='relu')(xx)

# 2-3. model3
in3 = Input(shape=img_size)
xx = Conv2D(32, kernel_size=(2,2), activation='relu', padding='same')(in3)
xx = Conv2D(32, kernel_size=(2,2), activation='relu', padding='same')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(xx)
xx = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(xx)
xx = MaxPooling2D(2,2)(xx)
xx = Flatten()(xx)
out3 = Dense(100, activation='relu')(xx)

# 2-5. model 1, 2, 3 merge
from tensorflow.keras.layers import concatenate

merge1 = concatenate([out1, out2, out3]) 
xx = Dense(64)(merge1)

out21 = Dense(32, activation='relu')(xx)
l_out1 = Dense(2, activation='softmax', name='MW')(out21)

out22 = Dense(32, activation='relu')(xx)
l_out2 = Dense(2, activation='softmax', name='JOB')(out22)

out33 = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
l_out3 = Dense(11, activation='softmax', name='Type')(xx)

model = Model(inputs=[in1, in2, in3], 
        outputs=[l_out1, l_out2, l_out3])

# model.summary()

# 3. compile train
from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', 
            optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', 
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
    epochs=200, verbose=1,
    validation_split=0.2, callbacks=[es, mcp], 
    batch_size=256)
end_time = time.time() - start_time


# 4. predict evaluate 

loss = model.evaluate(
    [x_test_MW, x_test_JOB, x_test_TYPE],
    [y_test_MW, y_test_JOB, y_test_TYPE],
    )

print('loss : ', hist.history['loss'][-10])
print('val_loss : ', hist.history['val_loss'][-10])

'''
Total params: 8,165,683
'''


'''
MW_acc: 0.7000 - JOB_acc: 0.5745 - Type_acc: 0.1733
val_MW_acc: 0.6539 - val_JOB_acc: 0.5579 - val_Type_acc: 0.1597
'''
