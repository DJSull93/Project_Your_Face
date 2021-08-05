from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate
from keras_vggface.vggface import VGGFace
from tensorflow.keras.applications import ResNet101V2, VGG16, InceptionResNetV2
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf


batch_size = 32
img_height = 224
img_width = 224
epochs = 1000
img_shape = (img_height , img_width, 3)

# 1. data

path = '_save/_NPY/color/'

x_train_MW = np.load(path+'MW_x_train.npy')
x_test_MW = np.load(path+'MW_x_test.npy')
y_train_MW = np.load(path+'MW_y_train.npy')
y_test_MW = np.load(path+'MW_y_test.npy')

model = ResNet101V2(weights='imagenet', 
                include_top=False,
                input_shape=img_shape)

model.trainable=True


poolinglyaer = GlobalAveragePooling2D()
prediction_layer = Dense(2, activation ='softmax' )

model = tf.keras.Sequential([
    model,
    poolinglyaer,
    prediction_layer
])


'''
model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', 
        verbose=2, restore_best_weights=True)

history = model.fit(x_train_MW, y_train_MW, epochs=epochs,
                   validation_steps=2,validation_split=0.2,
                   callbacks=[es])

print("정확도 : %.4f" % (model.evaluate(x_test_MW, y_test_MW, 
                    callbacks=[es], batch_size=4)[1]))
'''



model.summary()

'''
$ ResNet101V2
$ False
resnet101v2 (Functional)     (None, 4, 4, 2048)        42626560
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 2)                 4098
=================================================================
Total params: 42,630,658
Trainable params: 4,098
Non-trainable params: 42,626,560

$ True
resnet101v2 (Functional)     (None, 4, 4, 2048)        42626560
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 2)                 4098
=================================================================
Total params: 42,630,658
Trainable params: 42,532,994
Non-trainable params: 97,664

'''

'''
$ vgg16
$ False
vgg16 (Functional)           (None, 3, 3, 512)         14714688
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 2)                 1026
=================================================================
Total params: 14,715,714
Trainable params: 1,026
Non-trainable params: 14,714,688

$ True
vgg16 (Functional)           (None, 3, 3, 512)         14714688
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 2)                 1026
=================================================================
Total params: 14,715,714
Trainable params: 14,715,714
Non-trainable params: 0
'''

'''
$ InceptionResNetV2
$ False
inception_resnet_v2 (Functio (None, 1, 1, 1536)        54336736
_________________________________________________________________
global_average_pooling2d (Gl (None, 1536)              0
_________________________________________________________________
dense (Dense)                (None, 2)                 3074
=================================================================
Total params: 54,339,810
Trainable params: 3,074
Non-trainable params: 54,336,736

$ True
inception_resnet_v2 (Functio (None, 1, 1, 1536)        54336736
_________________________________________________________________
global_average_pooling2d (Gl (None, 1536)              0
_________________________________________________________________
dense (Dense)                (None, 2)                 3074
=================================================================
Total params: 54,339,810
Trainable params: 54,279,266
Non-trainable params: 60,544
'''

'''
$ include_top = True
`include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3)
'''