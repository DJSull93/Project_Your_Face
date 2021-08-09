import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=4,
    zoom_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

base_size = 120
color = 3

xy_train = train_datagen.flow_from_directory(
    '_data/JOB',
    target_size=(base_size, base_size),
    batch_size=9000,
    class_mode='categorical',
    shuffle=True,
    # color_mode='grayscale',
    subset='training'
)
# Found 8640 images belonging to 2 classes.
# 한혜진(모델) 임의로 가수 할당

xy_test = train_datagen.flow_from_directory(
    '_data/JOB',
    target_size=(base_size, base_size),
    batch_size=3000,
    class_mode='categorical',
    shuffle=True,
    # color_mode='grayscale',
    subset='validation'
)
# Found 2160 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

augment_size = 38600

randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 31000 feature from train in random

x_argmented = x_train[randidx].copy()
y_argmented = y_train[randidx].copy()

x_argmented = x_argmented.reshape(x_argmented.shape[0], base_size, base_size, color) # 
x_train = x_train.reshape(x_train.shape[0], base_size, base_size, color) # 
x_test = x_test.reshape(x_test.shape[0], base_size, base_size, color) # 

x_argmented = train_datagen.flow(x_argmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_argmented)) # 
y_train = np.concatenate((y_train, y_argmented)) # 

print(x_train.shape, x_test.shape) # (62031, 150, 150, 1) (10321, 150, 150, 1)
print(y_train.shape, y_test.shape) # (62031, 42) (10321, 42)

np.save('_save/_NPY/JOB_x_train', arr=x_train)
np.save('_save/_NPY/JOB_x_test', arr=x_test)
np.save('_save/_NPY/JOB_y_train', arr=y_train)
np.save('_save/_NPY/JOB_y_test', arr=y_test)

# # print(xy_train[0][0]) # 
# # print(xy_train[0][1]) # 
# print(xy_train[0][0].shape) # 
# print(xy_train[0][1].shape) # 