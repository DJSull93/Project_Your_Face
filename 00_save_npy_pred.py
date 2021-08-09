import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest'
)

base_size = 120
color = 3

test_datagen = ImageDataGenerator(rescale=1./255)

x_pred = train_datagen.flow_from_directory(
    '_data/sample',
    target_size=(base_size, base_size),
    batch_size=9000,
    class_mode='categorical',
    shuffle=True,
    # color_mode='grayscale',
)

np.save('_save/_NPY/pred_x_train', arr=x_pred[0][0])

print(x_pred[0]) # 
print(x_pred[0][0]) # 