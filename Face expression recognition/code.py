import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break
    break
 
train_dir='/kaggle/input/face-expression-recognition-dataset/images/images/train'
valid_dir='/kaggle/input/face-expression-recognition-dataset/images/validation'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,SeparableConv2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_gen=ImageDataGenerator(rescale=1/255,width_shift_range=0.2,brightness_range=[1.2,0.8],rotation_range=30,height_shift_range=0.2)
valid_gen=ImageDataGenerator(rescale=1/255,validation_split=0.5)

train=train_gen.flow_from_directory(
    directory=train_dir,
    target_size=(128,128),
    class_mode='categorical',
    batch_size=100,
    color_mode='grayscale',
)
valid=valid_gen.flow_from_directory(
    directory=valid_dir,
    target_size=(128,128),
    class_mode='categorical',
    batch_size=100,
    color_mode='grayscale',
    subset='validation'
)
test=valid_gen.flow_from_directory(
    directory=valid_dir,
    class_mode='categorical',
    batch_size=100,
    color_mode='grayscale',
    target_size=(128,128),
    subset='training'
)

model=Sequential()
model.add(SeparableConv2D(32,(3,3),input_shape=(128,128,1)))
model.add(MaxPool2D(2))
model.add(SeparableConv2D(32,(3,3)))
model.add(MaxPool2D(2))
model.add(SeparableConv2D(32,(3,3)))
model.add(MaxPool2D(2))
model.add(SeparableConv2D(64,(3,3)))
model.add(MaxPool2D(2))
model.add(SeparableConv2D(32,(3,3)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(7,activation='softmax'))

model.summary()

epoch=10
model.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics=['acc'])
history=model.fit(train,steps_per_epoch=144,epochs=epoch,validation_data=valid,validation_steps=35)

d=history.history
plt.plot(range(1,epoch+1),d['acc'],label='train')
plt.plot(range(1,epoch+1),d['val_acc'],label='valid')
plt.legend()
plt.show()

plt.plot(range(1,epoch+1),d['loss'],label='train')
plt.plot(range(1,epoch+1),d['val_loss'],label='valid')
plt.legend()
plt.show()
