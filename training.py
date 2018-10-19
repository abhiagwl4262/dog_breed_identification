import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping

import resnet
import keras

KAGGLE_DS_DIR = "/home/abhishek/dog_breed_id_kaggle/data/"
labels_file_path = os.path.join(KAGGLE_DS_DIR,"labels.csv")
train_data_path  = os.path.join(KAGGLE_DS_DIR, "train/")

## Analyse data
train_dogs = pd.read_csv(labels_file_path)
train_dogs.head()
ax=pd.value_counts(train_dogs['breed'],ascending=True).plot(kind='barh',
                                                       fontsize="5",
                                                       title="Class Distribution",
                                                       figsize=(25,50))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
plt.show()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_augmentation = False
batch_size = 32
num_classes = 120
epochs = 1000
img_rows, img_cols, img_channels = 512, 512, 3
input_shape = (img_rows, img_cols, img_channels)


breeds = []
breedCateogry = []
j = 0 
for i, (fname, breed) in train_dogs.iterrows():
  if breed not in breeds:
    breeds.append(breed)
    breedCateogry.append(j)
    j+=1

images = []
labels = []
labels_num=[]
for i, (fname, breed) in train_dogs.iterrows():
  image_path = train_data_path + fname + ".jpg"
  im = load_img(image_path, target_size=(512, 512))
  im = img_to_array(im)
  im = np.array(im).astype('float32')
  images.append(im)
  labels.append(breed)
  for k in range(0, len(breeds)):
    if(breeds[k] == breed):
      labels_num.append(breedCateogry[k])

print(len(images))
print(len(labels_num))
print(len(labels))

num_classes = len(breeds)

image_data = np.array(images)
y_train = np.array(labels_num)
labels_data = keras.utils.to_categorical(labels_num, num_classes)

x_train, x_validation, y_train, y_validation = train_test_split(image_data, labels_data, test_size=0.2, stratify=np.array(labels), random_state=100)

# num_train      = x_train.shape(0)
# num_validation = x_validation.shape(0)


model_path = './dog_breed_id_kaggle_firsttry.h5'
# prepare callbacks
callbacks = [
    # EarlyStopping(
    #     monitor='loss', 
    #     patience=10,
    #     mode='max',
    #     verbose=1),
    ModelCheckpoint(model_path,
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=0)
]


model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), num_classes)
# model = create_baseline()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(lr=0.001, decay=1e-5),
              metrics=['accuracy'])

if data_augmentation:
    train_datagen = ImageDataGenerator(
        # featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=True # randomly flip images
        )
    train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=batch_size
            )

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow(
            x_validation, y_validation,
            batch_size=batch_size
            )

    history=model.fit_generator(
            train_generator,
            steps_per_epoch= 250, # batch_size,
            epochs=epochs,
            shuffle = True,
            # validation_data=(x_validation, y_validation),
            validation_data=validation_generator,
            validation_steps=62,
            )


if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(image_data, labels_data,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle = True,
              # validation_data=(x_validation, y_validation),
              validation_split=0.2,              
              callbacks=callbacks
              )
