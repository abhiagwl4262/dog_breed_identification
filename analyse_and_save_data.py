import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


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


np.save("image_data.npy", image_data)
np.save("y_train.npy", y_train)