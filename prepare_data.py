import os
import pandas as pd
import matplotlib.pyplot as plt

KAGGLE_DS_DIR = "/home/abhishek/dog_breed_id_kaggle/data/"
labels_file_path = os.path.join(KAGGLE_DS_DIR,"labels.csv")
sample_submission_file_path = os.path.join(KAGGLE_DS_DIR,"sample_submission.csv")
train_img_dir = os.path.join(KAGGLE_DS_DIR,"train")
test_img_dir = os.path.join(KAGGLE_DS_DIR,"test")


## For training
df = pd.read_csv(labels_file_path)
path = 'for_train'
for i, (fname, breed) in df.iterrows():
    breed_imgs_dir = '%s/%s' % (path, breed)
    if not os.path.exists(breed_imgs_dir):
        os.makedirs(breed_imgs_dir)
    os.symlink('%s/%s.jpg' % (train_img_dir, fname), '%s/%s.jpg' % (breed_imgs_dir, fname))

# for test imgs
df = pd.read_csv('/home/abhishek/dog_breed_id_kaggle/data/sample_submission.csv')
path = 'for_test'
breed = 'unkown'
for fname in df['id']:
    breed_imgs_dir = '%s/%s' % (path, breed)
    if not os.path.exists(breed_imgs_dir):
        os.makedirs(breed_imgs_dir)
    os.symlink('%s/%s.jpg' % (test_img_dir, fname), '%s/%s.jpg' % (breed_imgs_dir, fname))


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