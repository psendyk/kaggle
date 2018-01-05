import numpy as np
from keras.applications.vgg16 import VGG16
from scipy import misc
import pickle
import os

test_img_names = os.listdir('data/test')
train_img_names = os.listdir('data/train')

def load_img(filename):
    img = misc.imread(filename)
    img = misc.imresize(img, size=(224,224,3))
    img = img/255
    return img

#Load test and training images
test_imgs = [load_img(os.path.join('data/test', name)) for name in test_img_names]
test_imgs = np.stack(test_imgs)

train_imgs = [load_img(os.path.join('data/train', name)) for name in train_img_names]
train_imgs = np.stack(train_imgs)

with open('data/train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)

#Load the pretrained InceptionV3 model
Model = vgg16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

print('loaded the model')

featurized_train_data = Model.predict(train_imgs, verbose=1)
featurized_test_data = Model.predict(test_imgs, verbose=1)

#Save featurized images
with open('featurized_train_imgs.pkl', 'wb') as f:
    pickle.dump(featurized_train_data, f)
with open('featurized_test_imgs.pkl', 'wb') as f:
    pickle.dump(featurized_test_data, f)

