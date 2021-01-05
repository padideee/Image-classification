#### Kaggle competition, team: EP

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import urllib.request
import glob
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation


# Custom Generator
class My_Custom_Generator(keras.utils.Sequence) :

    def __init__(self, images, labels, batch_size,num_classes=6, training=False) :
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.training = training
    
    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        IMG_SIZE = 224
        image_size = 28
        data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE,interpolation='bilinear'),
        tf.keras.layers.experimental.preprocessing.Rescaling(1., offset= -127.5),
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)])
        proc = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE,interpolation='bilinear'),
        tf.keras.layers.experimental.preprocessing.Rescaling(1., offset= -127.5)])

        batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x = batch_x.reshape(batch_x.shape[0], image_size, image_size, 1).astype('float32')
        batch_x = np.repeat(batch_x, 3, -1)
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = keras.utils.to_categorical(batch_y, num_classes)
        if self.training ==True:
            return np.array(data_augmentation(batch_x)), np.array(batch_y)
        else:
            return np.array(proc(batch_x)), np.array(batch_y)

        
# Prepping the data
train = np.load('train.npz')
x_train = train['arr_0']
y_train = train['arr_1']
num_classes = 6
image_size = 28
idx = np.array(list(range(len(x_train))))
np.random.shuffle(idx)

X = x_train[idx]
y = y_train[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

train_gen =  My_Custom_Generator ( X_train, y_train, batch_size=32,num_classes=6, training=True) 
val_gen = My_Custom_Generator ( X_val, y_val, batch_size=32,num_classes=6, training=False) 
test_gen = My_Custom_Generator ( X_test, y_test, batch_size=32,num_classes=6, training=False)


# VGG model
def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):

    image_input = inputs = tf.keras.Input(shape=(224, 224, 3))
    model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
    #model.summary()
    last_layer = model.get_layer('fc2').output
    out = Dense(num_classes, activation='softmax', name='output')(last_layer)
    model = Model(image_input, out)
    model.summary()

    #We set the first 8 layers to non-trainable (weights will not be updated)
    for layer in model.layers[:8]:
        layer.trainable = False

    # Learning rate is set to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


img_rows, img_cols = 128, 128 # input size
channel = 3
num_classes = 6
batch_size = 16
nb_epoch = 20

# Load model
model = vgg16_model(img_rows, img_cols, channel, num_classes)
model.summary()
hist = model.fit(train_gen, validation_data=val_gen,epochs=20,shuffle=True,verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

score = model.evaluate(test_gen, verbose=0)
print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))


# Use additional data (not same classes)
# classes =  ['ant','bulldozer','dolphin','flower','lobster','spider']
classes  = ['mosquito','firetruck',  'shark','tree','crab','snorkel']
!mkdir data


def download():
  
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:
        path = base+c+'.npy'
        print(path)
        urllib.request.urlretrieve(path, 'data/'+c+'.npy')

download() 


def load_data(root, vfold_ratio=0.2, max_items_per_class= 5000 ):
    
    all_files = glob.glob(os.path.join(root, '*.npy'))
    #initialize variables 
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    #load each data file 
    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)
        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels = None
    
    #randomize the dataset 
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    return x, y


x_train, y_train = load_data('data')
num_classes = 6
image_size = 28
idx = np.array(list(range(len(x_train))))
np.random.shuffle(idx)
X = x_train[idx]
y = y_train[idx]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

train_gen =  My_Custom_Generator ( X_train, y_train, batch_size=32,num_classes=6, training=True) 
val_gen = My_Custom_Generator ( X_val, y_val, batch_size=32,num_classes=6, training=False) 
test_gen = My_Custom_Generator ( X_test, y_test, batch_size=32,num_classes=6, training=False) 
model2 = vgg16_model(img_rows, img_cols, channel, num_classes)

# unfreeze the last layers:
for i,l in enumerate(model2.layers):
    print('{} : {}'.format(i,l.name))
    if i<20:
        l.trainable = False
    else:
        l.trainable = True

hist = model2.fit(train_gen, validation_data=val_gen,epochs=5,shuffle=True,verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# Train the model
train = np.load('train.npz')
x_train = train['arr_0']
y_train = train['arr_1']
num_classes = 6
image_size = 28
idx = np.array(list(range(len(x_train))))
np.random.shuffle(idx)
X = x_train[idx]
y = y_train[idx]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

train_gen =  My_Custom_Generator ( X_train, y_train, batch_size=32,num_classes=6, training=True) 
val_gen = My_Custom_Generator ( X_val, y_val, batch_size=32,num_classes=6, training=False) 
test_gen = My_Custom_Generator ( X_test, y_test, batch_size=32,num_classes=6, training=False)

hist = model2.fit(train_gen, validation_data=val_gen,epochs=20,shuffle=True,verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

score = model.evaluate(test_gen, verbose=0)
print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))

class My_Custom_Generator_test(keras.utils.Sequence) :
    def __init__(self, images,  batch_size,num_classes=6) :
        self.images = images
        self.batch_size = batch_size
    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
    def __getitem__(self, idx) :
        IMG_SIZE = 224
        image_size = 28
        proc = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE,interpolation='bilinear'),
        tf.keras.layers.experimental.preprocessing.Rescaling(1., offset= -127.5)])
        batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x = batch_x.reshape(batch_x.shape[0], image_size, image_size, 1).astype('float32')
        batch_x = np.repeat(batch_x, 3, -1)
        return np.array(proc(batch_x))

# Final prediction
test = np.load('test.npz')
x = test['arr_0']
test_gen = My_Custom_Generator_test(x,batch_size=100)
predict = model2.predict(test_gen)
preds = np.argmax(predict,axis=1)

with open('results.csv', 'w') as f:
    f.write('Id,Category\n')
    for i,line in enumerate(preds):
        f.write(str(i)+','+str(line)+'\n')
