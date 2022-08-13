### Se crea este archivo pero no se va a ejecutar! ----------------------------

### STEPS 1 Y 2 ###

# instalar en consola
# pip install pillow
# pip install scipy
# pip install matplotlib
# pip install split-folders

# subir archivos train.zip y test1.zip al workspace

# una vez subidos, extraer los archivos zip
filename1 = "data/train/train.zip"
filename2 = "data/test/test1.zip"
extract_dir1 = "data/train"
extract_dir2 = "data/test"
archive_format = "zip"

# Unpack the archive file
shutil.unpack_archive(filename1, extract_dir1, archive_format)
shutil.unpack_archive(filename2, extract_dir2, archive_format)

# librerías
import numpy as np
from tensorflow import keras
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing import image
import shutil
import matplotlib.pyplot as plt
from matplotlib.image import imread
import splitfolders

### STEP 3 ###

# Graficar las primeras 9 imágenes de perros

# define location of dataset
folder = "data/train/train/"
fig, ax = plt.subplots(3, 3, figsize = (16, 16))
ax = ax.flatten()
for i in range(len(ax)):
    # define filename
    filename = folder + 'dog.' + str(i) + '.jpg'
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    ax[i].imshow(image)
    # show the figure
plt.show()

# Grafico las primeras 9 imágenes de gatos

# define location of dataset
folder = "data/train/train/"
fig, ax = plt.subplots(3, 3, figsize = (16, 16))
ax = ax.flatten()
for i in range(len(ax)):
    # define filename
    filename = folder + 'cat.' + str(i) + '.jpg'
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    ax[i].imshow(image)
    # show the figure
plt.show()

### STEP 4 ###

# Reubicar las imágenes de train en las subcarpetas cat y dog (desde terminal):
# mv data/train/train/cat*.jpg data/train/train/cat
# mv data/train/train/dog*.jpg data/train/train/dog

# Dividir imágenes de train en train y test (dentro de train_split), usando librería split-folders
input_folder = "data/train/train"
output = "data/train/train_split"
splitfolders.ratio(input_folder, output = output, seed=908, ratio=(.75, 0, .25)) # 75% train, 0% val, 25% test

# Se eliminan desde terminal archivos que ya no necesito:
# rm data/test/test1.zip
# rm data/train/train
# rm -r data/train/train

# se cargan los datos para el modelo, cambiándoles el tamaño
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory = "data/train/train_split/train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="data/train/train_split/test", target_size=(224,224))

### STEP 5 ###

# de define el modelo
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

### STEP 6 ###

# se pasan los datos a capa densa
model.add(Flatten())
model.add(Dense(units=4096,activation="relu")) # Cristian sugirió 128
model.add(Dense(units=4096,activation="relu")) # Cristian sugirió 32
model.add(Dense(units=2, activation="softmax"))

### STEP 7 ###

# importar y usar optimizador Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

### STEP 8 ###

# resumen del modelo
model.summary()

### SETP 9 ###

# funciones de devolución de llamada para el modelo
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

# se entrena el modelo
hist = model.fit_generator(generator = traindata, validation_data = testdata, epochs = 4, steps_per_epoch = 10, callbacks=[checkpoint,early])

# se guarda el modelo
model.save('models/vgg16_1.h5')

### Steps 10 y 11 quedan sólo en explore.ipynb