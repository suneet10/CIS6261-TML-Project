import requests
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np

# load CIFAR-100 dataset with coarse labels
(x_train_all, y_train_all), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

# split dataset into train, validation, and test sets
split_idx_1 = int(0.8 * len(x_train_all))
split_idx_2 = int(0.9 * len(x_train_all))
train_x, train_y = x_train_all[:split_idx_1], y_train_all[:split_idx_1]
val_x, val_y = x_train_all[split_idx_1:split_idx_2], y_train_all[split_idx_1:split_idx_2]
test_x, test_y = x_train_all[split_idx_2:], y_train_all[split_idx_2:]

# preprocess the data
train_x = train_x.astype('float32') / 255.0
val_x = val_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0
train_y = to_categorical(train_y, num_classes=20)
val_y = to_categorical(val_y, num_classes=20)
test_y = to_categorical(test_y, num_classes=20)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))

# Compile the model
opt = Adam(lr=0.001)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
best_model_file = 'cifar100_coarse_model.h5'
checkpoint = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') # saves the best model file

# train the model
model.fit(train_x, train_y, epochs=150, batch_size=256, validation_data=(val_x, val_y), verbose=1, callbacks=[checkpoint])

# evaluate the model on test set
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy:', test_acc)

# save the model and dataset
# model.save('cifar100_coarse_model.h5')

class_names_url = 'https://raw.githubusercontent.com/yonghah/cifar100/master/cifar100_coarse_label_names.txt'
response = requests.get(class_names_url)
class_names = response.text.splitlines()
np.savez('cifar100_coarse_data.npz', train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, labels=class_names)
