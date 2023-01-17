from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization,MaxPool2D
from keras.regularizers import l2

svm = Sequential()

svm.add(Conv2D(filters=32,padding="same",kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]))
svm.add(MaxPool2D(pool_size=2, strides=2))
svm.add(Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))
svm.add(MaxPool2D(pool_size=2, strides=2))
svm.add(Flatten())
svm.add(Dense(units=128, activation='relu'))
svm.add(Dense(1, kernel_regularizer=l2(0.01),activation='linear'))

svm.summary()