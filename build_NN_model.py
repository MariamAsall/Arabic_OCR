import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD  # Import SGD optimizer

from load_NN import *

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
input_shape_original = X_train.shape[1]
# create model

model = Sequential()
model.add(Dense(300, input_dim=input_shape_original, activation='relu', kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(300, activation='relu' , kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.28))

model.add(Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history=model.fit(X_train, y_train, epochs=250, batch_size=256,validation_split=0.2)

# Saving the model, structure, and weights
model.save('C:/Users/Lenovo/Downloads/poject/project/models/NN_Model_1.h5')
model_json = model.to_json()
with open("C:/Users/Lenovo/Downloads/poject/project/models/NN_Model_structure_1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('C:/Users/Lenovo/poject/project/models/NN_Model_weights_1.h5')

# Plotting accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(0.4, 1)
plt.savefig('C:/Users/Lenovo/Downloads/poject/project/diagram/model_accuracy.jpg')
plt.show()

# Plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(0, 1)
plt.savefig('C:/Users/Lenovo/Downloads/poject/project/diagram/model_loss.jpg')
plt.show()
