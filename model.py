import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the folder location of the sample data
folder = '/rec'

# Load the samples from the csv-file
samples = []
with open('.{}/driving_log.csv'.format(folder)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split the data into a training and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Create a gererator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Iterate through the center, left and right image
                for i in range(3):
                    name = '.{}/IMG/'.format(folder) + batch_sample[i].split('\\')[-1]
                    image = cv2.imread(name)
                    # Add add or substract a value to the angle for the left and right image
                    if i == 1:
                        angle = float(batch_sample[3]) + 0.2
                    elif i == 2:
                        angle = float(batch_sample[3]) - 0.2
                    else:
                        angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(angle)
                    # Flip image to generate more examples
                    images.append(cv2.flip(image, 1))
                    angles.append(angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320 ,3)))
# Crop the images in order to cut out the distracting information
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Use the NVIDIA autonomous car architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
# From here on the NVIDIA autonomous car architecture has been changed. The fully connected layers are made bigger and two layers where added.
model.add(Dense(160))
model.add(Dense(160))
# Add a dropout layer to prevent overfitting
model.add(Dropout(0.7))
model.add(Dense(80))
model.add(Dense(80))
# Add a dropout layer to prevent overfitting
model.add(Dropout(0.7))
model.add(Dense(40))
model.add(Dense(1))

# Train the model with the training data and calculate the loss of the validation data
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(generator=train_generator, samples_per_epoch=len(train_samples)*6,
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples)*6,
                                     nb_epoch=7, verbose=1)

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save the model
model.save('model.h5')