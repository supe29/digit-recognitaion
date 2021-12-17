import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tenserflow as tf

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# grey scall pixel can have 0-255(ignoring RGB values) -(0-1)
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
# neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, actication='relu'))
model.add(tf.keras.layers.Dense(128, actication='relu'))
model.add(tf.keras.layers.Dense(10, actication='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentroppy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)
loss, accuracy = model.evaluate(X_test, y_test)
print(loss)
print(accuracy)

model.save()
for x in range(1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'the result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary )
    plt.show()




