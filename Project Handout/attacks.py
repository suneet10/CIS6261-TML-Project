import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# Function to calculate adversary noise
def generate_adversary(model, image, label):
    image = tf.convert_to_tensor(image, dtype=tf.float32) # convert to tensor
    label = tf.convert_to_tensor(label.reshape((1, -1)), dtype=tf.float32) # convert to tensor

    loss_function = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_function(label, prediction)
    gradient = tape.gradient(loss, image)
    sign_grad = tf.sign(gradient)

    return sign_grad


# Function to generate batch of images with adversary
def adversary_generator(model, aux_x,aux_y,batch_size):
    while True:
        benign_images = []
        adv_images = []
        labels = []
        for batch in range(batch_size):
            N = np.random.randint(4999)
            label = aux_y[N]
            image = aux_x[N].reshape((1,32, 32, 3))

            benign_images.append(image)

            perturbations = generate_adversary(model, image, label).numpy()
            adversarial = image + (perturbations * 0.1)

            adv_images.append(adversarial)
            labels.append(label)

            if batch%100 == 0:
                print(f"{batch} images generated")

        adv_images = np.asarray(adv_images).reshape((batch_size, 32, 32, 3))
        benign_images = np.asarray(benign_images).reshape((batch_size, 32, 32, 3))
        labels = np.asarray(labels)

        yield benign_images, adv_images, np.argmax(labels, axis=1)


