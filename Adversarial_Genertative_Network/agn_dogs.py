# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:39:56 2017

@author: rbarnes

This fun little project is about continuing education.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
import numpy as np
from PIL import Image
import argparse
import math
from cleandata import Cleaner


class adversarial_generative_network:
    
    def __init__(self):
        self.g=self.make_generator_model()
        self.d=self.make_discriminator_model()
        self.agn=self.make_agn(self.g,self.d)
        self.X=Cleaner('h:\\Desktop\\GitHub\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_Network\\Data\\train').build_X()
        return None
    
    def get_data(self):
        return None
    
    def make_generator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(56*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((56, 7, 7), input_shape=(56*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(56, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(1, 8)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        return model
    
    
    def make_discriminator_model(self):
        model = Sequential()
        model.add(Convolution2D(
                            1, 1, 112,
                            border_mode='same',
                            input_shape=(3, 112, 112)))
        model.add(Activation('tanh'))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model
    
    
    def make_agn(self,generator, discriminator):
        model = Sequential()
        model.add(generator)
        discriminator.trainable=False
        model.add(discriminator)
        return model
    
    def combine_images(self,generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[0, :, :]
        return image
    
    def inspect_generated_images(self,BATCH_SIZE):
        generator=self.g
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        print(generated_images.shape)
        image = self.combine_images(generated_images)
        image = image*127.5+127.5
        print(image.shape)
        Image.fromarray(image.astype(np.uint8)).save("h:\\Desktop\\GitHub\\practice_deep_learning\\deep_learning\\Adversarial_Genertative\\Results\\generated_image.png")
    
    def generate_single_image(self):
        generator=self.g
        noise = np.zeros((1, 100))
        for i in range(1):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        print(generated_images.shape)
        image = self.combine_images(generated_images)
        image = image*127.5+127.5
        return Image.fromarray(image.astype(np.uint8))
        
    def train(self,BATCH_SIZE):
        d_optim=SGD(lr=0.0005,momentum=0.9,nesterov=True)
        g_optim=SGD(lr=0.0005,momentum=0.9,nesterov=True)
        self.g.compile(loss='binary_crossentropy',optimizer="SGD")
        self.agn.compile(loss='binary_crossentropy',optimizer=g_optim)
        self.d.trainable=True
        self.d.compile(loss='binary_crossentropy',optimizer=d_optim)
        noise=np.zeros((BATCH_SIZE,100))
        for epoch in range(100):
            for index in range(int(len(self.X)/BATCH_SIZE)):
                for i in range(BATCH_SIZE):
                    noise[i,:]=np.random.uniform(-1,1,100)
                image_batch=self.X[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                generated_images=self.g.predict(noise,verbose=0)
                print(image_batch.shape)
                print(generated_images.shape)
                #image_batch=np.reshape(image_batch,(10,112,112))
                generated_images=np.reshape(generated_images,(10,1,112,112))
                X=np.concatenate((image_batch,generated_images))
                #X=np.reshape(X,(20,1,112,112))
                y=[1]*BATCH_SIZE+[0]*BATCH_SIZE
                d_loss=self.d.train_on_batch(X,y)
                for i in range(BATCH_SIZE):
                    noise[i, :] = np.random.uniform(-1, 1, 100)
                self.d.trainable=False
                g_loss=self.agn.train_on_batch(noise,[1]*BATCH_SIZE)
                self.d.trainable=True
                
        
if __name__=='__main__':
    model=adversarial_generative_network()
    model.train(10)
    model.generate_single_image().show()