# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:39:56 2017

@author: rbarnes

This fun little project is about continuing education.
"""

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from PIL import Image
import argparse
import math
from cleandata import Cleaner
BATCH_SIZE=1000

class adversarial_generative_network:
    
    def __init__(self):
        self.g=self.make_generator_model()
        self.d=self.make_discriminator_model()
        self.agn=self.make_agn(self.g,self.d)
        self.X=Cleaner('h:\\Desktop\\GitHub\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_Network\\Data\\',BATCH_SIZE).clean_photos()
        self.noise = np.zeros((1, 1000))
        for i in range(1):
            self.noise[i, :] = np.random.uniform(-1, 1, 1000)
        print('Setup Complete')
        return None
    
    def get_data(self):
        return None
    
    def make_generator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=1000, output_dim=12544,init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dense(1024*4*4,init='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((4,4,1024), input_shape=(1024*4*4,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(512,5, 5, border_mode='same',init='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(256, 5, 5, border_mode='same',init='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(128, 5, 5, border_mode='same',init='glorot_normal'))
        model.add(BatchNormalization())
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(3, 5, 5, border_mode='same',init='glorot_normal'))
        #model.add(Activation('tanh'))
        return model
    
    
    def make_discriminator_model(self):
        model = Sequential()
        model.add(Convolution2D(
                            3, 5, 5,
                            border_mode='same',
                            input_shape=(64, 64, 3)))
        model.add(Activation('tanh'))
        model.add(Convolution2D(8, 5, 5,subsample=(2,2),border_mode='same',activation='relu'))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu'))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu'))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu'))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu'))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.25))
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
        noise = np.zeros((BATCH_SIZE, 1000))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 1000)
        generated_images = generator.predict(noise, verbose=1)
        print(generated_images.shape)
        image = self.combine_images(generated_images)
        image = image*127.5+127.5
        print(image.shape)
        Image.fromarray(image.astype(np.uint8)).save("h:\\Desktop\\GitHub\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_network\\Results\\generated_image.png")
    
    def generate_single_image(self):
        generator=self.g
        generated_images = generator.predict(self.noise, verbose=1)
        print(generated_images.shape)
        #image = self.combine_images(generated_images)
        #image = generated_images*127.5+127.5
        image = generated_images
        print(image.shape)
        return Image.fromarray(image[0,:,:,:].astype(np.uint8))
        
    def train(self,BATCH_SIZE,epochs=100):
        
        #d_optim=SGD(lr=0.01,momentum=0.99,nesterov=True)
        #g_optim=SGD(lr=0.01,momentum=0.99,nesterov=True)
        d_optim=Adam()
        g_optim=Adam()
        self.g.compile(loss='binary_crossentropy',optimizer="Adam")
        self.agn.compile(loss='binary_crossentropy',optimizer=g_optim)
        self.d.trainable=True
        self.d.compile(loss='binary_crossentropy',optimizer=d_optim)
        noise=np.zeros((BATCH_SIZE,1000))
        noise2=np.zeros((BATCH_SIZE*2,1000))
        for epoch in range(epochs):
            num=0
            for j in range(25):
                images=self.X.next()
                num+=BATCH_SIZE
                for i in range(BATCH_SIZE):
                    noise[i,:]=np.random.uniform(-1,1,1000)
                image_batch=images.astype(np.float32)
                generated_images=self.g.predict(noise,verbose=0)
                X=np.concatenate((image_batch,generated_images))
                y=[1]*BATCH_SIZE+[0]*BATCH_SIZE
                d_loss=self.d.train_on_batch(X,y)
                for i in range(BATCH_SIZE*2):
                    noise2[i, :] = np.random.uniform(-1, 1, 1000)
                self.d.trainable=False
                g_loss=self.agn.train_on_batch(noise2,[1]*BATCH_SIZE*2)
                print('Number of images trained: {0}, EPOCH {1} of {4}, d_loss:  {2},   g_loss:  {3}'.format(num,epoch+1,d_loss,g_loss,epochs))
                self.d.trainable=True
            self.generate_single_image().save('h:\\desktop\\github\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_Network\\Results\\epoch{0}.png'.format(epoch+1))
                
        
if __name__=='__main__':
    model=adversarial_generative_network()
    model.generate_single_image().save('h:\\desktop\\github\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_Network\\Results\\epoch0.png')
    model.train(BATCH_SIZE,1000)
    