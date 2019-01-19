'''
vae.py
contains the setup for autoencoders.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from simple_models import Encoder, Decoder
#from keras.callbacks import TensorBoard

#import os

import numpy as np
from PIL import Image
import time
from keras.datasets import mnist

class mnist_manager(object):
    def __init__(self):
        self.img_size = 32
        self.n_samples = 4*64
        (self.x_train, _), (self.x_test, _) = mnist.load_data()

    @property
    def sample_size(self):
        return self.n_samples
    
    def generate_images_train(self):
        while True:
            current_index = 0
            images = np.zeros((self.n_samples,self.img_size,self.img_size,3), dtype=np.float32)
            for index in range(self.n_samples):
              img = self.x_train[current_index]
              images[index,2:30,2:30,0] = img * np.random.uniform()
              images[index,2:30,2:30,1] = img * np.random.uniform()
              images[index,2:30,2:30,2] = img * np.random.uniform()
              current_index += 1
              current_index = current_index % len(self.x_train)
            images = (images / 255)
            yield images, images

    def generate_images_test(self):
        while True:
            current_index = 0
            images = np.zeros((self.n_samples,self.img_size,self.img_size,3), dtype=np.float32)
            for index in range(self.n_samples):
              img = self.x_test[current_index]
              images[index,2:30,2:30,0] = img * np.random.uniform()
              images[index,2:30,2:30,1] = img * np.random.uniform()
              images[index,2:30,2:30,2] = img * np.random.uniform()
              current_index += 1
              current_index = current_index % len(self.x_test)
            images = (images / 255) 
            yield images, images
            
    def get_images(self, count):
        (images, _) = next(self.generate_images_test())
        return images
    
class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))


if __name__ == "__main__":
    (x_train, _), (x_test, _) = mnist.load_data()

    inputShape = (32, 32, 3)
    latentSize = 32

    manager = mnist_manager()
    batchSize = manager.sample_size
    ntrain=50#number_of_training_samples//batchSize 
    nval=1#number_of_validation_samples//batchSize
    # This is how you build the autoencoder
    encoder = Encoder(inputShape, batchSize, latentSize, 'vae', beta=69, capacity=15, randomSample=True)
    decoder = Decoder(inputShape, batchSize, latentSize)
    bvae = AutoEncoder(encoder, decoder)

    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    iteration_number = 0

    while iteration_number < 100:
        bvae.ae.fit_generator(manager.generate_images_train(), steps_per_epoch=ntrain, validation_data=next(manager.generate_images_test()), validation_steps=nval, epochs=1,verbose=1)
        
        sample_index = np.random.randint(batchSize)
        img = manager.get_images(batchSize)
        latentVec = bvae.encoder.predict(img, batch_size=batchSize)[sample_index]
        print(latentVec)
        print(str(iteration_number) + ' ' + time.ctime())
        train = img[sample_index] #get a sample image
        train = np.uint8(train* 255) # convert to regular image values
        train = Image.fromarray(train)
        train.save('./outputs/train_'+str(iteration_number)+'.bmp')
        #train.save('.\\outputs\\train_'+str(iteration_number)+'.bmp')
        
        pred = bvae.ae.predict(img, batch_size=batchSize)[sample_index] # get the reconstructed image
        pred = np.uint8(pred * 255) # convert to regular image values
        pred = Image.fromarray(pred)
        pred.save('./outputs/pred_'+str(iteration_number)+'.bmp')
        #pred.save('.\\outputs\\pred_'+str(iteration_number)+'.bmp')
        
        if iteration_number % 10 == 0:
            bvae.ae.save('./output_models/'+str(iteration_number)+'_autoencoder.h5')
            bvae.decoder.save('./output_models/'+str(iteration_number)+'_decoder.h5')
            bvae.encoder.save('./output_models/'+str(iteration_number)+'_encoder.h5')

#            bvae.ae.save('.\\output_models\\'+str(iteration_number)+'_autoencoder.h5')
#            bvae.decoder.save('.\\output_models\\'+str(iteration_number)+'_decoder.h5')
#            bvae.encoder.save('.\\output_models\\'+str(iteration_number)+'_encoder.h5')
        iteration_number+=1
        #check in once n iterations

        
