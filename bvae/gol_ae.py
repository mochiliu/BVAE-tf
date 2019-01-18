'''
vae.py
contains the setup for autoencoders.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from models import Darknet19Encoder, Darknet19Decoder
#import os

import numpy as np
from PIL import Image
from game_of_life_manager import GameManager
import time
from threading import Thread

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))


if __name__ == "__main__":
    inputShape = (32, 32, 3)
    latentSize = 256
    manager = GameManager()
    
    batchSize = manager.sample_size

    # This is how you build the autoencoder
    encoder = Darknet19Encoder(inputShape, batchSize, latentSize, 'vae', beta=69, capacity=15, randomSample=True)
    decoder = Darknet19Decoder(inputShape, batchSize, latentSize)
    bvae = AutoEncoder(encoder, decoder)

    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    iteration_number = 0
    
    twrv_one = ThreadWithReturnValue(target=manager.get_images, args=(batchSize,))
    twrv_one.start()
    img=twrv_one.join()
    
    while iteration_number < 1000000:
        twrv_one = ThreadWithReturnValue(target=manager.get_images, args=(batchSize,))
        twrv_one.start()
        
        while twrv_one.is_alive():
            bvae.ae.fit(img, img, epochs=1, batch_size=batchSize, verbose=0)
            iteration_number+=1
            if iteration_number % 100 == 0:
                print(str(iteration_number)+time.ctime())
            if iteration_number % 5000 == 0:
                latentVec = bvae.encoder.predict(img, batch_size=batchSize)[0]
                print(latentVec)
                print(time.ctime())
                sample_index = np.random.randint(batchSize)
                train = img[sample_index] #get a sample image
                train[train > 0.5] = 0.5 # clean it up a bit
                train[train < -0.5] = -0.5
                train = np.uint8((train + 0.5)* 255) # convert to regular image values
                train = Image.fromarray(train)
                train.save('./outputs/train_'+str(iteration_number)+'.bmp')
        
                pred = bvae.ae.predict(img, batch_size=batchSize)[sample_index] # get the reconstructed image
                pred[pred > 0.5] = 0.5 # clean it up a bit
                pred[pred < -0.5] = -0.5
                pred = np.uint8((pred + 0.5)* 255) # convert to regular image values
                pred = Image.fromarray(pred)
                pred.save('./outputs/pred_'+str(iteration_number)+'.bmp')
                
                #bvae.ae.save('./output_models/'+str(iteration_number)+'_autoencoder.h5')
                bvae.decoder.save('./output_models/'+str(iteration_number)+'_decoder.h5')
                bvae.encoder.save('./output_models/'+str(iteration_number)+'_encoder.h5')
        img=twrv_one.join()
        
        #check in once n iterations

        
