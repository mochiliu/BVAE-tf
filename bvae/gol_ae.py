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
import shutil, os
import numpy as np
from PIL import Image
from game_of_life_manager import GameManager
import time


class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))


if __name__ == "__main__":
    #set up output folders
    path = os.getcwd()  
    outputs_folder = os.path.join(path, 'outputs')
    output_models_folder = os.path.join(path, 'output_models')
    shutil.rmtree(outputs_folder,ignore_errors=True)
    shutil.rmtree(output_models_folder,ignore_errors=True)
    os.mkdir(outputs_folder)
    os.mkdir(output_models_folder)
    
    inputShape = (32, 32, 3)
    latentSize = 900
    batchSize = 4*64
    
    manager = GameManager(batchSize)
    ntrain=50#number_of_training_samples//batchSize 
    nval=1#number_of_validation_samples//batchSize
    # This is how you build the autoencoder
    encoder = Encoder(inputShape, batchSize, latentSize, 'vae', beta=69, capacity=15, randomSample=True)
    decoder = Decoder(inputShape, batchSize, latentSize)
    bvae = AutoEncoder(encoder, decoder)

    bvae.ae.compile(optimizer='adam', loss='binary_crossentropy')
    iteration_number = 0

    while iteration_number < 100:
        bvae.ae.fit_generator(manager.generate_images(), steps_per_epoch=ntrain, max_queue_size=20, workers=6, use_multiprocessing=True, validation_data=next(manager.generate_images()), validation_steps=nval, epochs=1,verbose=1)
        #bvae.ae.fit_generator(manager.generate_images(), steps_per_epoch=ntrain, workers=1, validation_data=next(manager.generate_images()), validation_steps=nval, epochs=1,verbose=1)

        img = manager.get_images(batchSize)
        latentVec = bvae.encoder.predict(img, batch_size=batchSize)[0]
        #print(latentVec)
        print(str(iteration_number) + ' ' + time.ctime())
        train = img[0] #get a sample image
        train = np.uint8(train* 255) # convert to regular image values
        train = Image.fromarray(train)
        train.save(os.path.join(outputs_folder,str(iteration_number)+'_train_'+'.bmp'))
        
        pred = bvae.ae.predict(img, batch_size=batchSize)[0] # get the reconstructed image
        pred = np.uint8(pred * 255) # convert to regular image values
        pred = Image.fromarray(pred)
        pred.save(os.path.join(outputs_folder,str(iteration_number)+'_pred_'+'.bmp'))

        if iteration_number % 10 == 0:
            #bvae.ae.save(os.path.join(outputs_folder, str(iteration_number)+'_autoencoder.h5'))
            bvae.decoder.save(os.path.join(outputs_folder, str(iteration_number)+'_decoder.h5'))
            bvae.encoder.save(os.path.join(outputs_folder, str(iteration_number)+'_encoder.h5'))
        iteration_number+=1
        #check in once n iterations

        
    #os.system('sudo shutdown -h now')
