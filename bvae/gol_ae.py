'''
vae.py
contains the setup for autoencoders.

created by shadySource

THE UNLICENSE
'''
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, TensorBoard
from simple_models import Encoder, Decoder, ConvEncoder, ConvDecoder, OptimalEncoder, OptimalDecoder
import shutil, os
import numpy as np
from PIL import Image
from game_of_life_manager import GameManager
import time
import subprocess

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

class ChangeMetrics(Callback):
    def on_epoch_end(self, epoch, logs):
        logs['loss'] = np.mean(logs['loss'])  # replace it with your metrics
        logs['val_loss'] = np.mean(logs['val_loss'])  # replace it with your metrics
            
    def on_batch_end(self, batch, logs):
        logs['loss'] = np.mean(logs['loss'])  # replace it with your metrics
    
            
class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))


if __name__ == "__main__":
    #test if windows or linux
    path = os.getcwd() 
    if os.name == 'nt':
        batchSize = 64
        ntrain=1#number_of_training_samples//batchSize 
        nval=1#number_of_validation_samples//batchSize  
        iterations = 0
        git_commit_msg_file = os.path.join(path, '..', '.git', 'COMMIT_EDITMSG')
        f = open(git_commit_msg_file, "r")
        msg = f.read()
    else:
        #rtx 2060 correction
        import tensorflow as tf
        from tensorflow.python.keras.backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        batchSize = 4*64
        ntrain=16*64#number_of_training_samples//batchSize 
        nval=16#number_of_validation_samples//batchSize  
        iterations = 1500
        msg = subprocess.check_output("git log -1 --pretty=%B", shell=True)
        msg = msg.decode('utf-8')
        os.system('tensorboard --logdir=/tmp/logs &')
        tensorboard = TensorBoard(log_dir='/tmp/logs', histogram_freq=0, batch_size=batchSize, write_graph=False)
        time.sleep(15) # wait for it to boot up
    inputShape = (32, 32, 3)
    intermediateSize = 256 
    latentSize = 256
    fast_multiplier = 8
    msg = msg.replace(' ', '_').lower()
    msg = msg.splitlines()[0]

    #set up output folders    
    print(msg)
    path = os.getcwd()  
    outputs_folder = os.path.join(path, 'outputs', msg)
    output_models_folder = os.path.join(path, 'output_models', msg)
    shutil.rmtree(outputs_folder,ignore_errors=True)
    shutil.rmtree(output_models_folder,ignore_errors=True)
    os.makedirs(outputs_folder,exist_ok=True)
    os.makedirs(output_models_folder,exist_ok=True)
  
    manager = GameManager(batchSize, fast_multiplier)
    
    #conv autoencoder
    encoder = OptimalEncoder(inputShape, batchSize, latentSize, intermediateSize, 'vae', beta=69, capacity=15, randomSample=True)
    decoder = OptimalDecoder(inputShape, batchSize, latentSize, intermediateSize)
    #encoder = ConvEncoder(inputShape, batchSize, latentSize, 'vae', beta=69, capacity=15, randomSample=True)
    #decoder = ConvDecoder(inputShape, batchSize, latentSize)
    bvae = AutoEncoder(encoder, decoder)

    bvae.ae.compile(optimizer='adam', loss='binary_crossentropy')
    #bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    iteration_number = 0

    while iteration_number <= iterations:
        if os.name == 'nt':
            bvae.ae.fit_generator(manager.generate_images(), steps_per_epoch=ntrain, workers=1, validation_data=next(manager.generate_images()), validation_steps=nval, epochs=1,verbose=1)
        else:
            #bvae.ae.fit_generator(manager.generate_images(), steps_per_epoch=ntrain, max_queue_size=30, workers=16, use_multiprocessing=True, validation_data=next(manager.generate_images()), validation_steps=nval, epochs=1,verbose=1,callbacks=[ChangeMetrics(), tensorboard])
            bvae.ae.fit_generator(manager.generate_images_fast_randomshift(), steps_per_epoch=ntrain, max_queue_size=30, workers=16, use_multiprocessing=True, validation_data=next(manager.generate_images()), validation_steps=nval, epochs=1,verbose=1,callbacks=[ChangeMetrics(), tensorboard])
            #bvae.ae.fit_generator(manager.generate_images(), steps_per_epoch=ntrain, max_queue_size=30, workers=16, use_multiprocessing=True, validation_data=next(manager.generate_images()), validation_steps=nval, epochs=1,verbose=1,callbacks=[ChangeMetrics(), tensorboard])

        if iteration_number % 20 == 0:
            img = manager.get_images(batchSize)
            latentVec = bvae.encoder.predict(img, batch_size=batchSize)[0]
            #print(latentVec)
            print(str(iteration_number) + ' ' + time.ctime())
            train = img[0] #get a sample image
            train = np.uint8(train* 255) # convert to regular image values
            train = Image.fromarray(train)
            train.save(os.path.join(outputs_folder,str(iteration_number)+'_train'+'.png'))
            
            pred = bvae.ae.predict(img, batch_size=batchSize)[0] # get the reconstructed image
            pred = np.uint8(pred * 255) # convert to regular image values
            pred = Image.fromarray(pred)
            pred.save(os.path.join(outputs_folder,str(iteration_number)+'_pred'+'.png'))

            #bvae.ae.save(os.path.join(output_models_folder, str(iteration_number)+'_autoencoder.h5'))
            #bvae.decoder.save(os.path.join(output_models_folder, str(iteration_number)+'_decoder.h5'))
            #bvae.encoder.save(os.path.join(output_models_folder, str(iteration_number)+'_encoder.h5'))
        if iteration_number == iterations:
            bvae.ae.save(os.path.join(output_models_folder, str(iteration_number)+'_autoencoder.h5'))
            
        iteration_number+=1
        #check in once n iterations
        
    if os.name != 'nt':
        os.system('sudo shutdown -h now')
