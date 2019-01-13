# coding: utf-8
import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
#from keras.models import load_model

import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img
from models import Darknet19Encoder, Darknet19Decoder

class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))

fit_path = 'C:\\Users\\Mochi\\Desktop\\fit\\'
epoch_number = '9'
encoder_model_path = os.path.join(fit_path + epoch_number + '_encoder.h5')
decoder_model_path = os.path.join(fit_path + epoch_number + '_decoder.h5')
ae_model_path = os.path.join(fit_path + epoch_number +  '_autoencoder.h5')

inputShape = (256, 256, 3)
batchSize = 10
latentSize = 100

img = load_img(os.path.join('..','images', 'img.jpg'), target_size=inputShape[:-1])
img.show()

img = np.array(img, dtype=np.float32) / 255 - 0.5
img = np.array([img]*batchSize) # make fake batches to improve GPU utilization

# This is how you build the autoencoder
encoder = Darknet19Encoder(inputShape, batchSize, latentSize, 'bvae', beta=69, capacity=15, randomSample=False)
decoder = Darknet19Decoder(inputShape, batchSize, latentSize)
bvae = AutoEncoder(encoder, decoder)

bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')

bvae.ae.load_weights(ae_model_path)
bvae.encoder.load_weights(encoder_model_path)
bvae.decoder.load_weights(decoder_model_path)
#
#bvae.ae.summary()
#bvae.encoder.summary()
#bvae.decoder.summary()

# example retrieving the latent vector
latentVec = bvae.encoder.predict(img)
print(latentVec[0])


for dim in range(10):
    for shift in range(-100, 101, 50):
        latentVecPrime = latentVec.copy()
        latentVecPrime[0][dim] = latentVecPrime[0][dim] + shift
        #latentVecPrime[0][1] = latentVecPrime[0][1] + 5
        pred = bvae.decoder.predict(latentVecPrime) # get the reconstructed image
        pred[pred > 0.5] = 0.5 # clean it up a bit
        pred[pred < -0.5] = -0.5
        pred = np.uint8((pred + 0.5)* 255) # convert to regular image values
        
        pred_img = Image.fromarray(pred[0])
    #    pred_img.show()
        pred_img.save(fit_path + epoch_number + 'epoch_' + str(dim) + 'dim_' + str(shift) + 'shift.bmp')