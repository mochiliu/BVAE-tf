'''
models.py
contains models for use with the BVAE experiments.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (InputLayer, Dense, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
            Reshape, GlobalAveragePooling2D)
from tensorflow.python.keras.models import Model

from model_utils import SampleLayer

class Architecture(object):
    '''
    generic architecture template
    '''
    def __init__(self, inputShape=None, batchSize=None, latentSize=None):
        '''
        params:
        ---------
        inputShape : tuple
            the shape of the input, expecting 3-dim images (h, w, 3)
        batchSize : int
            the number of samples in a batch
        latentSize : int
            the number of dimensions in the two output distribution vectors -
            mean and std-deviation
        '''
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize
        self.intermediate_dim = 512

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=1,
                 latentSize=1000, latentConstraints='bvae', beta=100., capacity=0.,
                 randomSample=True):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use 
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.latentCapacity = capacity
        self.randomSample = randomSample
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
                
        net = Dense(self.intermediate_dim, activation='relu')(inLayer)

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        stddev = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        stddev = GlobalAveragePooling2D()(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                            self.latentCapacity, self.randomSample)([mean, stddev])

        return Model(inputs=inLayer, outputs=sample)

class Decoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=1, latentSize=1000):
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Dense(self.intermediate_dim, activation='relu')(inLayer)
        net = Dense(np.prod(self.inputShape), activation='sigmoid')(net)
        net = Reshape(self.inputShape)(net)
        
        return Model(inLayer, net)


#
#if __name__ == '__main__':
#    test()
