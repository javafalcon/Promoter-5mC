# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:08:27 2020

@author: lwzjc
"""
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import callbacks
from Capsule import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
from sklearn.utils import class_weight, shuffle, resample


def CapsNet(input_shape, n_class, num_routing, kernel_size=7):
    """
    A Capsule Network on PDNA-543.
    :param input_shape: data shape, 3d, [width, height, channels],for example:[21,28,1]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=128, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=16, n_channels=32, kernel_size=kernel_size, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=32, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)
    
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])
    

def semisup_margin_loss(y_true, y_pred):  
    m = K.sum(y_true, axis=-1)
    #return  K.switch(K.equal(K.sum(y_true), 0), 0., K.sum(K.categorical_crossentropy(K.tf.boolean_mask(y_true,m), K.tf.boolean_mask(y_pred,m), from_logits=True)) / K.sum(y_true))
    t = tf.boolean_mask(y_true,m)
    p = tf.boolean_mask(y_pred,m)
    L = t * K.square(K.maximum(0.,0.9 - p)) + \
        0.5 * (1 - t) * K.square(K.maximum(0., p - 0.1))
    return  K.switch(K.equal(K.sum(y_true), 0), 0., K.mean(K.sum(L, 1)))

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[semisup_margin_loss,'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': "accuracy"})
    
    # Training without data augmentation:
    
    model.fit([x_train, y_train], [y_train, x_train], 
              batch_size=args.batch_size, 
              epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], 
              callbacks=[log, tb, checkpoint, lr_decay])
    """
    #cw = class_weight.compute_class_weight('balanced',np.unique(y_train), y_train)
    #cw = dict(enumerate(cw))
    model.fit(x_train, y_train, batch_size=args.batch_size, 
              epochs=args.epochs,
              validation_data=[x_test, y_test], 
              #class_weight=cw,
              callbacks=[log, tb, checkpoint, lr_decay])
    """
    """
    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    """
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-'*50)
    y_p = np.argmax(y_pred, 1)
    y_t = np.argmax(y_test,1)
    print('Test Accuracy:', accuracy_score(y_t, y_p))
    print('Test mattews-corrcoef', matthews_corrcoef(y_t, y_p))
    return y_p
    """
    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()
    """


if __name__ == "__main__":
    #import numpy as np
    import os
    #from keras.preprocessing.image import ImageDataGenerator
    #from keras.utils.vis_utils import plot_model

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lam_recon', default=0.465, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=9, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load data
    x_train, y_train = dataset.load_CAImage()
    
    N = 9345*4
    (x_train, y_train) = load_semisupTrain(traindatafile,N)
    (x_test, y_test) = load_test(testdatafile)
    
    y_pred = np.zeros(shape=(y_test.shape[0],))
    ker=[3,5,7,9,11]
    for k in range(len(ker)):
        
        print("predictor No.{}：x_train.shape：{}".format(k, x_train.shape))
        # define model
        model = CapsNet(input_shape=x_train.shape[1:],
                        n_class=len(np.unique(np.argmax(y_train, 1))),
                        num_routing=args.num_routing, kernel_size=ker[k])
        model.summary()
        #plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)
    
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    
        y_p = test(model=model, data=(x_test, y_test))
        y_pred = y_pred + y_p
        K.clear_session()
        tf.reset_default_graph()
        #(x_train, y_train), (x_test, y_test) = load_PDNA543_hhm()
        (x_train, y_train) = load_semisupTrain(traindatafile,N)
    
    y_pred = y_pred/len(ker)
    y_p = (y_pred>0.5).astype(float)
    y_t = np.argmax(y_test,1)
    print('Test Accuracy:', accuracy_score(y_t, y_p))
    print('Test mattews-corrcoef', matthews_corrcoef(y_t, y_p))
    print('Test confusion-matrix', confusion_matrix(y_t, y_p))
    """
    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=model, data=(x_test, y_test))
   """