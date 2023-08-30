#import imageio
import glob
import os
import time
#import cv2
import tensorflow as tf
from tensorflow.keras import layers
#from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from svj_antelope import Param_ANTELOPE
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.
# Batch and shuffle the data
print(x_train.shape)
if __name__=="__main__":
  ls_sig=[20000]
  ls_bkg=[200000]
  for sig_events, bkg_events in zip(ls_sig, ls_bkg):
    param1=Param_ANTELOPE(pfn_model=pfn_model,bkg_events=bkg_events, sig_events=sig_events, h5_dir='h5dir/antelope/aug17_jetpt/', arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'])
    stdoutOrigin=param1.open_print()
    all_dir, auc,bkg_events_num,sig_events_num=param1.evaluate_vae()
    setattr(param1, 'auc',auc )
    setattr(param1, 'sig_events_num',sig_events_num )
    setattr(param1, 'bkg_events_num',bkg_events_num )
    print(param1.close_print(stdoutOrigin))
    print(param1.save_info())

""" 
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(128)
figsize = 15
m, v = enc.predict(x_test[:25])
latent = sampling([m,v])
reconst = dec.predict(latent)
 
fig = plt.figure(figsize=(figsize, 10))
 
for i in range(25):
    ax = fig.add_subplot(5, 5, i+1)
    ax.axis('off')
    ax.text(0.5, -0.15, str(label_dict[y_test[i]]), fontsize=10, ha='center', transform=ax.transAxes)
     
    ax.imshow(reconst[i, :,:,0]*255, cmap = 'gray')

""" 

# Can I construct the image accurately? compare the original vs new

# apply scaling # is it already applied
# plot vectors
# just the VAE
# choose validation and train
# vae fitting
# vae save
# reconstruct

# PFN and VAE
