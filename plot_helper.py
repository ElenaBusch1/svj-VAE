#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(h):
      print(h.history)
      plt.plot(h.history['loss'])
      plt.plot(h.history['val_loss'])
      np.save(save+"_loss",h.history['loss'])
      np.save(save+"_valloss",h.history['val_loss'])
      plt.title('AE MSE Loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig('lossVsEpoch.pdf')
      plt.clf()
