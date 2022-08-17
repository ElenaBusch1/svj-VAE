#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

plot_dir = 'plots/'

def plot_loss(h):
      print(h.history)
      plt.plot(h.history['loss'])
      plt.plot(h.history['val_loss'])
      plt.title('AE MSE Loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig(plot_dir+'lossVsEpoch.pdf')
      plt.clf()

def make_single_roc(rocs,aucs,ylabel):
  plt.plot(rocs[0],rocs[1],label=str(np.round(r,4))+", $\sigma$="+str(sigs)+": AUC="+str(np.round(aucs,3)))
  plt.xlabel('fpr')
  plt.ylabel(Ylabel)
  plt.title('ROC: '+saveTag)
  plt.figtext(0.7,0.95,"size="+str(sizeeach)+", nvars="+str(nInputs))
  plt.legend()
  plt.savefig(saveTag+'_roc_aucs_'+Ylabel.replace("/","")+'.pdf')
  plt.clf()
