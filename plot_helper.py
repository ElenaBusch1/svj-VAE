#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from root_to_numpy import variable_array
from math import ceil

tag = 'signalEval'
plot_dir = '/a/home/kolya/ebusch/WWW/SVJ/autoencoder/'

def plot_loss(h,i):
  print(h.history)
  plt.plot(h.history['loss'])
  plt.plot(h.history['val_loss'])
  plt.title('AE MSE Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(plot_dir+'lossVsEpoch'+tag+str(i)+'.png')
  plt.clf()

def make_roc(fpr,tpr,auc):
  plt.plot(fpr,tpr,label="AUC = %0.2f" % auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("SVJ AE ROC")
  plt.legend()
  plt.savefig(plot_dir+'roc_auc'+tag+'.png')
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

def plot_score(bkg_score, sig_score):
  bins=np.histogram(np.hstack((bkg_score,sig_score)),bins=40)[1]
  plt.hist(bkg_score, bins=bins, alpha=0.5, label="bkg", density=True)
  plt.hist(sig_score, bins=bins, alpha=0.5, label="sig", density=True)
  plt.yscale('log')
  plt.legend()
  plt.xlabel('MSE(Input, Reconstructed)')
  plt.savefig(plot_dir+'score'+tag+'.png')
  plt.clf()

def plot_inputs(bkg, sig):
  for i in range(len(variable_array)):
    plt.subplot(2,2,i%4+1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.hist(bkg[:,i], bins=30, alpha=0.5, density=True)  
    plt.hist(sig[:,i], bins=30, alpha=0.5, density=True)
    plt.title(variable_array[i])
    if (i%4 == 3):
      plt.savefig(plot_dir+'input_vars_'+str(i)+tag+'.png')
      plt.clf()

def plot_vectors(bkg,sig,extra_tag):
  variable_array = ["pT + MET", "eta", "phi", "E"]
  for i in range(4):
    bkg_v = bkg[:,i::4].flatten()
    sig_v = sig[:,i::4].flatten()
    bins=np.histogram(np.hstack((bkg_v,sig_v)),bins=40)[1]
    plt.subplot(2,2,i+1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.hist(bkg_v, alpha=0.5, label="bkg", bins=bins)
    plt.hist(sig_v, alpha=0.5, label="sig", bins=bins)
    plt.yscale('log')
    plt.title(variable_array[i])
  plt.savefig(plot_dir+'v_kin_'+tag+extra_tag+'.png')
  plt.clf()



