#!/usr/bin/env python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from root_to_numpy import variable_array
from math import ceil

tag = "hlv_ae_compare"
plot_dir = '/a/home/kolya/ebusch/WWW/SVJ/autoencoder/'

def detect_outliers(x):
  z = np.abs(stats.zscore(x))
  print(max(z))
  x_smooth = x[z<40]
  n_removed = len(x)-len(x_smooth)
  print(n_removed, " outliers removed")
  return x_smooth, n_removed

def plot_loss(h, model="", loss='loss'):
  #print(h.history)
  plt.plot(h.history[loss])
  plt.plot(h.history['val_'+loss])
  plt.title(model+' '+loss)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'.png')
  plt.clf()
  print("Saved loss plot for ", model, loss)

def plot_saved_loss(h, model="", loss='loss'):
  plt.plot(h[loss])
  plt.plot(h['val_'+loss])
  plt.title(model+' '+loss)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  #plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'linear.png')
  plt.clf()
  print("Saved loss plot for ", model, loss)


def make_roc(fpr,tpr,auc,model=""):
  plt.plot(fpr,tpr,label="AUC = %0.2f" % auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("SVJ "+model+" ROC")
  plt.legend()
  plt.savefig(plot_dir+'roc_'+model+'_'+tag+'.png')
  plt.clf()
  print("Saved ROC curve for model", model)

def make_sic(fpr,tpr,auc, model=""):
  y = tpr[1:]/np.sqrt(fpr[1:])
  plt.plot(tpr[1:],y,label="AUC = %0.2f" % auc)
  plt.axhline(y=1, color='0.8', linestyle='--')
  plt.xlabel("Signal Efficiency (TPR)")
  plt.ylabel("Signal Sensitivity ($TPR/\sqrt{FPR}$)")
  plt.title("Significance Improvement Characteristic: "+model )
  plt.legend()
  plt.savefig(plot_dir+'sic_'+model+'_'+tag+'.png')
  plt.clf()
  print("Saved SIC for", model)


def make_single_roc(rocs,aucs,ylabel):
  plt.plot(rocs[0],rocs[1],label=str(np.round(r,4))+", $\sigma$="+str(sigs)+": AUC="+str(np.round(aucs,3)))
  plt.xlabel('fpr')
  plt.ylabel(Ylabel)
  plt.title('ROC: '+saveTag)
  plt.figtext(0.7,0.95,"size="+str(sizeeach)+", nvars="+str(nInputs))
  plt.legend()
  plt.savefig(saveTag+'_roc_aucs_'+Ylabel.replace("/","")+'.pdf')
  plt.clf()

def plot_score(bkg_score, sig_score, remove_outliers=True, xlog=True, extra_tag=""):
  if remove_outliers:
    bkg_score,nb = detect_outliers(bkg_score)
    sig_score,ns = detect_outliers(sig_score)
  #bins=np.histogram(np.hstack((bkg_score,sig_score)),bins=80)[1]
  bkg_score = np.absolute(bkg_score)
  sig_score = np.absolute(sig_score)
  bmax = max(max(bkg_score),max(sig_score))
  bmin = min(min(bkg_score),min(sig_score))
  if xlog and bmin == 0: bmin = 1e-9
  if xlog: bins = np.logspace(np.log10(bmin),np.log10(bmax),80)
  else: bins=np.histogram(np.hstack((bkg_score,sig_score)),bins=80)[1]
  #plt.hist(bkg_score, bins=bins, alpha=0.5, label="bkg (-"+str(nb)+")", density=True)
  #plt.hist(sig_score, bins=bins, alpha=0.5, label="sig(-"+str(ns)+")", density=True)
  plt.hist(bkg_score, bins=bins, alpha=0.5, label="bkg", density=True)
  plt.hist(sig_score, bins=bins, alpha=0.5, label="sig", density=True)
  if xlog: plt.xscale('log')
  plt.yscale('log')
  plt.legend()
  plt.title("Anomaly Score " + extra_tag)
  plt.xlabel('Loss')
  plt.savefig(plot_dir+'score_'+extra_tag+'_'+tag+'.png')
  plt.clf()
  print("Saved score distribution for", extra_tag)

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

def plot_vectors(train,sig,extra_tag):
  variable_array = ["pT + MET", "eta", "phi", "E"]
  for i in range(4):
    train_v = train[:,i::4].flatten()
    #test_v = test[:,i::4].flatten()
    sig_v = sig[:,i::4].flatten()
    bins=np.histogram(np.hstack((train_v,sig_v)),bins=40)[1]
    plt.subplot(2,2,i+1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.hist(train_v, alpha=0.5, label="bkg", bins=bins, density=True)
    #plt.hist(test_v, alpha=0.5, label="test", bins=bins, density=True, color='lightskyblue')
    plt.hist(sig_v, alpha=0.5, label="sig", bins=bins, density=True)
    plt.yscale('log')
    plt.title(variable_array[i])
    if i == 1: plt.legend()
  plt.savefig(plot_dir+'inputs_'+extra_tag+'_'+tag+'.png')
  plt.clf()
  print("Saved inputs (", extra_tag, ")")



