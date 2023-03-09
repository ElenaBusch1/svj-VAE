#!/usr/bin/env python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from root_to_numpy import variable_array
from math import ceil

tag = "znn_32"
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
  plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'log.png')
  plt.clf()
  print("Saved loss plot for ", model, loss)

def plot_var(x_dict, y_dict, x_cut, y_cut, key):
  #bmax = max(max(x_orig),max(y_orig))
  #bmin = min(min(x_orig),min(y_orig))
  bins=np.histogram(np.hstack((x_dict[key],y_dict[key])),bins=80)[1]
  plt.hist(x_dict[key], bins=bins, weights=x_dict['weight'], alpha=0.5, label="Full bkg")
  plt.hist(y_dict[key], bins=bins, weights=y_dict['weight'], histtype='step', label="Full sig")
  plt.hist(x_cut[key], bins=bins, weights=x_cut['weight'], histtype='step',  label="Cut bkg", color = 'k')
  plt.hist(y_cut[key], bins=bins, weights=y_cut['weight'], histtype='step', label="Cut sig", color = 'r')
  plt.yscale('log')
  plt.legend()
  plt.title(key + "; Before & After 50% cut")
  plt.xlabel('GeV')
  plt.savefig(plot_dir+key+'_'+tag+'.png')
  plt.clf()
  print("Saved cut distribution for", key)

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
  #bkg_score = np.absolute(bkg_score)
  #sig_score = np.absolute(sig_score)
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
    train_v = train[:,i].flatten()
    #test_v = test[:,i::4].flatten()
    sig_v = sig[:,i].flatten()
    bins=np.histogram(np.hstack((train_v,sig_v)),bins=60)[1]
    #if(bins[-1] > 3000): bins = np.arange(0,3000,50)
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



