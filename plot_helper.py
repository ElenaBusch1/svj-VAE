#!/usr/bin/env python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from root_to_numpy import variable_array
from math import ceil

tag = "2jets"
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

def plot_var(x_dict, x_cut1, x_cut2, key):
  #bmax = max(max(x_orig),max(y_orig))
  #bmin = min(min(x_orig),min(y_orig))
  #bins=np.histogram(np.hstack((x_dict[key],x_cut1[key])),bins=20)[1]
  bins= np.linspace(0,8000,20)
  fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
  h1 = ax[0].hist(x_dict[key], bins=bins, weights=1.39e8*x_dict['weight'], alpha=0.5, label="Full bkg", color='dimgray')
  h2 = ax[0].hist(x_cut1[key], bins=bins, weights=1.39e8*x_cut1['weight'], histtype='step',  label="Cut - 50%", color = 'mediumblue')
  h3 = ax[0].hist(x_cut2[key], bins=bins, weights=1.39e8*x_cut2['weight'], histtype='step', label="Cut - 2%", color = 'forestgreen')
  ax[0].set_yscale('log')
  ax[0].set_ylabel('Events')
  ax[0].legend()
  ax[0].set_title(key + "; 50% and 2% Cuts")
  #plt.subplot(2,1,2)
  ax[1].plot(bins[:-1],np.ones(len(bins)-1), linestyle='dashed', color = 'dimgray')
  ax[1].plot(bins[:-1],2*h2[0]/h1[0], drawstyle='steps', color='mediumblue')
  ax[1].plot(bins[:-1],50*h3[0]/h1[0], drawstyle='steps', color='forestgreen')
  ax[1].set_ylim(0,2)
  ax[1].set_xlabel('GeV')
  ax[1].set_ylabel('Ratio * (1/cut)')
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
  #bins = np.linspace(0,10000,80)
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

def plot_phi(phis,name,extra_tag):
  nphis = phis.shape[1]
  nevents = phis.shape[0]
  idx = [i for i in range(nphis)]*nevents

  phiT = phis.T
  print("n zeros = ", len(np.where(~phiT.any(axis=1))[0]))
  phis = phis.flatten()
  nbinsx = 10
  bin_width = max(phis)/nbinsx
  phis[phis==0] = -bin_width 

  fig, ax = plt.subplots()
  h = ax.hist2d(phis,idx,bins=[nbinsx+1,nphis])
  fig.colorbar(h[3], ax=ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Index')
  ax.set_title('PFN Set Representation - '+name)
  plt.savefig(plot_dir+'phi2D_'+name+'_'+extra_tag+'_'+tag+'.png')
  print("Saved 2D plot of phi-rep for", extra_tag)

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

def get_nTracks(x):
  n_tracks = []
  for i in range(x.shape[0]):
    tracks = x[i,:,:].any(axis=1)
    tracks = tracks[tracks == True]
    n_tracks.append(len(tracks))
  return n_tracks
 
def plot_nTracks(bkg, sig):
  bkg_tracks = get_nTracks(bkg)
  sig_tracks = get_nTracks(sig)
  #bins=np.histogram(np.hstack((bkg_tracks,sig_tracks)),bins=60)[1]
  bins = np.arange(0,50,1)
  plt.hist(bkg_tracks,alpha=0.5, label="bkg", bins=bins, density=False)
  plt.hist(sig_tracks,alpha=0.5, label="sig", bins=bins, density=False)
  plt.title("nTracks (after pT>10) - Tertiary")
  plt.legend()
  plt.savefig(plot_dir+'nTracks_'+tag+'.png')
  plt.clf()
  print("Saved plot of nTracks")

def plot_vectors(train,sig,extra_tag):
  variable_array = ["pT + MET", "eta", "phi", "E"]
  for i in range(4):
    train_v = train[:,i].flatten()
    #test_v = test[:,i::4].flatten()
    sig_v = sig[:,i].flatten()
    bins=np.histogram(np.hstack((train_v,sig_v)),bins=60)[1]
    if(bins[-1] > 3000): bins = np.arange(0,3000,50)
    plt.subplot(2,2,i+1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.hist(train_v, alpha=0.5, label="bkg", bins=bins, density=False)
    #plt.hist(test_v, alpha=0.5, label="test", bins=bins, density=True, color='lightskyblue')
    plt.hist(sig_v, alpha=0.5, label="sig", bins=bins, density=False)
    plt.yscale('log')
    plt.title(variable_array[i])
    if i == 1: plt.legend()
  plt.savefig(plot_dir+'inputs_'+extra_tag+'_'+tag+'.png')
  plt.clf()
  print("Saved inputs plot (", extra_tag, ")")



