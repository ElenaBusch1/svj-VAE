#!/usr/bin/env python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from root_to_numpy import variable_array
from math import ceil
from termcolor import cprint
#tag = "PFN_2jAvg_MM"
#plot_dir = '/a/home/kolya/ebusch/WWW/SVJ/autoencoder/'
#plot_dir = '/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun5_500000_nep100_nl100/'
plot_dir = '/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun9/'
def detect_outliers(x):
  z = np.abs(stats.zscore(x))
  print(max(z))
  x_smooth = x[z<40]
  n_removed = len(x)-len(x_smooth)
  print(n_removed, " outliers removed")
  return x_smooth, n_removed

def plot_loss(h, loss='loss', tag_file="", tag_title=""):
  #print(h.history)
  plt.plot(h.history[loss])
  plt.plot(h.history['val_'+loss])
  plt.title(loss +f' {tag_title}')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
#  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'.png')
  plt.savefig(plot_dir+loss+'VsEpoch_'+tag_file+'.png')
  plt.clf()
  print("Saved loss plot for ", tag_file, loss)

def plot_saved_loss(h, loss='loss', tag_file="", tag_title=""):
  plt.plot(h[loss])
  plt.plot(h['val_'+loss])
  plt.title(loss +f' {tag_title}')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
  #plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'log.png')
  plt.savefig(plot_dir+loss+'VsEpoch_'+tag_file+'log.png')
  plt.clf()
  print("Saved loss plot for ", tag_file)

def plot_var(x_dict, x_cut1, x_cut2, key , tag_file="", tag_title=""):
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
  ax[0].set_title(key + "; 50% and 2% Cuts"+f' {tag_title}')
  #plt.subplot(2,1,2)
  ax[1].plot(bins[:-1],np.ones(len(bins)-1), linestyle='dashed', color = 'dimgray')
  ax[1].plot(bins[:-1],2*h2[0]/h1[0], drawstyle='steps', color='mediumblue')
  ax[1].plot(bins[:-1],50*h3[0]/h1[0], drawstyle='steps', color='forestgreen')
  ax[1].set_ylim(0,2)
  ax[1].set_xlabel('GeV')
  ax[1].set_ylabel('Ratio * (1/cut)')
  plt.savefig(plot_dir+key+'_'+tag_file+'.png')
  plt.clf()
  print("Saved cut distribution for", key)

def make_roc(fpr,tpr,auc, tag_file="", tag_title=""):
  plt.plot(fpr,tpr,label="AUC = %0.2f" % auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("SVJ "+" ROC" +f' {tag_title}')
  plt.legend()
  plt.savefig(plot_dir+'roc_'+tag_file+'.png')
  plt.clf()
  print("Saved ROC curve for ", tag_file)

def make_sic(fpr,tpr,auc,  tag_file="", tag_title=""):
  y = tpr[1:]/np.sqrt(fpr[1:])
  plt.plot(tpr[1:],y,label="AUC = %0.2f" % auc)
  plt.axhline(y=1, color='0.8', linestyle='--')
  plt.xlabel("Signal Efficiency (TPR)")
  plt.ylabel("Signal Sensitivity ($TPR/\sqrt{FPR}$)")
  plt.title("Significance Improvement Characteristic: " +f' {tag_title}')
  plt.legend()
  plt.savefig(plot_dir+'sic_'+tag_file+'.png')
  plt.clf()
  print("Saved SIC for", tag_file)


def make_single_roc(rocs,aucs,ylabel, tag_file="", tag_title=""):
  plt.plot(rocs[0],rocs[1],label=str(np.round(r,4))+", $\sigma$="+str(sigs)+": AUC="+str(np.round(aucs,3)))
  plt.xlabel('fpr')
  plt.ylabel(Ylabel)
  plt.title('ROC: '+saveTag+f' {tag_title}')
  plt.figtext(0.7,0.95,"size="+str(sizeeach)+", nvars="+str(nInputs))
  plt.legend()
  plt.savefig(saveTag+'_roc_aucs_'+Ylabel.replace("/","")+tag_file+'.pdf')
  plt.clf()

def plot_score(bkg_score, sig_score, remove_outliers=True, xlog=True, tag_file="", tag_title=""):
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
  #bins = np.linspace(0,5e-3,80)
  #plt.hist(bkg_score, bins=bins, alpha=0.5, label="bkg (-"+str(nb)+")", density=True)
  #plt.hist(sig_score, bins=bins, alpha=0.5, label="sig(-"+str(ns)+")", density=True)
  plt.hist(bkg_score, bins=bins, alpha=0.5, label="bkg", density=True)
  plt.hist(sig_score, bins=bins, alpha=0.5, label="sig", density=True)
  if xlog: plt.xscale('log')
  plt.yscale('log')
  plt.legend()
  plt.title(f'Anomaly Score {tag_title}')
  plt.xlabel('Loss')
  plt.savefig(plot_dir+'score_'+tag_file+'.png')
  plt.clf()
  print("Saved score distribution for", tag_file)

def plot_phi(phis,name,extra_tag, tag_file="", tag_title=""):
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
  ax.set_title('PFN Set Representation - '+name +f' {tag_title}')
  plt.savefig(plot_dir+'phi2D_'+name+'_'+tag_file+'.png')
  print("Saved 2D plot of phi-rep for", tag_file)

def plot_inputs(bkg, sig, tag_file="", tag_title=""):
  for i in range(len(variable_array)):
    plt.subplot(2,2,i%4+1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.hist(bkg[:,i], bins=30, alpha=0.5, density=True)  
    plt.hist(sig[:,i], bins=30, alpha=0.5, density=True)
    plt.title(variable_array[i]+f' {tag_title}')
    if (i%4 == 3):
      plt.savefig(plot_dir+'input_vars_'+str(i)+tag_file+'.png')
      plt.clf()

def zero_div(a,b, bool_print=False): # this avoid zero division error
    a= np.array(a, dtype='float') # necessary to avoid "TypeError: No loop matching the specified signature and casting was found for ufunc add"
    b= np.array(b, dtype='float')
    if bool_print: print(a,b)
    mask=(b!=0) # where b is not zero, do the division, if it's zero, spit out the corresponding b element 
    #e.g. a= [2 2 4], b= [1 0 3], result [2 0 1.333]
    return np.divide(a, b, out=np.zeros_like(a), where=mask) 

def plot_single_variable(hists, h_names, weights_ls,title,density_top=True, logy=False, len_ls=[]):
  nbins=50
  hists_flat=np.concatenate(hists)
  bin_min=np.min(hists_flat)
  bin_max=np.max(hists_flat)
  gap=(bin_max-bin_min)*0.05
  bins=np.linspace(bin_min-gap,bin_max+gap,nbins)
  x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
  hists=list(hists)

  cut0_idx=0
  len0=len(hists[cut0_idx])
 
  ratio_all=np.array([]) 
  for data,name,weights,i in zip(hists,h_names,weights_ls, range(len(hists))):
    y,_, _=plt.hist(data, bins=bins, weights=weights,density=density_top,histtype='step', alpha=0.7, label=f'NE={len(data)}, {round(len(data)/len_ls[i]*100,1)}% left, cut={name}')
    #y,_, _=plt[0].hist(data, bins=bins, weights=weights,density=density_top,histtype='step', alpha=0.7, label=f'NE={len(data)}, {round(len(data)/len0*100,1)}% left, cut={name}')
#    y_unnorm,_, _=plt[0].hist(data, bins=bins, density=False,histtype='step', alpha=0)
#    print(i, len(bins), len(y), bins, y) 
    #if i ==len(hists)-1:
  plt.tick_params(axis='y', which='minor') 
  plt.grid()
 
  plt.ylabel('Event Number')
  if (logy): plt.yscale("log")
  plt.legend(loc='lower right')
  #plt.legend(loc='upper right')
  plt.title(title)

  plt.savefig(plot_dir+'hist_'+title.replace(" ","").replace('(','')+'_weighted1'+'.png')
  plt.clf()
  print("Saved plot",title)



def plot_single_variable_ratio(hists, h_names, weights_ls,title,density_top=True, logy=False):
  f, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
  nbins=50
  hists_flat=np.concatenate(hists)
  bin_min=np.min(hists_flat)
  bin_max=np.max(hists_flat)
  gap=(bin_max-bin_min)*0.05
  bins=np.linspace(bin_min-gap,bin_max+gap,nbins)
  x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
  hists=list(hists)

  cut0_idx=0
  len0=len(hists[cut0_idx])
 
  ratio_all=np.array([]) 
  for data,name,weights,i in zip(hists,h_names,weights_ls, range(len(hists))):
    y,_, _=axs[0].hist(data, bins=bins, weights=weights,density=density_top,histtype='step', alpha=0.7, label=f'NE={len(data)}, {round(len(data)/len_ls[i],1)*100}% left, cut={name}')
    #y,_, _=axs[0].hist(data, bins=bins, weights=weights,density=density_top,histtype='step', alpha=0.7, label=f'NE={len(data)}, {round(len(data)/len0*100,1)}% left, cut={name}')
#    y_unnorm,_, _=axs[0].hist(data, bins=bins, density=False,histtype='step', alpha=0)
#    print(i, len(bins), len(y), bins, y) 
    #if i ==len(hists)-1:
    if i ==cut0_idx: # make sure the first of hists list has the most number of events
      y0=y
      #y0=y_unnorm
#      axs[0].set_ylim([min(y)/1e2, max(y)*1e2])
    axs[1].scatter(x_bins,y/y0)
#    axs[1].set_ylim([])
    #axs[1].scatter(x_bins,zero_div(y,y0))
    ratio_all
  axs[1].set_ylim(0.5,3)  
  axs[1].set_ylabel('Ratio')
  axs[1].legend(loc='upper right')
  plt.tick_params(axis='y', which='minor') 
  plt.grid()
 
  axs[0].set_ylabel('Event Number')
  if (logy): axs[0].set_yscale("log")
  axs[0].legend(loc='upper right')
  axs[1].legend(loc='upper right')
  axs[0].set_title(title)

  plt.savefig(plot_dir+'hist_'+title.replace(" ","").replace('(','')+'_weighted'+'.png')
  plt.clf()
  print("Saved plot",title)



def get_nTracks(x):
  n_tracks = []
  for i in range(x.shape[0]):
    tracks = x[i,:,:].any(axis=1)
    tracks = tracks[tracks == True]
    n_tracks.append(len(tracks))
  return n_tracks
 
def plot_nTracks(bkg, sig, tag_file="", tag_title=""):
  bkg_tracks = get_nTracks(bkg)
  sig_tracks = get_nTracks(sig)
  #bins=np.histogram(np.hstack((bkg_tracks,sig_tracks)),bins=60)[1]
  bins = np.arange(0,50,1)
  plt.hist(bkg_tracks,alpha=0.5, label="bkg", bins=bins, density=False)
  plt.hist(sig_tracks,alpha=0.5, label="sig", bins=bins, density=False)
  plt.title("nTracks (after pT>10) - Tertiary"+f' {tag_title}')
  plt.legend()
  plt.savefig(plot_dir+'nTracks_'+'_'+tag_file+'.png')
  plt.clf()
  print("Saved plot of nTracks")

def plot_vectors_jet(train,sig,jet_array, tag_file="", tag_title="", bool_jet=False):
#  variable_array=["pt"]
  print('before reshaping, train and sig', train.shape, sig.shape)
#  size=2 # different than size in plot_vectors
  size=len(jet_array) # different than size in plot_vectors
  n=2
  for i in range(size):
    #train_v= train[:,i,0]
    train_v= train[:,i%n,i//n]
    sig_v= sig[:,i%n,i//n]
    bins=np.histogram(np.hstack((train_v,sig_v)),bins=60)[1] 
    plt.hist(train_v, alpha=0.5, label=f"bkg ({len(train_v)})", bins=bins, density=False)
    plt.hist(sig_v, alpha=0.5, label=f"sig ({len(sig_v)})", bins=bins, density=False)
    plt.title(f'{jet_array[i]} {tag_title}')
    #plt.title(f'jet{i+1}_{variable_array[0]} {tag_title}')
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.yscale('log')
    plt.legend()
    plt.savefig(plot_dir+f'inputs_{jet_array[i]}'+tag_file+'.png')
    #plt.savefig(plot_dir+f'inputs_jet{i+1}_{variable_array[0]}'+tag_file+'.png')
    plt.cla()
    plt.clf()
    plt.close()
    print("Saved inputs plot (", plot_dir+'inputs_jet_'+tag_file+'.png')

def plot_vectors(train,sig, tag_file="", tag_title="", bool_one=True):
  #variable_array = ["pT", "eta", "phi", "E"]
  variable_array = ["pT", "eta", "phi", "E", "z0", "d0", "qOverP"]
  print('before reshaping, train and sig', train.shape, sig.shape)
 
  if (len(train.shape) == 3):
    train = train.reshape(train.shape[0], train.shape[1] * train.shape[2])
  if (len(sig.shape) == 3):
    sig = sig.reshape(sig.shape[0], sig.shape[1] * sig.shape[2])
  print('after reshaping, train and sig', train.shape, sig.shape)
#  size=4
  size=7 # length of different variables in variable_array
  for i in range(size): 
    
    train_v = train[:,i::size].flatten()
    #test_v = test[:,i::size].flatten()
    sig_v = sig[:,i::size].flatten()
    print(f'{variable_array[i]}, after reshaping and flattening, train and sig', train_v.shape, sig_v.shape)
    bins=np.histogram(np.hstack((train_v,sig_v)),bins=60)[1]
    if(bins[-1] > 3000): bins = np.arange(0,3000,50)
    #plt.subplot(4,1,i+1)
    if bool_one:    plt.subplot(4,2,i+1)
    #plt.subplot(2,2,(i%4)+1)
#    fig,ax= plt.subplots(2,2)
#    plt.subplot(2,2,1)
#    row, col=((i%4)//2),((i%4)%2)
    plt.hist(train_v, alpha=0.5, label=f"bkg ({len(train_v)})", bins=bins, density=False)
#    ax[row,col].hist(train_v, alpha=0.5, label=f"bkg ({len(train_v)})", bins=bins, density=False)
    #plt.hist(test_v, alpha=0.5, label="test", bins=bins, density=True, color='lightskyblue')
#    ax[row,col].hist(sig_v, alpha=0.5, label=f"sig ({len(sig_v)})", bins=bins, density=False)
    plt.hist(sig_v, alpha=0.5, label=f"sig ({len(sig_v)})", bins=bins, density=False)
    #ax[row,col].set_title(f'{variable_array[i]} {tag_title}')
    plt.title(f'{variable_array[i]} {tag_title}')
    if i == 0: plt.legend(loc='upper right')
    #if (i %1)== 1: ax[row,col].legend()
#    print('all',i, i%4, i//4, row, col) 
    #print('all',i, i%4, i//4, row, col) 
#    if (i%4==3):
#      print('selected',i, i%4, i//4)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.yscale('log')
    if bool_one:
      if i==size-1:
        filepath=plot_dir+'inputs_'+tag_file+str(i//4)+'.png'
        plt.savefig(filepath)
        print("Saved inputs plot (", filepath)
        plt.close()
    else: 
      filepath=plot_dir+'inputs_'+variable_array[i]+'_'+tag_file+'.png' 
      plt.savefig(filepath)
      print("Saved inputs plot (", filepath)
      plt.close()
  #print("Saved inputs plot (", plot_dir+'inputs_'+tag_file+str(i//4)+'.png')
    
#  plt.savefig(plot_dir+'inputs_'+tag_file+'.png')


