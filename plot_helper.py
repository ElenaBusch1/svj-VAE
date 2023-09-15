#!/usr/bin/env python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
from termcolor import cprint
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
# tag = "pfnEvalTest"      
#tag = "PFN_2jAvg_MM"
#plot_dir = '/a/home/kolya/ebusch/WWW/SVJ/autoencoder/'
#plot_dir = '/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun5_500000_nep100_nl100/'
#plot_dir = '/nevis/katya01/data/users/kpark/svj-vae/plots_result/sig_elena/jun12_sig/'
#plot_dir = '/nevis/katya01/data/users/kpark/svj-vae/plots_result/bkg_elena/jun12_bkg/'
plot_dir = '/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun29/lala/'
params = {'legend.fontsize': 'x-large',
           'figure.figsize': (10, 8),
         'axes.labelsize': 'large',
          'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',

         }
"""

         'axes.labelsize': 'x-large',
          'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',

"""
plt.rcParams.update(params)
"""
bkg_sigma=.7123
bkg_mu=.234
plt.plot([],[], label=f"$\sigma$ = {round(bkg_sigma,3)}, $\mu$ = {round(bkg_mu,3)}")
plt.legend()
plt.show()
sys.exit()
"""
def plot_pca(latent,latent_label, nlabel,  n_components=2, tag_file="", tag_title="", plot_dir=""): # if the number of features is very high
  # principal components = linear combinations of initial variables -> variable number reduced to say 2 (n_components) for visualization
  # n_components is number of dimensions I want for the scatter plot -> usually 2 for x and y axes
  # nlabel is the number of classification label types
  # latent_label is a list of labels (truth values) so the length of this should be the same as the number of samples
  pca = PCA(n_components=n_components)
  x_transform = pca.fit_transform(latent)
  print(x_transform.shape)
  if n_components==2:
    plt.scatter(x_transform[:,0], x_transform[:,1], c=latent_label, cmap=plt.cm.get_cmap("jet", nlabel))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title(f'{tag_title} Latent Space Visualization (PCA)')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.legend(loc='upper right')
    plt.savefig(plot_dir+'/pca'+tag_file+'.png')
  #plt.show()
    plt.clf()

  else: print('cannot plot a scatter plot of PCA as n_components is not 2D')
  return x_transform  

def plot_tsne(latent, n_components=2, tag_file="", tag_title="", plot_dir=""):
  #https://github.com/npitsillos/mnist-vae/blob/master/main.py
  # draw just the z_means
  tsne = TSNE(n_components=2, init="pca", random_state=0)
  x_transform = tsne.fit_transform(latent)
  if n_components==2:
    data = np.vstack((x_transform.T, target)).T
    df = pd.DataFrame(data=data, columns=["z1", "z2", "label"])
    df["label"] = df["label"].astype(str)

    df.plot.scatter(x='z1', y= 'z2', c='label') # color determined by label column
    plt.title(f'{tag_title} Latent Space Visualization (TSNE)')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.legend(loc='upper right')
    plt.savefig(plot_dir+'/tsne'+tag_file+'.png')
  #plt.show()
    plt.clf()
 
  return x_transform 

def plot_ntrack(h_ls,  tag_file="", tag_title="", plot_dir="", bin_max=0):
  #label=['no cuts','ntrack >= 3','pt > 10 GeV in leading jet','pt > 10 GeV in subleading jet']
  print(len(h_ls))
  if len(h_ls)==5:
    label=['no cuts','ntrack >= 0','ntrack >= 1', 'ntrack >= 2', 'ntrack >= 3']
  elif len(h_ls)==4:
    label=['sig (before)', 'bkg (before)','sig (after)', 'bkg (after)']
  else:label=['signal', 'QCD']
  bin_max_ls=[]
  count_ls=[]
  nevent_ls=[]
  n=0
  for i,h in enumerate(h_ls):
  #print(h.history)
    first_var=h[:,:,0]
    if i!=0 and n!=first_var.shape[1]:
      cprint('not the same number of max_tracks -> incompatible','red')
      sys.exit()
    else: n=first_var.shape[1]
    nevent=first_var.shape[0]
    count=np.count_nonzero(first_var, axis = 1)
   
    bin_min=0
    bin_max_ls.append(np.max(count))
    nevent_ls.append(nevent) 
    count_ls.append(count) 

  if bin_max==0: # default value is 0
    bin_max=np.max(np.array(bin_max_ls))     # this ensures that we are not cutting out any events 
  
  bins=np.array(range(bin_min-1,bin_max+2))
  x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
  print(f'{bin_max_ls=},{bin_max=}')

  for i,h in enumerate(h_ls):
    plt.hist(count_ls[i], label=f'NE={int(nevent_ls[i])}, {label[i]}',align='right', bins=x_bins,histtype='step',log=True, alpha=0.7) # use the set bins
    #plt.hist(count, label=f'NE={int(nevent)}, {label[i]}',align='right', bins=x_bins,histtype='step',log=True) # use the set bins
 
  plt.title(f' Number of tracks ({tag_title})')
  plt.xlabel('ntrack')
  plt.ylabel('count')
  ax = plt.gca()
  ax.xaxis.set_major_locator(MultipleLocator(10))  
  ax.xaxis.set_minor_locator(MultipleLocator(1))  
  plt.legend(loc='upper right')
#  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'.png')
  plt.savefig(plot_dir+'/ntrack'+tag_file+'.png')
  #plt.show()
  plt.clf()
  print("Saved ntrack plot for ", tag_file)
 
def plot_hist(dict_ls, var='auc'):
    
  for key in dict_ls:
    dict_ls[key][var]

def my_metric(s,b):
    return np.sqrt(2*((s+b)*np.log(1+s/b)-s))

    
def detect_outliers(x):
  z = np.abs(stats.zscore(x))
  print(max(z))
  x_smooth = x[z<40]
  n_removed = len(x)-len(x_smooth)
  print(n_removed, " outliers removed")
  return x_smooth, n_removed

def plot_loss(h, loss='loss', tag_file="", tag_title="", plot_dir=""):
  #print(h.history)
  plt.plot(h.history[loss])
  plt.plot(h.history['val_'+loss])
  plt.title(loss +f' {tag_title}')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
#  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'.png')
  plt.tight_layout()
  plt.savefig(plot_dir+loss+'VsEpoch_'+tag_file+'.png')
  plt.clf()
  print("Saved loss plot for ", tag_file, loss)

def plot_saved_loss(h, loss='loss', tag_file="", tag_title="", plot_dir=""):
  plt.plot(h[loss])
  plt.plot(h['val_'+loss])
  plt.title(loss +f' {tag_title}')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
  plt.tight_layout()
  #plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'log.png')
  plt.savefig(plot_dir+loss+'VsEpoch_'+tag_file+'log.png')
  plt.clf()
  print("Saved loss plot for ", tag_file)

def plot_var(x_dict, x_cut1, x_cut2, key , tag_file="", tag_title="", plot_dir=""):
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

def make_roc(fpr,tpr,auc, tag_file="", tag_title="", plot_dir=""):
  plt.plot(fpr,tpr,label="AUC = %0.2f" % auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("SVJ "+" ROC" +f' {tag_title}')
  plt.legend()
  plt.tight_layout()
  plt.savefig(plot_dir+'roc_'+tag_file+'.png')
  plt.clf()
  print("Saved ROC curve for ", tag_file)

def make_sic(fpr,tpr,auc, bkg, tag_file="", tag_title="",  plot_dir=""):
  print('make_sic')
  y = tpr[1:]/np.sqrt(fpr[1:])
  good = (y != np.inf) & (tpr[1:] > 0.08)
  ymax = max(y[good])
  ymax_i = np.argmax(y[good])
  sigEff = tpr[1:][good][ymax_i]
  qcdEff = fpr[1:][good][ymax_i]
  score_cut = np.percentile(bkg,100-(qcdEff*100))
  print("Max improvement: ", ymax)
  print("Sig eff: ", sigEff)
  print("Bkg eff: ", qcdEff)
  print("Score selection: ", score_cut)
  plt.plot(tpr[1:],y,label="AUC = %0.2f" % auc)
  plt.axhline(y=1, color='0.8', linestyle='--')
  plt.xlabel("Signal Efficiency (TPR)")
  plt.ylabel("Signal Sensitivity ($TPR/\sqrt{FPR}$)")
  plt.title("Significance Improvement Characteristic " )
  plt.legend()
  plt.tight_layout()
  plt.savefig(plot_dir+'sic_'+tag_file+'.png')
  plt.clf()
  print("Saved SIC ")
  return {'sicMax':ymax, 'sigEff': sigEff, 'qcdEff': qcdEff, 'score_cut': score_cut}

def make_grid_plot(values,title,method,plot_dir,tag=''):
  #values must be 4 X 10

  fig,ax = plt.subplots(1,1)
  if (method.find("compare") != -1): img = ax.imshow(values, cmap='PiYG',norm=colors.LogNorm(vmin=0.1,vmax=10))
  else:
    if (title == "qcdEff"): img = ax.imshow(values,norm=colors.LogNorm(vmin=1e-7,vmax=1e-1))
    elif (title == "sigEff"): img = ax.imshow(values,vmin=-0.1,vmax=0.7)
    elif (title == "sensitivity_Inclusive" or title == "sensitivity_mT"): img = ax.imshow(values, norm=colors.LogNorm(vmin=1e-5,vmax=1.5))
    elif (title == "auc"): img = ax.imshow(values, vmin=0.7, vmax=1)
    elif (title == "sicMax"): img = ax.imshow(values, vmin=-2, vmax=20)
    else: img = ax.imshow(values)

  # add text to table
  for (j,i),label in np.ndenumerate(values):
    if label == 0.0: continue
    if title == "qcdEff" or title == "sensitivity_Inclusive" or title == "sensitivity_mT": ax.text(i,j,'{0:.1e}'.format(label),ha='center', va='center', fontsize = 'x-small')
    elif title == "score_cut": ax.text(i,j,'{0:.3f}'.format(label),ha='center', va='center', fontsize = 'x-small')
    else: ax.text(i,j,'{0:.2f}'.format(label),ha='center', va='center', fontsize = 'x-small')

  # x-y labels for grid 
  x_label_list = ['1.0', '1.25', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '5.0', '6.0']
  y_label_list = ['0.2', '0.4', '0.6', '0.8']
  ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
  ax.set_xticklabels(x_label_list)
  ax.set_xlabel('Z\' Mass [TeV]')
  ax.set_yticks([0,1,2,3])
  ax.set_yticklabels(y_label_list)
  ax.set_ylabel('$R_{inv}$')
  
  ax.set_title(method+"; "+title)
  plt.tight_layout()
  plt.savefig(plot_dir+'table_'+method+'_'+title+'_'+tag+'.png')
  print("Saved grid plot for", title)

def make_single_roc(rocs,aucs,ylabel, tag_file="", tag_title="",  plot_dir=""):
  plt.plot(rocs[0],rocs[1],label=str(np.round(r,4))+r", $\sigma$="+str(sigs)+": AUC="+str(np.round(aucs,3)))
  plt.xlabel('fpr')
  plt.ylabel(Ylabel)
  plt.title('ROC: '+saveTag+f' {tag_title}')
  plt.figtext(0.7,0.95,"size="+str(sizeeach)+", nvars="+str(nInputs))
  plt.legend()
  plt.tight_layout()
  plt.savefig(saveTag+'_roc_aucs_'+Ylabel.replace("/","")+tag_file+'.pdf')
  plt.clf()

def plot_score(bkg_score, sig_score, remove_outliers=True, xlog=True, tag_file="", tag_title="",  plot_dir="", bool_pfn=True):
  if remove_outliers:
    bkg_score,nb = detect_outliers(bkg_score)
    sig_score,ns = detect_outliers(sig_score)

  #bins=np.histogram(np.hstack((bkg_score,sig_score)),bins=80)[1]
  try:
    bmax = np.max(np.max(bkg_score),np.max(sig_score))
    bmin = np.min(np.min(bkg_score),np.min(sig_score))
  except:
    try:
      bmax = np.max(bkg_score)
      bmin = np.min(bkg_score)
    except:
      try: 
        bmax = np.max(sig_score)
        bmin = np.min(sig_score)
      except: 
        print(f'both sig_score and bkg_score are empty arrays so not creating the plot for {tag_file}')
        return
  
  cprint(f'{bmax}, {bmin}', 'magenta')
  if xlog and bmin == 0  : bmin = 1e-9
  if xlog  : bins = np.logspace(np.log10(bmin),np.log10(bmax),80)
  else:
    try: bins=np.histogram(np.hstack((bkg_score,sig_score)),bins=80)[1]
    except: 
      try: bins=np.histogram(bkg_score,bins=80)[1]
      except: bins=np.histogram(sig_score,bins=80)[1]
  try: 
    #plt.hist(bkg_score, bins=bins, alpha=0.5, label=f"normal ({len(bkg_score)})", density=True, color='blue')
    plt.hist(bkg_score, bins=bins, alpha=0.5, label=f"bkg ({len(bkg_score)})", density=True, color='blue')
  except: 
    #plt.hist([],[], label="normal ({len(bkg_score)})")
    plt.hist([],[], label="bkg ({len(bkg_score)})")
    print('bkg_score not plotted: check if it is an empty array')
  try: 
    #plt.hist(sig_score, bins=bins, alpha=0.5, label=f"anomalous ({len(sig_score)})", density=True, color='red')
    plt.hist(sig_score, bins=bins, alpha=0.5, label=f"sig ({len(sig_score)})", density=True, color='red')

  except: 
    #plt.hist([],[], label="anomalous ({len(sig_score)})")
    plt.hist([],[], label="sig ({len(sig_score)})")
    print('sig_score not plotted: check if it is an empty array')
  if xlog : plt.xscale('log')
  plt.yscale('log')
  plt.legend()
  if bool_pfn:
    plt.title(f'PFN Score {tag_title}')
    plt.xlabel('PFN Score')
  else:
    plt.title(f'Anomaly Score {tag_title}')
    plt.xlabel('Anomaly Score')
  #plt.xlabel('Loss')
  plt.tight_layout()
  plt.savefig(plot_dir+'score_'+tag_file+'.png')
  plt.clf()
  print("Saved score distribution for", tag_file)

def plot_phi(phis, tag_file="", tag_title="",  plot_dir=""):
#def plot_phi(phis,name,extra_tag, tag_file="", tag_title="",  plot_dir=""):
  nphis = phis.shape[1]
  nevents = phis.shape[0]
  idx = [i for i in range(nphis)]*nevents

  phiT = phis.T
  print("n zeros = ", len(np.where(~phiT.any(axis=1))[0]))
  phis = phis.flatten()
  nbinsx = 10
  bin_width = max(phis)/nbinsx
  print("max: ", max(phis))
  print("bin_width", bin_width)
  phis[phis==0] = -bin_width 

  fig, ax = plt.subplots()
  h = ax.hist2d(phis,idx,bins=[nbinsx+1,nphis],norm=colors.LogNorm())
  fig.colorbar(h[3], ax=ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Index')
  ax.set_title(f'PFN Set Representation - {tag_title}')
  plt.tight_layout()
  plt.savefig(plot_dir+f'phi2D_{tag_file}.png')
  plt.clf()
  print("Saved 2D plot of phi-rep for", tag_file)

def plot_inputs(bkg, sig, variable_array,tag_file="", tag_title="",  plot_dir=""):
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

def plot_single_variable(hists, h_names, weights_ls,tag_title,density_top=True, logy=False, len_ls=[],  plot_dir="", tag_file=''):
  nbins=100
  #nbins=50
  hists_flat=np.concatenate(hists)
  bin_min=np.min(hists_flat)
  bin_max=np.max(hists_flat)
  gap=(bin_max-bin_min)*0.05
  bins=np.linspace(bin_min-gap,bin_max+gap,nbins)
  x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
  hists=list(hists)

  cut0_idx=0
  len0=len(hists[cut0_idx])
 
  for data,name,weights,i in zip(hists,h_names,weights_ls, range(len(hists))):
    y,_, _=plt.hist(data, bins=bins, weights=weights,density=density_top,histtype='step', alpha=0.7, label=f'NE={len(data)}, {name}')
    #y,_, _=plt.hist(data, bins=bins, weights=weights,density=density_top,histtype='step', alpha=0.7, label=f'NE={len(data)}, {round(len(data)/len_ls[i]*100,1)}% left, cut={name}')
#    y_unnorm,_, _=plt[0].hist(data, bins=bins, density=False,histtype='step', alpha=0)
#    print(i, len(bins), len(y), bins, y) 
    #if i ==len(hists)-1:
  plt.tick_params(axis='y', which='minor') 
  plt.grid()
 
  plt.ylabel('Event Number')
  plt.xlabel(tag_title)
  if (logy): plt.yscale("log")
  plt.legend(loc='lower right')
  #plt.legend(loc='upper right')
  plt.title(tag_title)

  plt.tight_layout()
  plt.savefig(plot_dir+'hist_'+tag_file+'_weighted'+'.png')
  #plt.savefig(plot_dir+'hist_'+tag_title.replace(" ","").replace('(','')+'_weighted_cut'+'.png')
  plt.clf()
  print("Saved plot",tag_title)



def plot_single_variable_ratio(hists, h_names, weights_ls,title,density_top=True, logy=False, len_ls=[],  plot_dir=""):
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
  axs[1].set_ylim(0,3)  
  axs[1].set_ylabel('Ratio')
  axs[1].legend(loc='upper right')
  plt.tick_params(axis='y', which='minor') 
  plt.grid()

  axs[0].set_ylabel('Event Number')
  if (logy): axs[0].set_yscale("log")
  axs[0].legend(loc='upper right')
  axs[1].legend(loc='upper right')
  axs[0].set_title(title)

  plt.savefig(plot_dir+'histlin_'+title.replace(" ","")+'_'+tag+'.png')
  plt.clf()
  print("Saved plot",title)

def plot_ratio(hists, weights, h_names, title, logy=False):
  colors = ['black', 'darkblue', 'deepskyblue', 'firebrick', 'orange']

  f, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
  nbins=20
  hists_flat=np.concatenate(hists)
  #bin_min=np.min(hists_flat)
  #bin_max=np.max(hists_flat)
  #gap=(bin_max-bin_min)*0.05
  bins=np.linspace(1500,6500,50)
  x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
  hists=list(hists)
  nTot = len(hists[0])
  for data,weight,name,i in zip(hists,weights,h_names, range(len(hists))):
    y,_, _=axs[0].hist(data, bins=bins, label=f'{name}', density=False, histtype='step', weights=weight, color=colors[i])
    #print(i, len(bins), len(y), bins, y) 
    #if i ==len(hists)-1:
    if i ==0:
      y0=y # make sure the first of hists list has the most number of events
      continue
    axs[1].scatter(x_bins,my_metric(y,y0), marker="+", color=colors[i], label=f'{max(my_metric(y,y0)):.1E}')
    #axs[1].scatter(x_bins,zero_div(y,y0))

  #axs[1].set_ylim(0.5,3.0)  
  axs[1].set_ylabel('Fig of Merit')
  axs[1].set_yscale('log')
  #axs[1].legend(loc='upper right', fontsize='x-small')
  plt.tick_params(axis='y', which='minor') 
  plt.grid()

  axs[0].set_ylabel('Event Number') 
  if (logy): axs[0].set_yscale("log")
  axs[0].legend(loc='upper right', fontsize='x-small')
  axs[0].set_title(title)
  axs[1].legend(loc='upper right')

  lt.savefig(plot_dir+'ratio_'+title.replace(" ","").replace('(','')+'_weighted'+'.png')
  plt.clf()
  print("Saved plot",title)

def get_nTracks(x):
  n_tracks = []
  for i in range(x.shape[0]):
    tracks = x[i,:,:].any(axis=1)
    tracks = tracks[tracks == True]
    n_tracks.append(len(tracks))
  return n_tracks
 

def plot_nTracks(bkg, sig, tag_file="", tag_title="",  plot_dir=""):
  bkg_tracks = get_nTracks(bkg)
  sig_tracks = get_nTracks(sig)
  #bins=np.histogram(np.hstack((bkg_tracks,sig_tracks)),bins=60)[1]
  bins = np.arange(0,100,2)
  plt.hist(bkg_tracks,alpha=0.5, label="bkg", bins=bins, density=False)
  plt.hist(sig_tracks,alpha=0.5, label="sig", bins=bins, density=False)
  plt.title("nTracks (after pT>10) - Tertiary"+f' {tag_title}')
  plt.legend()
  plt.savefig(plot_dir+'nTracks_'+'_'+tag_file+'.png')
  plt.clf()
  print("Saved plot of nTracks")

def plot_vectors_jet(train,sig,jet_array, tag_file="", tag_title="", bool_jet=False,  plot_dir=""):
#  variable_array=["pt"]
  print('before reshaping, train and sig', train.shape, sig.shape)
#  size=2 # different than size in plot_vectors
  size=len(jet_array) # different than size in plot_vectors
  n=2
  for i in range(size):
    #train_v= train[:,i,0]
    train_v=train[:, i]
    #train_v= train[:,i%n,i//n]
    sig_v= sig[:,i]
    #sig_v= sig[:,i%n,i//n]
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


def plot_nTracks_2d_hist(leadingJetTracks, subleadingJetTracks):
  leading_nTracks = get_nTracks(leadingJetTracks)
  subleading_nTracks = get_nTracks(subleadingJetTracks)
  bins = np.arange(0, 75, 1)
  plt.hist2d(leading_nTracks, subleading_nTracks, bins=bins)
  plt.title("nTracks (after pT > 10)")
  plt.xlabel("Leading Jet Number of Tracks")
  plt.ylabel("Subleading Jet Number of Tracks")
  plt.tight_layout()
  plt.savefig(plot_dir+'nTracks_2d_'+tag+'.png')
  plt.clf()
  print("Saved plot of nTracks 2D")

def plot_vectors(train,sig, tag_file="", tag_title="", bool_one=True,  plot_dir=""):
  #variable_array = ["pT", "eta", "phi", "E"]
  variable_array = ["pT", "eta", "phi", "E", "z0", "d0", "qOverP"]
  print('before reshaping, train and sig', train.shape, sig.shape)

  if (len(train.shape) == 3):
    train = train.reshape(train.shape[0], train.shape[1] * train.shape[2])
  if (len(sig.shape) == 3):
    sig = sig.reshape(sig.shape[0], sig.shape[1] * sig.shape[2])
#  size=4
  size=7 # length of different variables in variable_array
  for i in range(size): 
    
    train_v = train[:,i::size].flatten()
    #test_v = test[:,i::size].flatten()
    sig_v = sig[:,i::size].flatten()
#    print(f'{variable_array[i]}, after reshaping and flattening, train and sig', train_v.shape, sig_v.shape)
    bins=np.histogram(np.hstack((train_v,sig_v)),bins=60)[1]
    if(bins[-1] > 3000): bins = np.arange(0,3000,50)
    #plt.subplot(4,1,i+1)
    if bool_one:    plt.subplot(4,2,i+1)
    plt.hist(train_v, alpha=0.5, label=f"bkg ({len(train_v)})", bins=bins, density=False)
    plt.hist(sig_v, alpha=0.5, label=f"sig ({len(sig_v)})", bins=bins, density=False)
    plt.title(f'{variable_array[i]} {tag_title}')
    if i == 0: plt.legend(loc='upper right')
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


def plot_1D_phi(bkg, sig, labels, plot_dir, tag_file, tag_title, bool_norm=False, ylog=True, bins=[]):
  
  per_plot=4 # 4 plots per figure
  length= int(bkg.shape[1]/per_plot)# 12
  for j in range(length):
    for i in range(per_plot):
      bkg_phi = bkg[:,i+j*4].flatten()
      try: sig_phi = sig[:,i+j*4].flatten()
      except:sig_phi=np.array([])
      print(f'plot_1D_phi, {bkg_phi.shape=}')
      if bins ==[]: bins=np.histogram(np.hstack((bkg_phi,sig_phi)),bins=50)[1]
      else:bins=bins
      #print(bins)
      plt.subplot(2,2,i+1)
#      plt.tight_layout(h_pad=1, w_pad=1)
      plt.hist(bkg_phi, alpha=0.7, label=labels[0]+f' ({len(bkg_phi)})', bins=bins, density=False, color = 'darkblue', histtype='step')
      if bool_norm:
        (bkg_mu, bkg_sigma) = norm.fit(bkg_phi)
        bkg_mu_rd, bkg_sigma_rd=str(round(bkg_mu,3)), str(round(bkg_sigma,3))
#        print(r"$\mu$ = "+ str(round(bkg_mu,3)))
        y = norm.pdf( bins, bkg_mu, bkg_sigma)
        plt.plot(bins, y, 'b+', linewidth=1, label=r"$\mu$ = "+ bkg_mu_rd +", $\sigma$ = " + bkg_sigma_rd, alpha=0.5)
      plt.hist(sig_phi, alpha=0.7, label=labels[1]+f' ({len(sig_phi)})', bins=bins, density=False, color = 'darkred',histtype='step')
      if bool_norm:
        (sig_mu, sig_sigma) = norm.fit(sig_phi)
        sig_mu_rd, sig_sigma_rd=str(round(sig_mu,3)), str(round(sig_sigma,3))
        y = norm.pdf( bins, sig_mu, sig_sigma)
        #plt.plot(bins, y, 'r--', linewidth=2, label=f"$\mu_rd$ = {sig_mu_rd}, $\sigma_rd$ = {sig_sigma_rd}", alpha=0.5)
        plt.plot(bins, y, 'r--', linewidth=1, label=r"$\mu$ = "+ sig_mu_rd +", $\sigma$ = " + sig_sigma_rd, alpha=0.5)
      if ylog:
        plt.yscale('log')
      plt.title(f'{tag_title} Latent Space - '+str(i+j*4))
      if i == 1: plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(plot_dir+'PFNlatent_'+str(j)+'_'+tag_file+'.png')
    plt.clf()
    print("Saved PFN latent space plot (", j, ")")

