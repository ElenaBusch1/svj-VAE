import numpy as np
import json
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from plot_helper import *
from models import *
from models_archive import *
from eval_helper import *
import h5py

tag = "PFNv6_phi"
plot_dir = '/a/home/kolya/ebusch/WWW/SVJ/autoencoder/'

def plot_1D_phi(qcd,met,sig):
  for j in range(16):
    for i in range(4):
      qcd_phi = qcd[:,i+j*4].flatten()
      met_phi = met[:,i+j*4].flatten()
      sig_phi = sig[:,i+j*4].flatten()
      bins=np.histogram(np.hstack((qcd_phi,met_phi,sig_phi)),bins=50)[1]
      plt.subplot(2,2,i+1)
      plt.tight_layout(h_pad=1, w_pad=1)
      plt.hist(qcd_phi, alpha=0.5, label="QCD", bins=bins, density=True, color = 'darkblue', histtype='step')
      plt.hist(met_phi, alpha=0.5, label="MET", bins=bins, density=True, color='lightskyblue', histtype='step')
      plt.hist(sig_phi, alpha=0.5, label="sig", bins=bins, density=True, color = 'orange',histtype='step')
      #plt.yscale('log')
      plt.title('PFN Latent Space - '+str(i+j*4))
      if i == 1: plt.legend()
    plt.savefig(plot_dir+'PFNlatent_'+str(j)+'_'+tag+'.png')
    plt.clf()
    print("Saved PFN latent space plot (", j, ")")

def plot_corr_matrix(values):
  fig,ax = plt.subplots(1,1)
  img = ax.imshow(values, cmap='Wistia')
  

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
pfn_models = ['PFNv6']
arch_dir = "architectures_saved/"
pfn_model = 'PFNv6'
x_events = -1

## ---------- Load graph model ----------
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

qcd = getTwoJetSystem(50000,"../v9.1/skim0.user.ebusch.QCDskim.root", [], True)
met = getTwoJetSystem(50000,"../v9.1/skim0.user.ebusch.METbkg.root", [], False)
sig = getTwoJetSystem(50000,"../v8.1/skim3_0.user.ebusch.SIGall.root", [], False)
scaler = load(arch_dir+pfn_model+'_scaler.bin')
qcd2,_ = apply_StandardScaling(qcd,scaler,False) 
met2,_ = apply_StandardScaling(met,scaler,False) 
sig2,_ = apply_StandardScaling(sig,scaler,False) 
phi_qcd = graph.predict(qcd2)
phi_met = graph.predict(met2)
phi_sig = graph.predict(sig2)

plot_1D_phi(phi_qcd,phi_met,phi_sig)
