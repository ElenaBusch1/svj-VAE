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

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"

# ## AE loss

with h5py.File("../v8.1/v8p1bkg.hdf5","r") as f:
  bkg_data = f.get('qcd')[:]

variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
bkg_loss = bkg_data["score"]
bkg20 = np.percentile(bkg_loss,99.5)

with h5py.File("../v8.1/v8p1_515495.hdf5","r") as f:
  sig1_data = f.get('data')[:]
with h5py.File("../v8.1/v8p1_515498.hdf5","r") as f:
  sig2_data = f.get('data')[:]
with h5py.File("../v8.1/v8p1_515515.hdf5","r") as f:
  sig3_data = f.get('data')[:]
with h5py.File("../v8.1/v8p1_515518.hdf5","r") as f:
  sig4_data = f.get('data')[:]

sig1_loss = sig1_data["score"]
sig2_loss = sig2_data["score"]
sig3_loss = sig3_data["score"]
sig4_loss = sig4_data["score"]

print(variables)

w0 = 100*bkg_data["weight"][bkg_loss>bkg20] 
w1 = sig1_data["weight"][sig1_loss>bkg20] 
w2 = sig2_data["weight"][sig2_loss>bkg20] 
w3 = sig3_data["weight"][sig3_loss>bkg20] 
w4 = sig4_data["weight"][sig4_loss>bkg20] 
w = [w0,w1,w2,w3,w4]

labels = ["QCD", "1500 GeV,0.2", "1500 GeV,0.8", "4000 GeV,0.2", "4000 GeV,0.8"]
for var in ["mT_jj"]:
  if (var=="weight" or var=="mcEventWeight"): continue
  bkg = bkg_data[var][bkg_loss>bkg20]
  sig1 = sig1_data[var][sig1_loss>bkg20] 
  sig2 = sig2_data[var][sig2_loss>bkg20] 
  sig3 = sig3_data[var][sig3_loss>bkg20] 
  sig4 = sig4_data[var][sig4_loss>bkg20] 
  labels[0] += "({0:.0%})".format(len(bkg)/len(bkg_data[var]))
  labels[1] += "({0:.0%})".format(len(sig1)/len(sig1_data[var]))
  labels[2] += "({0:.0%})".format(len(sig2)/len(sig2_data[var]))
  labels[3] += "({0:.0%})".format(len(sig3)/len(sig3_data[var]))
  labels[4] += "({0:.0%})".format(len(sig4)/len(sig4_data[var]))
  d = [bkg, sig1, sig2, sig3, sig4]
  plot_single_variable(d,w,labels, var, logy=True) 
  plot_ratio(d,w,labels, var, logy=True) 

quit()
w = [weight00, weight20, weight10, weight05, weight01]
mT = [mT_jj00, mT_jj20,mT_jj10,mT_jj05,mT_jj01]
names =  ["100% QCD", "20% QCD", "10% QCD", "5% QCD", "1% QCD"]
#plot_single_variable(mT,w,names, "mT Shape Check- Signal(515518)", logy=True) 
plot_ratio(mT,w,names, "mT Shape Check - Signal(515518)", logy=True) 



##  #--- Eval plots 
##  # 1. Loss vs. epoch 
##  plot_saved_loss(h, ae_model, "loss")
##  if model.find('VAE') > -1:
##      plot_saved_loss(h, ae_model, "kl_loss")
##      plot_saved_loss(h, ae_model, "reco_loss")
# 2. Anomaly score
#plot_score(bkg_loss, sig_loss, False, False, ae_model)
#bkg_loss = np.log(bkg_loss)
#sig_loss = np.log(sig_loss)
#plot_score(bkg_loss, sig_loss, False, True, ae_model)

#do_roc(bkg_loss, sig_loss, ae_model, False)

#transform_loss_ex(bkg_loss, sig_loss, True)
##  #plot_score(bkg_loss, sig_loss, False, True, ae_model+"_xlog")
##  if ae_model.find('VAE') > -1:
##      plot_score(bkg_kl_loss, sig_kl_loss, remove_outliers=False, xlog=True, extra_tag=model+"_KLD")
##      plot_score(bkg_reco_loss, sig_reco_loss, False, False, model_name+'_Reco')
##  # 3. Signal Sensitivity Score
##  score = getSignalSensitivityScore(bkg_loss, sig_loss)
##  print("95 percentile score = ",score)
# 4. ROCs/AUCs using sklearn functions imported above  

#bkg_loss, sig_loss = vrnn_transform(bkg_loss, sig_loss, True)

## 5. Plot Phi's
## plot_phi(phi_bkg, 'QCD', pfn_model)

##  # --- analysis variable checks
##  ## Load analysis variables
##  variables = ['mT_jj', 'met_met', 'weight']
##  x_dict = read_test_variables("../v6.4/v6p4smallQCD2.root", nevents, variables)
##  sig_dict = read_test_variables("../v6.4/user.ebusch.515500.root", nevents, variables)
##  
##  #apply cut & plot
##  x_cut = {}
##  x_cut2 = {}
##  sig_cut = {}
##  cut = np.percentile(bkg_loss,50)
##  cut2 = np.percentile(bkg_loss,98)
##  for key in variables:
##      if (len(x_dict[key]) != len(bkg_loss) or len(sig_dict[key]) != len(sig_loss)): print("ERROR: evaluated loss and test variables must have same length")
##      x_cut[key] = applyScoreCut(bkg_loss, x_dict[key], cut)
##      x_cut2[key] = applyScoreCut(bkg_loss, x_dict[key], cut2)
##      sig_cut[key] = applyScoreCut(sig_loss, sig_dict[key], cut)
##  
##  for key in variables:
##      if key == 'weight': continue
##      plot_var(x_dict, x_cut, x_cut2, key) 

def grid_scan():
  with h5py.File("../v8.1/v8p1bkg.hdf5","r") as f:
    bkg_data = f.get('qcd')[:]
  
  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg_loss = bkg_data["score"]
  print("bkg events", len(bkg_loss))
  bkg20 = np.percentile(bkg_loss, 80)
  
  sic_values = {}
  
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    print()
    try:
      with h5py.File("../v8.1/v8p1_"+str(dsid)+".hdf5","r") as f:
        sig1_data = f.get('data')[:]
      sig1_loss = sig1_data["score"]
      bkg1_loss = bkg_loss[:len(sig1_loss)]
      sic_vals = do_roc(bkg1_loss, sig1_loss, str(dsid), False)
      sic_values[dsid] = sic_vals
    except Exception as e:
      print(e)
    #sig1_cut = sig1_loss[sig1_loss>bkg20]
    #cut = len(sig1_cut)/total
    #print(dsid, f'{cut:.0%}') 
  
  print("bkg events: ", len(bkg_loss))
  
  do_grid_plots(sic_values)

