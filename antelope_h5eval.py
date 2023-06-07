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

bkg_loss = bkg_data["score"]

with h5py.File("../v8.1/v8p1_515518.hdf5","r") as f:
  sig1_data = f.get('data')[:]

mT_sig = sig1_data["mT_jj"]
weights = sig1_data["weight"]
sig_loss = sig1_data["score"]

##  #--- Grid test
##  scores = np.zeros((10,4))
##  aucs = np.zeros((10,4))
##  j = -1
##  for i in range(487,527):
##    k = i%4-3
##    if k == 0: j+=1
##    if i in [488,511,514,517,520,522]:continue
##    sig_raw = read_vectors("../v6.4/user.ebusch.515"+str(i)+".root", nevents)
##    sig = apply_EventScaling(sig_raw)
##    phi_sig = graph.predict(sig)
##    pred_phi_sig = ae.predict(phi_sig)['reconstruction']
##    sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)
##  
##    score = getSignalSensitivityScore(bkg_loss, sig_loss)
##    #print("95 percentile score = ",score)
##    auc = do_roc(bkg_loss, sig_loss, ae_model, False)
##    print(auc,score)
##    scores[j,k] = score
##    aucs[j,k] = auc
##  
##  print(scores)
##  print(aucs)

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
bkg100 = np.percentile(bkg_loss, 0)
bkg20 = np.percentile(bkg_loss, 80)
bkg10 = np.percentile(bkg_loss, 90)
bkg05 = np.percentile(bkg_loss, 95)
bkg01 = np.percentile(bkg_loss, 99)

print("Cuts: ", bkg20, bkg10, bkg05,bkg01)
#mT_jj0 = mT_bkg[bkg_loss > bkg100]
#mT_jj20 = mT_bkg[bkg_loss > bkg20]
#mT_jj10 = mT_bkg[bkg_loss > bkg10]
#mT_jj05 = mT_bkg[bkg_loss > bkg05]
#mT_jj01 = mT_bkg[bkg_loss > bkg01]
#weight0 = weights[bkg_loss > bkg100]
#weight20 = weights[bkg_loss > bkg20]
#weight10 = weights[bkg_loss > bkg10]
#weight05 = weights[bkg_loss > bkg05]
#weight01 = weights[bkg_loss > bkg01]
mT_jj00 = mT_sig[sig_loss > bkg100]
mT_jj20 = mT_sig[sig_loss > bkg20]
mT_jj10 = mT_sig[sig_loss > bkg10]
mT_jj05 = mT_sig[sig_loss > bkg05]
mT_jj01 = mT_sig[sig_loss > bkg01]
weight00 = weights[sig_loss > bkg100]
weight20 = weights[sig_loss > bkg20]
weight10 = weights[sig_loss > bkg10]
weight05 = weights[sig_loss > bkg05]
weight01 = weights[sig_loss > bkg01]

w = [weight00, weight20, weight10, weight05, weight01]
mT = [mT_jj00, mT_jj20,mT_jj10,mT_jj05,mT_jj01]
names =  ["100% QCD", "20% QCD", "10% QCD", "5% QCD", "1% QCD"]
#plot_single_variable(mT,w,names, "mT Shape Check- Signal(515518)", logy=True) 
plot_ratio(mT,w,names, "mT Shape Check - Signal(515518)", logy=True) 

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

