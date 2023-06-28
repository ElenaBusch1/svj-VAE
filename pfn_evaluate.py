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
pfn_model = 'PFNv2'
arch_dir = "architectures_saved/"

## ---------- Load graph model ----------
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## Load classifier model
classifier = keras.models.load_model(arch_dir+pfn_model+'_classifier_arch')
classifier.load_weights(arch_dir+pfn_model+'_classifier_weights.h5')
classifier.compile()

## Load history
# with open(arch_dir+ae_model+"_history.json", 'r') as f:
#     h = json.load(f)
# print(h)
# print(type(h))

print ("Loaded model")

## Load testing data
x_events = -1 ## -1 for all events
#dsids = [515487, 515488, 515489, 515490, 515491, 515492, 515493, 515494, 515504, 515507, 515508, 515509, 515510, 515511, 515514, 515515, 515516, 515518, 515520, 515521, 515522, 515523, 515525, 515526]
dsids = range(515486,515527)
#dsids = ["QCDskim"]
my_variables = ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]

## evaluate bkg
bkg2,mT_bkg = getTwoJetSystem(x_events,"../v8.1/skim0.user.ebusch.QCDskim.root", my_variables, False, True)
scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg2,scaler,False) 
phi_bkg = graph.predict(bkg2)
pred_phi_bkg = classifier.predict(phi_bkg)
# ## Classifier loss
bkg_loss = pred_phi_bkg[:,1]
my_variables.insert(0,"score")
print(my_variables)
save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)
#print(save_bkg)
ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
with h5py.File("v8p1_PFNv2_QCDskim0_2.hdf5","w") as h5f:
  dset = h5f.create_dataset("data",data=rec_bkg)
print("Saved hdf5 for QCDskim")

quit()

## evaluate signals
for dsid in dsids:
  my_variables = ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]
  try:
    bkg2,mT_bkg = getTwoJetSystem(x_events,"../v8.1/user.ebusch."+str(dsid)+".root", my_variables, False)
  except:
    continue
  scaler = load(arch_dir+pfn_model+'_scaler.bin')
  bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
  
  phi_bkg = graph.predict(bkg2)
   
  pred_phi_bkg = classifier.predict(phi_bkg)
  
  # ## Classifier loss
  bkg_loss = pred_phi_bkg[:,1]
  
  my_variables.insert(0,"score")
  print(my_variables)
  save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)
  #print(save_bkg)
  ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
  rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
  
  with h5py.File("v8p1_PFNv2_"+str(dsid)+".hdf5","w") as h5f:
    dset = h5f.create_dataset("data",data=rec_bkg)
  print("Saved hdf5 for ", dsid)

quit()


##  #--- Eval plots 
##  # 1. Loss vs. epoch 
##  plot_saved_loss(h, ae_model, "loss")
##  if model.find('VAE') > -1:
##      plot_saved_loss(h, ae_model, "kl_loss")
##      plot_saved_loss(h, ae_model, "reco_loss")
# 2. Anomaly score
#plot_score(bkg_loss, sig_loss, False, False, ae_model)

#print(mT)
#print(bkg_loss > -11)
#mT_in = mT[bkg_loss > -11]
#print(mT_in)
#plot_score(mT, mT_in, False, False, "mTSel")
#quit()

#transform_loss_ex(bkg_loss, sig_loss, True)
##  #plot_score(bkg_loss, sig_loss, False, True, ae_model+"_xlog")
##  if ae_model.find('VAE') > -1:
##      plot_score(bkg_kl_loss, sig_kl_loss, remove_outliers=False, xlog=True, extra_tag=model+"_KLD")
##      plot_score(bkg_reco_loss, sig_reco_loss, False, False, model_name+'_Reco')
##  # 3. Signal Sensitivity Score
##  score = getSignalSensitivityScore(bkg_loss, sig_loss)
##  print("95 percentile score = ",score)
