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

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
pfn_model = 'PFNv1'
ae_model = 'ANTELOPEv1'
arch_dir = "architectures_saved/"

## ---------- Load graph model ----------
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## ---------- Load AE model ----------
encoder = keras.models.load_model(arch_dir+ae_model+'_encoder_arch')
decoder = keras.models.load_model(arch_dir+ae_model+'_decoder_arch')
ae = AE(encoder,decoder)

ae.get_layer('encoder').load_weights(arch_dir+ae_model+'_encoder_weights.h5')
ae.get_layer('decoder').load_weights(arch_dir+ae_model+'_decoder_weights.h5')

ae.compile(optimizer=keras.optimizers.Adam())

## Load history
# with open(arch_dir+ae_model+"_history.json", 'r') as f:
#     h = json.load(f)
# print(h)
# print(type(h))

print ("Loaded model")

## Load testing data
x_events = 250000
y_events = 250000
bkg2, sig2, mT_bkg, mT_sig = getTwoJetSystem(x_events,y_events, ["mT_jj"])
scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
sig2,_ = apply_StandardScaling(sig2,scaler,False)
#plot_vectors(bkg2,sig2,"ANTELOPE")

phi_bkg = graph.predict(bkg2)
phi_sig = graph.predict(sig2)

## Scale phis - values from v1 training
eval_min = 0.0
eval_max = 167.20311
phi_bkg = (phi_bkg - eval_min)/(eval_max-eval_min)
phi_sig = (phi_sig - eval_min)/(eval_max-eval_min)

pred_phi_bkg = ae.predict(phi_bkg)['reconstruction']
pred_phi_sig = ae.predict(phi_sig)['reconstruction']

# ## AE loss
bkg_loss = keras.losses.mse(phi_bkg, pred_phi_bkg)
sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)

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
plot_score(bkg_loss, sig_loss, False, True, ae_model)
bkg20 = np.percentile(bkg_loss, 80)
bkg10 = np.percentile(bkg_loss, 90)
bkg05 = np.percentile(bkg_loss, 95)
bkg01 = np.percentile(bkg_loss, 99)

print("Cuts: ", bkg20, bkg10, bkg05,bkg01)

mT_jj20 = mT_bkg[bkg_loss > bkg20]
mT_jj10 = mT_bkg[bkg_loss > bkg10]
mT_jj05 = mT_bkg[bkg_loss > bkg05]
mT_jj01 = mT_bkg[bkg_loss > bkg01]
print(len(mT_jj20), len(mT_jj10), len(mT_jj05), len(mT_jj01))
plot_single_variable([mT_bkg, mT_jj20,mT_jj10,mT_jj05,mT_jj01], ["100% QCD", "20% QCD", "10% QCD", "5% QCD", "1% QCD"], "mT Shape Check", logy=True) 

if (len(bkg_loss) > len(sig_loss)):
   bkg_loss = bkg_loss[:len(sig_loss)]
else:
   sig_loss = sig_loss[:len(bkg_loss)]
do_roc(bkg_loss, sig_loss, ae_model, False)

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

