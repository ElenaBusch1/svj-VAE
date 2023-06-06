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
pfn_model = 'PFN'
#pfn_model = 'PFNv1'
#arch_dir = "/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/"
arch_dir="architectures_saved_old/architectures_saved_jun5/"
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
x_events = 5000
y_events = 5000
bool_weight=True
if bool_weight:weight_tag='ws'
else:weight_tag='nws'
tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
bkg2, sig2, mT_bkg, mT_sig = getTwoJetSystem(x_events,y_events,tag_file=tag, tag_title=tag, bool_weight=bool_weight,  extraVars=["mT_jj"])
#bkg2, sig2, mT_bkg, mT_sig = getTwoJetSystem(x_events,y_events, ["mT_jj"])
scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
sig2,_ = apply_StandardScaling(sig2,scaler,False)
#plot_vectors(bkg2,sig2,"PFN")

phi_bkg = graph.predict(bkg2)
phi_sig = graph.predict(sig2)

# each event has a pfn score 
pred_phi_bkg = classifier.predict(phi_bkg)
pred_phi_sig = classifier.predict(phi_sig)

# write on html
"""
print('*'*30)
print(phi_bkg)
print('-'*30)
print(pred_phi_bkg)
#print(np.min(pred_phi_sig[:,1]))
#arr=pred_phi_bkg
for arr, add in zip([pred_phi_bkg, pred_phi_sig], ['bkg', 'sig']):
  print(arr, add)
  h5dir='h5dir'
  filename=f'{tag}_{add}'
  h5path=h5dir+filename+'.h5'
  if not os.path.exists(h5path):
    print(h5dir+filename+'.h5')
    with h5py.File(h5path, 'w') as f:
      data = f.create_dataset("default", data = arr)
  else:
    with h5py.File(h5path, 'r') as f:
      data = f["default"]

print(np.min(data[:,1]))
"""
## Classifier loss
bkg_loss = pred_phi_bkg[:,1]
sig_loss = pred_phi_sig[:,1]

if (len(bkg_loss) > len(sig_loss)):
   bkg_loss = bkg_loss[:len(sig_loss)]
else:
   sig_loss = sig_loss[:len(bkg_loss)]
#do_roc(bkg_loss, sig_loss, pfn_model, True)
make_transformed_plot=False
auc=do_roc(bkg_loss, sig_loss, tag_file=tag, tag_title=tag, make_transformed_plot=make_transformed_plot)

#cut on each event depending on a pfn score
#find which score gives us signal to percentile of background

percentile=50 #10 # 95
score = getSignalSensitivityScore(bkg_loss, sig_loss, percentile=percentile)
print(f'{percentile}% -score {score}')

cuts=[0, .3, .6,.9, score] 
bkg_ls=[]
bkg_loss_arr=np.array(bkg_loss)
print(mT_bkg.shape, mT_bkg)
for i, cut in enumerate(cuts):
  bkg_cut_idx=np.argwhere(bkg_loss_arr>=cut)
  bkg_cut=mT_bkg[bkg_cut_idx]  
#  bkg_cut=bkg_loss_arr[bkg_cut_idx]
  bkg_cut=bkg_cut.flatten()
  print(i, cut, len(bkg_cut),bkg_cut)
  bkg_ls.append(bkg_cut)
#plot mT distribution
#plot_single_variable([bkg_loss_arr,bkg_cut], cuts, "mT distribution", logy=True) 
plot_single_variable(bkg_ls, cuts, "mT distribution", logy=True)
 
#plot_single_variable([bkg_loss,sig_loss], ["Background", "Signal"], "mT distribution", logy=True) 
print('done')
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
