import numpy as np
import json
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_helper import *
from models import *
from models_archive import *
from eval_helper import *

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
ae_model = "trackPFN_AE"
pfn_model = 'trackPFN'
arch_dir = "architectures_saved/"

nevents = 10000

## ---------- CODE ----------


# Load graph model
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## Load classifier model
classifier = keras.models.load_model(arch_dir+pfn_model+'_classifier_arch')
classifier.load_weights(arch_dir+pfn_model+'_classifier_weights.h5')
classifier.compile()

# ## Load ae model
# encoder = keras.models.load_model(arch_dir+ae_model+'_encoder_arch')
# decoder = keras.models.load_model(arch_dir+ae_model+'_decoder_arch')
# ae = AE(encoder,decoder)

# ae.get_layer('encoder').load_weights(arch_dir+ae_model+'_encoder_weights.h5')
# ae.get_layer('decoder').load_weights(arch_dir+ae_model+'_decoder_weights.h5')
# 
# ae.compile(optimizer=keras.optimizers.Adam())
#ae.summary()

## Load history
# with open(arch_dir+ae_model+"_history.json", 'r') as f:
#     h = json.load(f)
# print(h)
# print(type(h))

print ("Loaded model")

## Load testing data
x_raw = read_vectors(data_dir + "v8SmallPartialQCDmc20e.root", nevents)
sig_raw = read_vectors(data_dir + "v8SmallSIGmc20e.root", nevents)

## apply per-event scaling
bkg = apply_EventScaling(x_raw)
sig = apply_EventScaling(sig_raw)

## Make graph representation
phi_bkg = graph.predict(bkg)
phi_sig = graph.predict(sig)
pred_phi_bkg = classifier.predict(phi_bkg)
pred_phi_sig = classifier.predict(phi_sig)

bkg_loss = pred_phi_bkg[:,1]
sig_loss = pred_phi_sig[:,1]

#bkg_loss = keras.losses.mse(phi_bkg, pred_phi_bkg)
#sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)

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
plot_score(bkg_loss, sig_loss, False, False, pfn_model)
##  #plot_score(bkg_loss, sig_loss, False, True, ae_model+"_xlog")
##  if ae_model.find('VAE') > -1:
##      plot_score(bkg_kl_loss, sig_kl_loss, remove_outliers=False, xlog=True, extra_tag=model+"_KLD")
##      plot_score(bkg_reco_loss, sig_reco_loss, False, False, model_name+'_Reco')
##  # 3. Signal Sensitivity Score
##  score = getSignalSensitivityScore(bkg_loss, sig_loss)
##  print("95 percentile score = ",score)
# 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, ae_model, False)
if ae_model.find('VAE') > -1:
    do_roc(bkg_reco_loss, sig_reco_loss, ae_model+'_Reco', True)
    do_roc(bkg_kl_loss, sig_kl_loss, ae_model+'_KLD', True)
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

