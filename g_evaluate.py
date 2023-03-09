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
ae_model = "znnPFN32_AE"
pfn_model = 'znnPFN32'
arch_dir = "architectures_saved/"

nevents = 10000

## ---------- CODE ----------

## Load testing data
x_raw = read_vectors("../v6.4/v6p4smallQCD2.root", nevents, False)
sig_raw = read_vectors("../v6.4/user.ebusch.515500.root", nevents, False)

## apply per-event scaling
bkg, sig = apply_EventScaling(x_raw, sig_raw)

# Load graph model
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## Load ae model
encoder = keras.models.load_model(arch_dir+ae_model+'_encoder_arch')
decoder = keras.models.load_model(arch_dir+ae_model+'_decoder_arch')
ae = AE(encoder,decoder)

ae.get_layer('encoder').load_weights(arch_dir+ae_model+'_encoder_weights.h5')
ae.get_layer('decoder').load_weights(arch_dir+ae_model+'_decoder_weights.h5')

ae.compile(optimizer=keras.optimizers.Adam())
#ae.summary()

## Load history
# with open(arch_dir+ae_model+"_history.json", 'r') as f:
#     h = json.load(f)
# print(h)
# print(type(h))

print ("Loaded model")

## Make graph representation
phi_bkg = graph.predict(bkg)
phi_sig = graph.predict(sig)

pred_phi_bkg = ae.predict(phi_bkg)['reconstruction']
pred_phi_sig = ae.predict(phi_sig)['reconstruction']

bkg_loss = keras.losses.mse(phi_bkg, pred_phi_bkg)
sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)
#bkg_loss, sig_loss = get_single_loss(ae, phi_bkg, phi_sig)

# --- Eval plots 
# # 1. Loss vs. epoch 
# plot_saved_loss(h, ae_model, "loss")
# if model.find('VAE') > -1:
#     plot_saved_loss(h, ae_model, "kl_loss")
#     plot_saved_loss(h, ae_model, "reco_loss")
# # 2. Anomaly score
plot_score(bkg_loss, sig_loss, False, False, ae_model)
#plot_score(bkg_loss, sig_loss, False, True, ae_model+"_xlog")
if ae_model.find('VAE') > -1:
    plot_score(bkg_kl_loss, sig_kl_loss, remove_outliers=False, xlog=True, extra_tag=model+"_KLD")
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, model_name+'_Reco')
# # 3. Signal Sensitivity Score
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, ae_model, True)
if ae_model.find('VAE') > -1:
    do_roc(bkg_reco_loss, sig_reco_loss, ae_model+'_Reco', True)
    do_roc(bkg_kl_loss, sig_kl_loss, ae_model+'_KLD', True)

