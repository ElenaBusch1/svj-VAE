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
from eval_helper import *

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
model = "PFN_AE"

nevents = 5000

## ---------- CODE ----------

hlvs = False #defunct
jets_1D = False
jets_2D = False
if model.find("PFN")>-1: jets_2D = True
else: jets_1D = True

## Load testing data
x_raw = read_vectors("../v6smallQCD.root", nevents, False)
sig_raw = read_vectors("../user.ebusch.515499.root", nevents, False)

## Apply scaling
x= np.zeros((nevents,16,4))
sig= np.zeros((nevents,16,4))

x_nz = np.any(x_raw,2) #find zero padded events
sig_nz = np.any(sig_raw,2)

x_scale = x_raw[x_nz] #scale only non-zero jets
sig_scale = sig_raw[sig_nz]

x_fit = StandardScaler().fit_transform(x_scale) #do the scaling
sig_fit = StandardScaler().fit_transform(sig_scale)

x[x_nz]= x_fit #insert scaled values back into zero padded matrix
sig[sig_nz]= sig_fit

if (jets_1D):
    x = x.reshape(nevents,16*4)
    sig = sig.reshape(nevents,16*4)


## Load model
encoder = keras.models.load_model(model+'_encoder_arch')
decoder = keras.models.load_model(model+'_decoder_arch')
if model.find("PFN") >-1:
    pfn = keras.models.load_model(model+'_pfn_arch')

if model == "AE":
    model_svj = AE(encoder,decoder)
elif model == "VAE":
    model_svj = VAE(encoder,decoder)
elif model == "PFN_AE":
    model_svj = PFN_AE(pfn,encoder,decoder)
elif model == "PFN_VAE":
    model_svj = PFN_VAE(pfn,encoder,decoder)

model_svj.get_layer('encoder').load_weights(model+'_encoder_weights.h5')
model_svj.get_layer('decoder').load_weights(model+'_decoder_weights.h5')
if model.find("PFN") >-1:
    model_svj.get_layer('pfn').load_weights(model+'_pfn_weights.h5')

model_svj.compile(optimizer=keras.optimizers.Adam())
#model_svj.summary()

## Load history
with open(model+"_history.json", 'r') as f:
    h = json.load(f)
print(h)
print(type(h))

print ("Loaded model")

## Evaluate multi Loss model
# if (model.find("VAE") > -1):
#     bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss , sig_reco_loss = get_multi_loss(model_svj, x, sig)
# 
# ## Evaluate single Loss model
# else:
#     bkg_loss, sig_loss = get_single_loss(model_svj, x, sig)

# --- Eval plots 
# 1. Loss vs. epoch 
plot_saved_loss(h, model, "loss")
# if model.find('VAE') > -1:
#     plot_saved_loss(h, model, "kl_loss")
#     plot_saved_loss(h, model, "reco_loss")
# # 2. Anomaly score
# plot_score(bkg_loss, sig_loss, False, False, model)
# plot_score(bkg_loss, sig_loss, False, True, model+"_xlog")
# if model.find('VAE') > -1:
#     plot_score(bkg_kl_loss, sig_kl_loss, remove_outliers=False, xlog=True, extra_tag=model+"_KLD")
#     plot_score(bkg_reco_loss, sig_reco_loss, False, False, model_name+'_Reco')
# # 3. ROCs/AUCs using sklearn functions imported above  
# do_roc(bkg_loss, sig_loss, model_name, True)
# if model_name.find('VAE') > -1:
#     do_roc(bkg_reco_loss, sig_reco_loss, model_name+'_Reco', True)
#     do_roc(bkg_kl_loss, sig_kl_loss, model_name+'_KLD', True)

