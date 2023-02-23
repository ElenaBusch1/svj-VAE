import numpy as np
import json
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from plot_helper import *
from models import *
from eval_helper import *

#---- REFERENCES 
#- Keras tutorials: https://blog.keras.io/building-autoencoders-in-keras.html
#- https://towardsdatascience.com (for any topic)
#- VRNN code: https://gitlab.cern.ch/dawillia/cxaod-scripts/-/tree/master/train
#- Weakly supervised CWoLa with PFNs: https://github.com/juliagonski/ILCAnomalies


## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
model_name = "AE"

# nEvents: 80% train, 10% valid, 10% test
# make sig %10 of bkg
x_events = 500000
y_events = 20000

## Model architecture
latent_dim = 2
encoding_dim = 16
phi_dim = 64

# Hyper parameters
nepochs = 30
batchsize = 32


## ---------- CODE  ----------

## Input options - set only 1 to true
hlvs = False #defunct
jets_1D = False
jets_2D = False
if model_name.find("PFN")>-1: jets_2D = True
else: jets_1D = True

## Check input setting
if(not ((hlvs ^ jets_1D ^ jets_2D) and not (hlvs and jets_1D and jets_2D)) ):
    print("Specify exactly one input type")

# params
if (hlvs):
    input_dim = 12
    scale = False
if (jets_1D):
    input_dim = 64
    scale = True
if (jets_2D):
    input_dim = [16, 4]
    scale = False

# Read in data
if (hlvs):
    x_raw = read_hlvs("../largerBackground.root", x_events)
    sig_raw = read_hlvs("../largerSignal.root", y_events)

#if (jets_1D):
#    x_raw = read_vectors("../v6smallQCD.root", x_events, flatten=True)
#    sig_raw = read_vectors("../user.ebusch.515502.root", y_events, flatten=True)

if (jets_1D or jets_2D):
    x_raw = read_vectors("../v6smallQCD.root", x_events, flatten=False)
    sig_raw = read_vectors("../user.ebusch.515500.root", y_events, flatten=False)

## Apply scaling
x= np.zeros((x_events,16,4))
sig= np.zeros((y_events,16,4))

x_nz = np.any(x_raw,2) #find zero padded events
sig_nz = np.any(sig_raw,2)

x_scale = x_raw[x_nz] #scale only non-zero jets
sig_scale = sig_raw[sig_nz]

x_fit = StandardScaler().fit_transform(x_scale) #do the scaling
sig_fit = StandardScaler().fit_transform(sig_scale)

x[x_nz]= x_fit #insert scaled values back into zero padded matrix
sig[sig_nz]= sig_fit

if (jets_1D):
    x = x.reshape(x_events,16*4)
    sig = sig.reshape(y_events,16*4)

print("Bkg input shape: ", x.shape)
print("Sig input shape: ", sig.shape)

## Train / Valid / Test split
x_train, x_test, _, _ = train_test_split(x, x, test_size=0.04) #done randomly
#x_valid, x_test, _, _ = train_test_split(x_temp, x_temp, test_size=0.5)

print("Length train :", len(x_train), ", test: ", len(x_test))

if (len(x_test) != len(sig)):
    print("WARNING: Testing with ", len(x_test), "background samples and ", len(sig), "signal samples")


## Define the model
if (model_name == "AE" or model_name == "VAE"):
    model_svj = get_model(model_name, input_dim, encoding_dim, latent_dim)
elif (model_name == "PFN_AE"  or model_name == "PFN_VAE"):
    model_svj, pfn = get_model(model_name, input_dim, encoding_dim, latent_dim, phi_dim)

## Train the model
h = model_svj.fit(x_train,
                epochs=nepochs,
                batch_size=batchsize,
                validation_split=0.04)
                #validation_data=x_valid)

## Save the model
model_svj.get_layer('encoder').save_weights(model_name+'_encoder_weights.h5')
model_svj.get_layer('decoder').save_weights(model_name+'_decoder_weights.h5')
model_svj.get_layer('encoder').save(model_name+'_encoder_arch')
model_svj.get_layer('decoder').save(model_name+'_decoder_arch')
if model_name.find('PFN') > -1:
    model_svj.get_layer('pfn').save_weights(model_name+'_pfn_weights.h5')
    model_svj.get_layer('pfn').save(model_name+'_pfn_arch')
with open(model_name+'_history.json', 'w') as f:
    json.dump(h.history, f)

print("Saved model")

## Evaluate multi Loss model
if (model_name.find("VAE") > -1):
    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss , sig_reco_loss = get_multi_loss(model_svj, x_test, sig)

## Evaluate single Loss model
else:
    bkg_loss, sig_loss = get_single_loss(model_svj, x_test, sig)


# # --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h, model_name, 'loss')
if model_name.find('VAE') > -1:
    plot_loss(h, model_name, 'kl_loss')
    plot_loss(h, model_name, 'reco_loss')
# 2. Anomaly score
plot_score(bkg_loss, sig_loss, False, False, model_name)
plot_score(bkg_loss, sig_loss, False, True, model_name+"_logx")
if model_name.find('VAE') > -1:
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, model_name+'_KLD')
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, model_name+'_Reco')
# 3. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, model_name, True)
if model_name.find('VAE') > -1:
    do_roc(bkg_reco_loss, sig_reco_loss, model_name+'_Reco', True)
    do_roc(bkg_kl_loss, sig_kl_loss, model_name+'_KLD', True)

#4. Plot inputs
#plot_vectors(x_fit,sig_fit,"scaled")
#plot_vectors(x_scale,sig_scale,"raw")
