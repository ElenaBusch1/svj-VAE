import numpy as np
import json
import joblib
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
arch_dir = "architectures_saved/"

# nEvents: 80% train, 10% valid, 10% test
# make sig %10 of bkg
x_events = 100000
y_events = 10000

## Model architecture
latent_dim = 12
encoding_dim = 32
phi_dim = 64 #Should we be changing this? Originally 64

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
    #input_dim = 64
    input_dim = 400
    scale = True
if (jets_2D):
    input_dim = [16, 4]
    scale = False

# Read in data
if (hlvs):
    x_in = read_hlvs("../largerBackground.root", x_events)
    sig_in = read_hlvs("../largerSignal.root", y_events)

if (jets_1D or jets_2D):
   x, sig = getTwoJetSystem(x_events,y_events)
##  # 4. Plot inputs
#x_sel_nz = remove_zero_padding(bkg_sel)
#sig_sel_nz = remove_zero_padding(sig_sel)
##  x_nz = remove_zero_padding(bkg)
##  sig_nz = remove_zero_padding(sig)
#plot_vectors(x_sel_nz,sig_sel_nz,"AEraw")
##  plot_vectors(x_nz,sig_nz,"AErotated")

## Non-mulitjet
##x_2D,scaler = apply_StandardScaling(bkg)
##sig_2D,_ = apply_StandardScaling(sig, scaler, False)

#print(sig_2D.shape)
#print(x_2D.shape)

#plot_vectors(remove_zero_padding(x_2D),remove_zero_padding(sig_2D),"AEscaled_avg")
#quit()

##  if (jets_1D):
##      x = x_2D.reshape(x_2D.shape[0],8*4)
##      sig = sig_2D.reshape(sig_2D.shape[0],8*4)
##  else:
##      x = x_2D
##      sig = sig_2D

print("Bkg input shape: ", x.shape)
print("Sig input shape: ", sig.shape)

## Train / Valid / Test split
x_train, x_test, _, _ = train_test_split(x, x, test_size=sig.shape[0]) #done randomly
#x_valid, x_test, _, _ = train_test_split(x_temp, x_temp, test_size=0.5)
#x_train = reshape_3D(x_train, 16, 4)
#x_test = reshape_3D(x_test, 16, 4)
#sig = reshape_3D(sig, 16, 4)
x_train, scaler = apply_StandardScaling(x_train)
x_test,_ = apply_StandardScaling(x_test,scaler,False)
sig,_ = apply_StandardScaling(sig,scaler,False)

x_train = x_train.reshape(x_train.shape[0], 100*4)
x_test = x_test.reshape(x_test.shape[0], 100*4)
#x_test,_ = apply_StandardScaling(x_test_1,scaler,False)
sig = sig.reshape(sig.shape[0],100*4)
plot_vectors(x_train,sig,"AEtrain")
plot_vectors(x_test,sig,"AEtest")

print("Train shape:", x_train.shape, ", test shape: ", x_test.shape)
print("scaled sig shape:", sig.shape)

if (len(x_test) != len(sig)):
    print("WARNING: Testing with ", len(x_test), "background samples and ", len(sig), "signal samples")

## Define the model
if (model_name == "AE" or model_name == "VAE"):
    model_svj = get_model(model_name, input_dim, encoding_dim, latent_dim)
elif (model_name == "PFN_AE"  or model_name == "PFN_VAE"):
    model_svj, pfn = get_model(model_name, input_dim, encoding_dim, latent_dim, phi_dim)

#quit()

## Train the model
h = model_svj.fit(x_train,
                epochs=nepochs,
                batch_size=batchsize,
                validation_split=0.2)
                #validation_data=x_valid)

## Save the model
model_svj.get_layer('encoder').save_weights(arch_dir+model_name+'8_encoder_weights.h5')
model_svj.get_layer('decoder').save_weights(arch_dir+model_name+'8_decoder_weights.h5')
model_svj.get_layer('encoder').save(arch_dir+model_name+'8_encoder_arch')
model_svj.get_layer('decoder').save(arch_dir+model_name+'8_decoder_arch')
if model_name.find('PFN') > -1:
    model_svj.get_layer('pfn').save_weights(arch_dir+model_name+'_pfn_weights.h5')
    model_svj.get_layer('pfn').save(arch_dir+model_name+'_pfn_arch')
with open(arch_dir+model_name+'8_history.json', 'w') as f:
    json.dump(h.history, f)
print("Saved model")

## Evaluate multi Loss model
if (model_name.find("VAE") > -1):
    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss , sig_reco_loss = get_multi_loss(model_svj, x_test, sig)

## Evaluate single Loss model
else:
    pred_bkg = model_svj.predict(x_test)['reconstruction']
    pred_sig = model_svj.predict(sig)['reconstruction']
    plot_vectors(pred_bkg,pred_sig,"AEpred")
    
    bkg_loss = keras.losses.mse(x_test, pred_bkg)
    sig_loss = keras.losses.mse(sig, pred_sig)
    #bkg_loss, sig_loss = get_single_loss(model_svj, x_test, sig)


# # --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h, model_name, 'loss')
if model_name.find('VAE') > -1:
    plot_loss(h, model_name, 'kl_loss')
    plot_loss(h, model_name, 'reco_loss')

# 2. Anomaly score
plot_score(bkg_loss, sig_loss, False, False, model_name)
#plot_score(bkg_loss, sig_loss, False, True, model_name+"_logx")
if model_name.find('VAE') > -1:
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, model_name+'_KLD')
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, model_name+'_Reco')

# # 3. Signal Sensitivity Score
# score = getSignalSensitivityScore(bkg_loss, sig_loss)
# print("95 percentile score = ",score)
# if model_name.find('VAE') > -1:
#     score_kld = getSignalSensitivityScore(bkg_kl_loss, sig_kl_loss)
#     print("95 percentile score_kl =", score_kld)

# 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, model_name, True)
if model_name.find('VAE') > -1:
    do_roc(bkg_reco_loss, sig_reco_loss, model_name+'_Reco', True)
    do_roc(bkg_kl_loss, sig_kl_loss, model_name+'_KLD', True)

#4. Plot inputs
# x_raw_nz = remove_zero_padding(x_raw)
# sig_raw_nz = remove_zero_padding(sig_raw)
# x_nz = remove_zero_padding(x_2D)
# sig_nz = remove_zero_padding(sig_2D)
# 
# plot_vectors(x_raw_nz,sig_raw_nz,"raw")
# plot_vectors(x_nz,sig_nz,"scaled")
