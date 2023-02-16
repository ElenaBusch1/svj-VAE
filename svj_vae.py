import numpy as np
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_helper import *
from models import *
from evaluate import get_single_loss, get_multi_loss

#---- REFERENCES 
#- Keras tutorials: https://blog.keras.io/building-autoencoders-in-keras.html
#- https://towardsdatascience.com (for any topic)
#- VRNN code: https://gitlab.cern.ch/dawillia/cxaod-scripts/-/tree/master/train
#- Weakly supervised CWoLa with PFNs: https://github.com/juliagonski/ILCAnomalies


## ---------- USER PARAMETERS ----------
## Model options:
##    "ae", "vae", "pfn_ae", "pfn_vae"
model_name = "ae"
model_tag = "_"
plot_tag = "ae_vae_pfn_compare"

## Input options - set only 1 to true
hlvs = False
jets_1D = True
jets_2D = False

# nEvents: 80% train, 10% valid, 10% test
# make sig %10 of bkg
x_events = 5000
y_events = 500

## Model architecture
latent_dim = 2
encoding_dim = 32
phi_dim = 64

# Hyper parameters
nepochs = 10
batchsize = 50


## ---------- CODE  ----------

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

if (jets_1D):
    x_raw = read_vectors("../v6smallQCD.root", x_events, flatten=True)
    sig_raw = read_vectors("../user.ebusch.515502.root", y_events, flatten=True)

if (jets_2D):
    x_raw = read_vectors("../v6smallQCD.root", x_events, flatten=False)
    sig_raw = read_vectors("../user.ebusch.515502.root", y_events, flatten=False)

## Apply scaling
if (scale):
    x_scaler = StandardScaler()
    sig_scaler = StandardScaler()
    x = x_scaler.fit_transform(x_raw)
    sig = sig_scaler.fit_transform(sig_raw)
else:
    x = x_raw
    sig = sig_raw

print("Bkg input shape: ", x.shape)
print("Sig input shape: ", sig.shape)

## Train / Valid / Test split
x_train, x_temp, _, _ = train_test_split(x, x, test_size=0.2) #done randomly
x_valid, x_test, _, _ = train_test_split(x_temp, x_temp, test_size=0.5)

print("Length train :", len(x_train), ", valid: ", len(x_valid), ", test: ", len(x_test))

if (len(x_test) != len(sig)):
    print("WARNING: Testing with ", len(x_test), "background samples and ", len(sig), "signal samples")


## Define the model
if (model_name == "ae" or model_name == "vae"):
    model_svj = get_model(model_name, input_dim, encoding_dim, latent_dim)
elif (model_name == "pfn_ae"  or model_name == "pfn_vae"):
    model_svj, pfn = get_model(model_name, input_dim, encoding_dim, latent_dim, phi_dim)

## Train the model
h = model_svj.fit(x_train,
                epochs=nepochs,
                batch_size=batchsize,
                validation_split=0.1)
                #validation_data=x_valid)

## Save the model
#model_svj.save("vae_getvae2")
#saved_model.save(model_svj, "vae_getvae2")
model_svj.get_layer('encoder').save_weights(model_name+model_tag+'encoder_weights.h5')
model_svj.get_layer('decoder').save_weights(model_name+model_tag+'decoder_weights.h5')
model_svj.get_layer('encoder').save(model_name+model_tag+'encoder_arch')
model_svj.get_layer('decoder').save(model_name+model_tag+'decoder_arch')

print("Saved model")

## Evaluate Single Loss model
if (model_name.find("vae") == -1):
    bkg_loss, sig_loss = get_single_loss(model_svj, x_test, sig)

## Evaluate multi Loss model
else:
    bkg_loss, sig_loss, _, _ , _ , _ = get_multi_loss(model_svj, x_test, sig)


# --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h)
# 2. Anomaly score
plot_score(bkg_loss, sig_loss, False, True, model_name+"orig")
# 3. ROCs/AUCs using sklearn functions imported above  
nevents = len(sig_loss)
truth_sig = np.ones(nevents)
truth_bkg = np.zeros(nevents)
truth_labels = np.concatenate((truth_bkg, truth_sig))
eval_vals = np.concatenate((bkg_loss,sig_loss))
eval_max = max(eval_vals)
eval_min = min(eval_vals)
eval_transformed = [(x - eval_min)/eval_max for x in eval_vals]
bkg_transformed = [(x - eval_min)/eval_max for x in bkg_loss]
sig_transformed = [(x - eval_min)/eval_max for x in sig_loss]
plot_score(bkg_transformed, sig_transformed, False, False, model_name+"transformed")
print("truth", truth_labels)
print("eval", eval_transformed)

fpr, tpr, trh = roc_curve(truth_labels, eval_transformed) #[fpr,tpr]
auc = roc_auc_score(truth_labels, eval_vals)
print("AUC: ", auc)
make_roc(fpr,tpr,auc)
#4. Plot inputs
#plot_vectors(x,sig,"vec")
