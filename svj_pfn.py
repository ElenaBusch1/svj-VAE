import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import *
from root_to_numpy import *
from plot_helper import *
from eval_helper import *

# Example usage
nevents = 500000
num_elements = 80
element_size = 7
encoding_dim = 32
latent_dim = 4
phi_dim = 64
nepochs= 100
batchsize_pfn=500
batchsize_ae=32

pfn_model = 'PFNv8'
arch_dir = "architectures_saved/"

## Load leading two jets
bkg_file = "../v9.1/skim0.user.ebusch.totalBkgALL.root"
#bkg2_file = "../v9.1/skim0.user.ebusch.METbkg.root"
#bkg3_file = "../v9.1/skim0.user.ebusch.topPhys.root"
sig_file = "../v9.1/user.ebusch.data16.root"
#bkgpt = read_flat_vars(bkg_file,nevents, ["met_met"], True)
#plot_single_variable([bkgpt], [np.ones(len(bkgpt))], ["totalBKG"],"met_met",True)

bkg = getTwoJetSystem(50000,bkg_file,[],use_weight=True)
#bkg2 = getTwoJetSystem(193900,bkg2_file,[],use_weight=False)
#bkg3 = getTwoJetSystem(80000,bkg3_file,[],use_weight=False)
sig = getTwoJetSystem(50000,sig_file,[],use_weight=False)
#bkg = np.concatenate((bkg1,bkg2,bkg3))
print("Total bkg shape:", bkg.shape)
print("Sig shape:", sig.shape)

# Plot inputs
plot_vectors(bkg,sig,"PFNinput")
#check_weights(nevents)
#plot_nTracks(bkg, "bkg")
#plot_nTracks(sig, "sig")
quit()

# Create truth target
input_data = np.concatenate((bkg,sig),axis=0)

truth_bkg = np.zeros(bkg.shape[0])
truth_sig = np.ones(sig.shape[0])

truth_1D = np.concatenate((truth_bkg,truth_sig))
truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)

print("Training shape, truth shape")
print(input_data.shape, truth.shape)

# Load the model
pfn,graph_orig = get_full_PFN([num_elements,element_size], phi_dim)
#pfn = get_dnn(160)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(input_data, truth, test_size=0.02)
#X_train, X_val, Y_train, Y_val = train_test_split(x_eval, y_eval, test_size=0.2)

# Fit scaler to training data, apply to testing data
x_train, scaler = apply_StandardScaling(x_train)
dump(scaler, arch_dir+pfn_model+'_scaler.bin', compress=True) #save the scaler
x_test,_ = apply_StandardScaling(x_test,scaler,False)

# Check the scaling & test/train split
bkg_train_scaled = x_train[y_train[:,0] == 1]
sig_train_scaled = x_train[y_train[:,0] == 0]
bkg_test_scaled = x_test[y_test[:,0] == 1]
sig_test_scaled = x_test[y_test[:,0] == 0]
plot_vectors(bkg_train_scaled,sig_train_scaled,"PFNtrain")
plot_vectors(bkg_train_scaled,sig_train_scaled,"PFNtest")

# Train
h = pfn.fit(x_train, y_train,
    epochs=nepochs,
    batch_size=batchsize_pfn,
    validation_split=0.2,
    verbose=1)

# Save the model
pfn.get_layer('graph').save_weights(arch_dir+pfn_model+'_graph_weights.h5')
pfn.get_layer('classifier').save_weights(arch_dir+pfn_model+'_classifier_weights.h5')
pfn.get_layer('graph').save(arch_dir+pfn_model+'_graph_arch')
pfn.get_layer('classifier').save(arch_dir+pfn_model+'_classifier_arch')

## PFN training plots
# 1. Loss vs. epoch 
plot_loss(h, pfn_model, 'loss')
# 2. Score 
preds = pfn.predict(x_test)
bkg_score = preds[:,1][y_test[:,1] == 0]
sig_score = preds[:,1][y_test[:,1] == 1]
plot_score(bkg_score, sig_score, False, False, pfn_model)
n_test = min(len(sig_score),len(bkg_score))
bkg_score = bkg_score[:n_test]
sig_score = sig_score[:n_test]
do_roc(bkg_score, sig_score, pfn_model, False)


