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
nevents = 50000
num_elements = 100
element_size = 7 # change here
#element_size = 4 # change here

encoding_dim = 32
latent_dim = 4
phi_dim = 64
nepochs=30
batchsize_pfn=500
batchsize_ae=32

pfn_model = 'PFN'
ae_model = 'PFN'
arch_dir = "architectures_saved/"
## Load leading two jets
# Plot inputs before the jet rotation
bool_weight=True
if bool_weight:weight_tag='weightSampled'
else:weight_tag='notweightSampled'
tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'

bkg, sig = getTwoJetSystem(nevents,nevents, tag_file=tag+"_NSNR", tag_title=weight_tag+"_NSNR", bool_weight=bool_weight)

# Plot inputs after the jet rotation
#plot_vectors(bkg,sig,"PFNrotated")
#plot_vectors(bkg,sig,tag_file=tag+"_NSYR", tag_title=tag+"_NSYR")
plot_vectors(bkg,sig,tag_file=tag+"_NSYR", tag_title=weight_tag+"_NSYR")

# Create truth target
input_data = np.concatenate((bkg,sig),axis=0)

truth_bkg = np.zeros(bkg.shape[0])
truth_sig = np.ones(sig.shape[0])

truth_1D = np.concatenate((truth_bkg,truth_sig))
truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)

## MAKE 1D DATA - DNN only
#input_data = input_data.reshape(input_data.shape[0],40*4)

print("Training shape, truth shape")
print(input_data.shape, truth.shape)

# Load the model
pfn,graph_orig = get_full_PFN([num_elements,element_size], phi_dim)
#pfn = get_dnn(160)

# Split the data 
# ADD 5/23/23 
x_train, x_test, y_train, y_test = train_test_split(input_data, truth, test_size=0.02, random_state=42)
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

# Plot inputs before the jet rotation 
print('bkg_train_scaled, bkg_test_scaled',bkg_train_scaled.shape, bkg_test_scaled.shape)
#print(bkg_test_scaled)
bkg_scaled=np.concatenate((bkg_train_scaled, bkg_test_scaled), axis=0)
sig_scaled=np.concatenate((sig_train_scaled, sig_test_scaled), axis=0)
plot_vectors(bkg_scaled,sig_scaled,tag_file=tag+"_YSYR", tag_title=weight_tag+"_YSYR")

plot_vectors(bkg_train_scaled,sig_train_scaled,tag_file=tag+"_train", tag_title=weight_tag+"_train")
plot_vectors(bkg_test_scaled,sig_test_scaled,tag_file=tag+"_test", tag_title=weight_tag+"_test")

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
#plot_loss(h, pfn_model, 'loss')
plot_loss(h, loss= 'loss', tag_file=tag, tag_title=tag)
# 2. Score 
preds = pfn.predict(x_test)
bkg_score = preds[:,1][y_test[:,1] == 0]
sig_score = preds[:,1][y_test[:,1] == 1]
plot_score(bkg_score, sig_score, False, False,tag_file=tag, tag_title=tag)
#plot_score(bkg_score, sig_score, False, False, pfn_model)
n_test = min(len(sig_score),len(bkg_score))
bkg_score = bkg_score[:n_test]
sig_score = sig_score[:n_test]
do_roc(bkg_score, sig_score, tag_file=tag, tag_title=tag, make_transformed_plot=False)


