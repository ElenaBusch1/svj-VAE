import numpy as np
from root_to_numpy import *
from tensorflow import keras
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from plot_helper import *
from models import get_ae

#---- REFERENCES 
#- Keras tutorials: https://blog.keras.io/building-autoencoders-in-keras.html
#- https://towardsdatascience.com (for any topic)
#- VRNN code: https://gitlab.cern.ch/dawillia/cxaod-scripts/-/tree/master/train
#- Weakly supervised CWoLa with PFNs: https://github.com/juliagonski/ILCAnomalies

# params
input_dim = 12 #start with N HLVs (from KP's BDT)
encoding_dim = 2
nepochs = 5
batchsize = 16

# model 
model_svj = get_ae(input_dim, encoding_dim)

# prepare input events
#x = np.random.rand(100, input_dim) #TODO: this is a dummy 100 events modeled by 12 vars, but need a function to pull these from JZW dijet
#sig = np.random.rand(5, input_dim) #TODO same function but loaded from SVJ sample vars
x = read_files("../smallBackground.root", 10000)
sig = read_files("../smallSignal.root", 500)

print(x)
print(type(x))
print(x.shape)
print(sig)
print(type(sig))
print(sig.shape)

x_temp, x_test, _, _ = train_test_split(x, x, test_size=0.05) #done randomly
x_train, x_valid, _, _ = train_test_split(x_temp,
                                          x_temp,
                                          test_size=0.1)
n_train = len(x_train)
n_valid = len(x_valid)
n_test = len(x_test)
print("Length train :", n_train, ", valid: ", n_valid, ", test: ", n_test)

# train
h = model_svj.fit(x_train, x_train,
                epochs=nepochs,
                batch_size=batchsize,
                shuffle=True,
                validation_data=(x_valid, x_valid))

# evaluate
# anomaly score = loss (TODO how to improve?)
truth_bkg = np.zeros(len(x_test))
truth_sig = np.ones(len(sig))
accu_bkg = model_svj.evaluate(x_test, truth_bkg)
accu_sig = model_svj.evaluate(sig, truth_sig)
pred_bkg = model_svj.predict(x_test)
pred_sig = model_svj.predict(sig)
pred_err_bkg = keras.losses.mse(pred_bkg, x_test).numpy()
pred_err_sig = keras.losses.mse(pred_sig, sig).numpy()

print("data evaluated loss, accuracy: ", accu_bkg)
print("sig evaluated loss, accuracy: ", accu_sig)
#print("truth_bkg: ", truth_bkg)
#print("truth_sig: ", truth_sig)
#print("pred_bkg: ", pred_bkg)
#print("pred_sig: ", pred_sig)
#print("pred_err_bkg: ", pred_err_bkg)
#print("pred_err_sig: ", pred_err_sig)

truth_labels = np.concatenate((truth_bkg, truth_sig))
eval_vals = np.concatenate((pred_err_bkg, pred_err_sig))
#eval_vals = [0.3, 0.2, 0.3, 0.5, 0.4, 0.6, 0.8, 0.7, 0.9, 0.7]

# --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h)
# 2. Histogram of reco error (loss) for JZW and evaled SVJ signals (test sets)
# 3. ROCs/AUCs using sklearn functions imported above  
# TODO
fpr, tpr, _ = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
auc = roc_auc_score(truth_labels, eval_vals) #Y_test = true labels, Y_predict = model-determined positive rate
make_roc(fpr,tpr,auc)
#make_single_roc(roc_curve, auc, 'tpr') #TODO plot tpr/sqrt(fpr) vs. fpr
