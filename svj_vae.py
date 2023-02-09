import numpy as np
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_helper import *
from models import *

#---- REFERENCES 
#- Keras tutorials: https://blog.keras.io/building-autoencoders-in-keras.html
#- https://towardsdatascience.com (for any topic)
#- VRNN code: https://gitlab.cern.ch/dawillia/cxaod-scripts/-/tree/master/train
#- Weakly supervised CWoLa with PFNs: https://github.com/juliagonski/ILCAnomalies

# input
hlvs = False
jets = True
if(not hlvs and not jets):
	print("No input type specified")

# params
if (jets):
	input_dim = 64 #start with N HLVs (from KP's BDT)
	encoding_dim = 16
	
if (hlvs):
	input_dim = 12
	encoding_dim = 4

nepochs = 50
batchsize = 64

# model 
#model_svj = get_better_ae(input_dim, encoding_dim)

# prepare input events
#x = np.random.rand(10000, input_dim) #TODO: this is a dummy 100 events modeled by 12 vars, but need a function to pull these from JZW dijet
#sig = np.random.rand(500, input_dim) #TODO same function but loaded from SVJ sample vars
if (hlvs):
	x_raw = read_hlvs("../largerBackground.root", 30000)
	sig_raw = read_hlvs("../largerSignal.root", 1500)

if (jets):
	x_raw = read_vectors("../v6smallQCD.root", 50000)
	sig_raw = read_vectors("../user.ebusch.515502.root", 1000)

x_scaler = StandardScaler()
sig_scaler = StandardScaler()
x = x_scaler.fit_transform(x_raw)
sig = sig_scaler.fit_transform(sig_raw)

#x = x_raw
#sig = sig_raw
print(x)
print(type(x))
print(x.shape)
print(sig)
print(type(sig))
print(sig.shape)

model_svj = get_vae2(input_dim, encoding_dim)

x_temp, x_test, _, _ = train_test_split(x, x, test_size=0.02) #done randomly
x_train, x_valid, _, _ = train_test_split(x_temp,
                                          x_temp,
                                          test_size=0.05)
n_train = len(x_train)
n_valid = len(x_valid)
n_test = len(x_test)
print("Length train :", n_train, ", valid: ", n_valid, ", test: ", n_test)

# train
h = model_svj.fit(x_train,
                epochs=nepochs,
                batch_size=batchsize,
                #shuffle=True,
                validation_data=(x_valid,x_valid) )

#save
#model_svj.save("vae_getvae2")
#saved_model.save(model_svj, "vae_getvae2")
model_svj.get_layer('encoder').save_weights('encoder2_weights.h5')
model_svj.get_layer('decoder').save_weights('decoder2_weights.h5')
model_svj.get_layer('encoder').save('encoder2_arch')
model_svj.get_layer('decoder').save('decoder2_arch')

print("Saved model")

# evaluate
# anomaly score = loss (TODO how to improve?)
truth_bkg = np.zeros(len(x_test))
truth_sig = np.ones(len(sig))
accu_bkg = model_svj.evaluate(x_test, x_test)
accu_sig = model_svj.evaluate(sig, sig)
pred_bkg = model_svj.predict(x_test)
pred_sig = model_svj.predict(sig)
#pred_err_bkg = keras.losses.mse(pred_bkg, x_test).numpy()
#pred_err_sig = keras.losses.mse(pred_sig, sig).numpy()

#print(model_svj.metrics_names)
#print("data evaluated loss, accuracy: ", accu_bkg)
#print("sig evaluated loss, accuracy: ", accu_sig)
#print("truth_bkg: ", truth_bkg)
#print("truth_sig: ", truth_sig)
#print("pred_bkg: ", pred_bkg)
#print("pred_sig: ", pred_sig)
#print("pred_err_bkg: ", pred_err_bkg)
#print("pred_err_sig: ", pred_err_sig)

#truth_labels = np.concatenate((truth_bkg, truth_sig))
#eval_vals = np.concatenate((pred_err_bkg, pred_err_sig))
#auc = roc_auc_score(truth_labels, eval_vals)
#print("Iteration test", " AUC = ", auc)

# --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h,1)
# 2. Histogram of reco error (loss) for JZW and evaled SVJ signals (test sets)
# 3. ROCs/AUCs using sklearn functions imported above  
# 
#fpr, tpr, trh = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
#auc = roc_auc_score(truth_labels, eval_vals) #Y_test = true labels, Y_predict = model-determined positive rate
#make_roc(fpr,tpr,auc)
# 4. Anomaly score
#plot_score(pred_err_bkg, pred_err_sig)

#5. Plot inputs
#plot_vectors(x,sig,"vec")
