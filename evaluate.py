import numpy as np
from root_to_numpy import *
from tensorflow import keras
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_helper import *
from models import *

#json_file = open("model.json", "r")
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = keras.models.model_from_json(loaded_model_json)
#loaded_model.load_weights("model.h5")

#load testing data
x_raw = read_vectors("../largerBackground.root", 5000)
sig_raw = read_vectors("../WpJets.root", 5000)
x_scaler = StandardScaler()
sig_scaler = StandardScaler()
x_test = x_scaler.fit_transform(x_raw)
sig = sig_scaler.fit_transform(sig_raw)

#load model
model_svj = keras.models.load_model("test_model")
print ("Loaded model")

#evaluate
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

truth_labels = np.concatenate((truth_bkg, truth_sig))
eval_vals = np.concatenate((pred_err_bkg, pred_err_sig))

auc = roc_auc_score(truth_labels, eval_vals)
print("Iteration test", " AUC = ", auc)

# --- Eval plots 
# 1. Loss vs. epoch 
#plot_loss(h,1)
# 2. Histogram of reco error (loss) for JZW and evaled SVJ signals (test sets)
# 3. ROCs/AUCs using sklearn functions imported above  
# TODO
fpr, tpr, trh = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
#print("eval:  ", eval_vals)
#print("truth: ", truth_labels)
#print("fpr:   ", fpr)
#print("tpr:   ", tpr)
#print("trh:   ", trh)
#auc = roc_auc_score(truth_labels, eval_vals) #Y_test = true labels, Y_predict = model-determined positive rate
#make_roc(fpr,tpr,auc)
make_sic(fpr,tpr,auc)
#make_single_roc(roc_curve, auc, 'tpr') #TODO plot tpr/sqrt(fpr) vs. fpr
# 4. Anomaly score
plot_score(pred_err_bkg, pred_err_sig, False)

#5. Plot inputs
#plot_inputs(x,sig)
#plot_vectors(x_raw,sig_raw,"unscaled")
#plot_vectors(x_test,sig,"scaled")
