import numpy as np
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
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
nevents = 10000
x_raw = read_vectors("../v6smallQCD.root", nevents)
sig_raw = read_vectors("../user.ebusch.515502.root", nevents)
x_scaler = StandardScaler()
sig_scaler = StandardScaler()
x_test = x_scaler.fit_transform(x_raw)
sig = sig_scaler.fit_transform(sig_raw)

#load model
#model_svj = keras.models.load_model("vae_getvae2")

#load vae
encoder = keras.models.load_model('encoder_arch')
decoder = keras.models.load_model('decoder_arch')
model_svj = VAE(encoder,decoder)
model_svj.get_layer('encoder').load_weights('encoder_weights.h5')
model_svj.get_layer('decoder').load_weights('decoder_weights.h5')
model_svj.compile(optimizer=keras.optimizers.Adam())

print ("Loaded model")
#model_svj.summary()
#print ("Metric names")
print(model_svj.metrics_names)

#evaluate
truth_bkg = np.zeros(len(x_test))
truth_sig = np.ones(len(sig))

bkg_loss = []
sig_loss = []
bkg_kld_loss = []
sig_kld_loss = []

#for x,y in zip(x_test,sig):
step_size = 4
for i in range(0,nevents, step_size):
  xt = x_test[i:i+step_size]
  yt = sig[i:i+step_size]
  #xt = x[np.newaxis, :] #give x and y the correct shape (,64)
  #yt = y[np.newaxis, :]

  # NOTE - unclear why they are printed in this order, but it seems to be the case
  x_reco,x_kld,x_loss = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
  y_reco,y_kld,y_loss = model_svj.evaluate(yt, batch_size = step_size, verbose=0)

  bkg_loss.append(x_loss)
  bkg_kld_loss.append(x_kld)
  sig_loss.append(y_loss)
  sig_kld_loss.append(y_kld)
  if i%1000 == 0: print("Processed", i, "events")

#accu_bkg = model_svj.evaluate(x_test, truth_bkg)
#accu_sig = model_svj.evaluate(sig, truth_sig)
#pred_bkg = model_svj.predict(x_test)
#pred_sig = model_svj.predict(sig)
#pred_err_bkg = keras.losses.mse(pred_bkg, x_test).numpy()
#pred_err_sig = keras.losses.mse(pred_sig, sig).numpy()

#print(bkg_loss)
#print(sig_kld_loss)

#print("data evaluated", model_svj.metrics_names, ":", accu_bkg)
#print("sig evaluated", model_svj.metrics_names, ";", accu_sig)
#print("data predict", pred_bkg.shape)
#print(pred_bkg)
#truth_labels = np.concatenate((truth_bkg, truth_sig))
#eval_vals = np.concatenate((pred_bkg, pred_sig))

#auc = roc_auc_score(truth_labels, eval_vals)
#print("Iteration test", " AUC = ", auc)

# --- Eval plots 
# 1. Loss vs. epoch 
#plot_loss(h,1)
# 2. Histogram of reco error (loss) for JZW and evaled SVJ signals (test sets)
# 3. ROCs/AUCs using sklearn functions imported above  
# TODO
#fpr, tpr, trh = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
#print("eval:  ", eval_vals)
#print("truth: ", truth_labels)
#print("fpr:   ", fpr)
#print("tpr:   ", tpr)
#print("trh:   ", trh)
#auc = roc_auc_score(truth_labels, eval_vals) #Y_test = true labels, Y_predict = model-determined positive rate
#make_roc(fpr,tpr,auc)
#make_sic(fpr,tpr,auc)
#make_single_roc(roc_curve, auc, 'tpr') #TODO plot tpr/sqrt(fpr) vs. fpr
# 4. Anomaly score
plot_score(bkg_loss, sig_loss, False, "total_loss_515502")
plot_score(bkg_kld_loss, sig_kld_loss, False, "kld_515502")

#5. Plot inputs
#plot_inputs(x,sig)
#plot_vectors(x_raw,sig_raw,"unscaled")
#plot_vectors(x_test,sig,"scaled")
