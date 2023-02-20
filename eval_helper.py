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
#nevents = 10000
#x_raw = read_vectors("../v6smallQCD.root", nevents)
#sig_raw = read_vectors("../user.ebusch.515499.root", nevents)
#x_scaler = StandardScaler()
#sig_scaler = StandardScaler()
#x_test = x_scaler.fit_transform(x_raw)
#sig = sig_scaler.fit_transform(sig_raw)
#
##load model
##model_svj = keras.models.load_model("vae_getvae2")
#
##load vae
#encoder = keras.models.load_model('encoder2_arch')
#decoder = keras.models.load_model('decoder2_arch')
#model_svj = VAE(encoder,decoder)
#model_svj.get_layer('encoder').load_weights('encoder2_weights.h5')
#model_svj.get_layer('decoder').load_weights('decoder2_weights.h5')
#model_svj.compile(optimizer=keras.optimizers.Adam())
#
#print ("Loaded model")
##model_svj.summary()
##print ("Metric names")
#print(model_svj.metrics_names)
#
##evaluate
#truth_bkg = np.zeros(len(x_test))
#truth_sig = np.ones(len(sig))


def get_multi_loss(model_svj, x_test, y_test):
    bkg_total_loss = []
    sig_total_loss = []
    bkg_kld_loss = []
    sig_kld_loss = []
    bkg_reco_loss = []
    sig_reco_loss = []
    nevents = min(len(y_test),len(x_test))
    step_size = 4
    for i in range(0,nevents, step_size):
        xt = x_test[i:i+step_size]
        yt = y_test[i:i+step_size]
      
        # NOTE - unclear why they are printed in this order, but it seems to be the case
        x_loss,x_reco,x_kld = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
        y_loss,y_reco,y_kld = model_svj.evaluate(yt, batch_size = step_size, verbose=0)
      
        bkg_total_loss.append(x_loss)
        sig_total_loss.append(y_loss)
        bkg_kld_loss.append(x_kld)
        sig_kld_loss.append(y_kld)
        bkg_reco_loss.append(x_reco)
        sig_reco_loss.append(y_reco)
        if i%100 == 0: print("Processed", i, "events")

    return bkg_total_loss, sig_total_loss, bkg_kld_loss, sig_kld_loss, bkg_reco_loss, sig_reco_loss

def get_single_loss(model_svj, x_test, y_test):
    bkg_loss = []
    sig_loss = []
    nevents = min(len(y_test),len(x_test))
    step_size = 4
    for i in range(0,nevents, step_size):
        xt = x_test[i:i+step_size]
        yt = y_test[i:i+step_size]
    
        x_loss = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
        y_loss = model_svj.evaluate(yt, batch_size = step_size, verbose=0)
        
        bkg_loss.append(x_loss)
        sig_loss.append(y_loss)
        if i%100 == 0: print("Processed", i, "events")

    return bkg_loss, sig_loss

def transform_loss(bkg_loss, sig_loss, make_plot=False, plot_tag=''):
    nevents = len(sig_loss)
    truth_sig = np.ones(nevents)
    truth_bkg = np.zeros(nevents)
    truth_labels = np.concatenate((truth_bkg, truth_sig))
    eval_vals = np.concatenate((bkg_loss,sig_loss))
    eval_min = min(eval_vals)
    eval_max = max(eval_vals)-eval_min
    eval_transformed = [(x - eval_min)/eval_max for x in eval_vals]
    bkg_transformed = [(x - eval_min)/eval_max for x in bkg_loss]
    sig_transformed = [(x - eval_min)/eval_max for x in sig_loss]
    if make_plot:
        plot_score(bkg_transformed, sig_transformed, False, False, plot_tag+'_Transformed')
    return truth_labels, eval_vals 

def do_roc(bkg_loss, sig_loss, plot_tag, make_transformed_plot=False):
    truth_labels, eval_vals = transform_loss(bkg_loss, sig_loss, make_plot=make_transformed_plot, plot_tag=plot_tag) 
    fpr, tpr, trh = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
    auc = roc_auc_score(truth_labels, eval_vals)
    print("AUC - "+plot_tag+": ", auc)
    make_roc(fpr,tpr,auc,plot_tag)
    make_sic(fpr,tpr,auc,plot_tag)

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
#plot_score(bkg_loss, sig_loss, False, "total_loss_515499")
#plot_score(bkg_kld_loss, sig_kld_loss, False, "kld_515499")

#5. Plot inputs
#plot_inputs(x,sig)
#plot_vectors(x_raw,sig_raw,"unscaled")
#plot_vectors(x_test,sig,"scaled")
