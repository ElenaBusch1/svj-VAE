import numpy as np
import json
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from plot_helper import *
from models import *
from models_archive import *
from eval_helper import *
import matplotlib.pyplot as plt
import h5py

def my_metric(s,b):
    return np.sqrt(2*((s+b)*np.log(1+s/b)-s))

def get_weighted_elements_h5(my_weight_array, nEvents):
    np.random.seed(1)
    idx = np.random.choice( my_weight_array.size,size= nEvents, p=my_weight_array/float(my_weight_array.sum()),replace=False) # IMPT that replace=False so that event is picked only once
    return idx

def mT_shape_compare():
  with h5py.File("../v8.1/v8p1_PFNv1_QCDskim.hdf5","r") as f:
    bkg_datav1 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_1.hdf5","r") as f:
    bkg_data1 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_2.hdf5","r") as f:
    bkg_data2 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_3.hdf5","r") as f:
    bkg_data3 = f.get('data')[:]

  #bkg_data = np.concatenate((bkg_data1,bkg_data2))
  bkg_datav2 = np.concatenate((bkg_data1,bkg_data2, bkg_data3))

  #variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg_lossv1 = bkg_datav1["score"]
  bkg_lossv2 = bkg_datav2["score"]
  bkg1_weights = np.reshape(bkg_datav1["weight"],len(bkg_datav1["weight"]))
  bkg2_weights = np.reshape(bkg_datav2["weight"],len(bkg_datav2["weight"]))
  dsids = [515518,515522,515523,515515]
  for dsid in dsids:
    with h5py.File("../v8.1/v1_hdf5/v8p1_PFN_"+str(dsid)+".hdf5","r") as f:
      sigv1_data = f.get('data')[:]
    with h5py.File("../v8.1/v8p1_PFNv2_"+str(dsid)+".hdf5","r") as f:
      sigv2_data = f.get('data')[:]
    sigv1_loss = sigv1_data["score"]
    sigv2_loss = sigv2_data["score"]
    #bkgv1_idx = get_weighted_elements_h5(bkg1_weights,len(sigv1_loss))
    #bkgv2_idx = get_weighted_elements_h5(bkg2_weights,len(sigv2_loss))
    bkgv1_loss = bkg_lossv1[:len(sigv1_loss)] 
    bkgv2_loss = bkg_lossv2[:len(sigv2_loss)]
    d = [bkgv1_loss, bkgv2_loss, sigv1_loss, sigv2_loss]
    w = [np.ones(len(x)) for x in d]
    lab = ["v1 BKG (bad sel)", "v2 BKG (bad sel)", "v1 Signal", "v2 Signal"]
    labels = [l+str(len(ds)) for l,ds in zip(lab,d)]
    plot_single_variable(d,w,labels, "score"+str(dsid), logy=False) 
    

def cms_mT_plots():
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_1.hdf5","r") as f:
    bkg_data1 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_2.hdf5","r") as f:
    bkg_data2 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_3.hdf5","r") as f:
    bkg_data3 = f.get('data')[:]

  bkg_data = np.concatenate((bkg_data1,bkg_data2,bkg_data3))
  
  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  #bkg20 = np.percentile(bkg_loss,99.5)
  
  with h5py.File("../v8.1/v8p1_PFNv2_515499.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_515507.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_515504.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  selection0 = (bkg_data["rT"] > 0.25) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deta_12"] < 1.5)
  selection1 = (sig1_data["rT"] > 0.25) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deta_12"] < 1.5)
  selection2 = (sig2_data["rT"] > 0.25) & (sig2_data["dphi_min"] < 0.8) & (sig2_data["deta_12"] < 1.5)
  selection3 = (sig3_data["rT"] > 0.25) & (sig3_data["dphi_min"] < 0.8) & (sig3_data["deta_12"] < 1.5)
  selection4 = (sig4_data["rT"] > 0.25) & (sig4_data["dphi_min"] < 0.8) & (sig4_data["deta_12"] < 1.5)
  
  print(variables)
  
  w0 = 100*bkg_data["weight"][selection0] 
  w1 = sig1_data["weight"][selection1] 
  w2 = sig2_data["weight"][selection2] 
  w3 = sig3_data["weight"][selection3] 
  w4 = sig4_data["weight"][selection4] 
  
  w = [w0,w1,w2,w3,w4]
  
  labels = ["QCD", "2000 GeV,0.2", "3000 GeV,0.2", "4000 GeV,0.6", "2500 GeV,0.4"]
  for var in ["mT_jj"]:
    if (var=="weight" or var=="mcEventWeight"): continue
    bkg = bkg_data[var][selection0]
    sig1 = sig1_data[var][selection1] 
    sig2 = sig2_data[var][selection2] 
    sig3 = sig3_data[var][selection3] 
    sig4 = sig4_data[var][selection4] 
    labels[0] += "({0:.0%})".format(len(bkg)/len(bkg_data[var]))
    labels[1] += "({0:.0%})".format(len(sig1)/len(sig1_data[var]))
    labels[2] += "({0:.0%})".format(len(sig2)/len(sig2_data[var]))
    labels[3] += "({0:.0%})".format(len(sig3)/len(sig3_data[var]))
    labels[4] += "({0:.0%})".format(len(sig4)/len(sig4_data[var]))
    d = [bkg, sig1, sig2, sig3, sig4]
    #plot_single_variable(d,w,labels, var, logy=True) 
    plot_ratio(d,w,labels, var, logy=True) 


def score_cut_mT_plot():
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_1.hdf5","r") as f:
    bkg_data1 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_2.hdf5","r") as f:
    bkg_data2 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_3.hdf5","r") as f:
    bkg_data3 = f.get('data')[:]

  bkg_data = np.concatenate((bkg_data1,bkg_data2,bkg_data3))
  
  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg_loss = bkg_data["score"]
  bkg20 = np.percentile(bkg_loss,90)
  
  with h5py.File("../v8.1/v8p1_PFNv2_515495.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_515498.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_515518.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  sig1_loss = sig1_data["score"]
  sig2_loss = sig2_data["score"]
  sig3_loss = sig3_data["score"]
  sig4_loss = sig4_data["score"]
  
  print(variables)
  
  w0 = 5*bkg_data["weight"][bkg_loss>bkg20] 
  w1 = sig1_data["weight"][sig1_loss>bkg20] 
  w2 = sig2_data["weight"][sig2_loss>bkg20] 
  w3 = sig3_data["weight"][sig3_loss>bkg20] 
  w4 = sig4_data["weight"][sig4_loss>bkg20] 
  
  w = [w0,w1,w2,w3,w4]
  
  labels = ["QCD", "1500 GeV,0.2", "1500 GeV,0.8", "4000 GeV,0.2", "4000 GeV,0.8"]
  for var in ["mT_jj"]:
    if (var=="weight" or var=="mcEventWeight"): continue
    bkg = bkg_data[var][bkg_loss>bkg20]
    sig1 = sig1_data[var][sig1_loss>bkg20] 
    sig2 = sig2_data[var][sig2_loss>bkg20] 
    sig3 = sig3_data[var][sig3_loss>bkg20] 
    sig4 = sig4_data[var][sig4_loss>bkg20] 
    labels[0] += "({0:.0%})".format(len(bkg)/len(bkg_data[var]))
    labels[1] += "({0:.0%})".format(len(sig1)/len(sig1_data[var]))
    labels[2] += "({0:.0%})".format(len(sig2)/len(sig2_data[var]))
    labels[3] += "({0:.0%})".format(len(sig3)/len(sig3_data[var]))
    labels[4] += "({0:.0%})".format(len(sig4)/len(sig4_data[var]))
    d = [bkg, sig1, sig2, sig3, sig4]
    #plot_single_variable(d,w,labels, var, logy=True) 
    plot_ratio(d,w,labels, var, logy=True) 


def grid_scan():
  #with h5py.File("../v8.1/v8p1_PFNv1_QCDskim.hdf5","r") as f:
  #  bkg_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_1.hdf5","r") as f:
    bkg_data1 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_2.hdf5","r") as f:
    bkg_data2 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_3.hdf5","r") as f:
    bkg_data3 = f.get('data')[:]

  #bkg_data = np.concatenate((bkg_data1,bkg_data2))
  bkg_data = np.concatenate((bkg_data1,bkg_data2, bkg_data3))

  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg_loss = bkg_data["score"]
  bkg_weights = np.reshape(bkg_data["weight"],len(bkg_data["weight"]))
  print("bkg events", len(bkg_loss))
  bkg20 = np.percentile(bkg_loss, 80)
  
  sic_values = {}
  
  dsids = range(515507,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    print()
    try:
      with h5py.File("../v8.1/v1_hdf5/v8p1_PFN_"+str(dsid)+".hdf5","r") as f:
        sig1_data = f.get('data')[:]
      sig1_loss = sig1_data["score"]
      bkg_idx = get_weighted_elements_h5(bkg_weights,len(sig1_loss))
      #bkg1_loss = bkg_loss[:len(sig1_loss)]
      bkg1_loss = bkg_loss[bkg_idx]
      #plot_single_variable([bkg1_loss,sig1_loss],[np.ones(len(bkg1_loss)),np.ones(len(sig1_loss))],["bkg","sig"], "score"+str(dsid), logy=True) 
      sic_vals = do_roc(bkg1_loss, sig1_loss, str(dsid), False)
      sic_values[dsid] = sic_vals
    except Exception as e:
      print(e)
    #sig1_cut = sig1_loss[sig1_loss>bkg20]
    #cut = len(sig1_cut)/total
    #print(dsid, f'{cut:.0%}') 
  
  print("bkg events: ", len(bkg_loss))
  
  do_grid_plots(sic_values)

def grid_s_sqrt_b(bkg_percent):
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_1.hdf5","r") as f:
    bkg_data1 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_2.hdf5","r") as f:
    bkg_data2 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_3.hdf5","r") as f:
    bkg_data3 = f.get('data')[:]

  print(type(bkg_data1))  
  #bkg_data = np.concatenate((bkg_data1,bkg_data2))
  bkg_data = np.concatenate((bkg_data1,bkg_data2,bkg_data3))
  
  bkg_loss = bkg_data["score"]
  score_cut = np.percentile(bkg_loss,100.0-bkg_percent)
  print(score_cut)
  selection0 = (bkg_data["rT"] <= 0.25) & (bkg_data["rT"] > 0.15) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deta_12"] < 1.5)
  bkg_mT = bkg_data["mT_jj"][selection0]
  bkg_weight = bkg_data["weight"][selection0]
  #bkg_mT = bkg_data["mT_jj"][bkg_loss>score_cut]
  #bkg_weight = bkg_data["weight"][bkg_loss>score_cut]
  bkg_weight = 100*bkg_weight

  bins=np.linspace(1500,6500,5)
  y0,_,_ =plt.hist(bkg_mT, bins=bins, density=False, histtype='step', weights=bkg_weight)
  y0_total = np.sum(y0)

  sb_values = {}
  
  dsids = range(515495,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    try:
      with h5py.File("../v8.1/v8p1_PFNv2_"+str(dsid)+".hdf5","r") as f:
        sig1_data = f.get('data')[:]
      sig1_loss = sig1_data["score"]
      selection1 = (sig1_data["rT"] <= 0.25) & (sig1_data["rT"] > 0.15) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deta_12"] < 1.5)
      sig1_mT = sig1_data["mT_jj"][selection1]
      sig1_weight = sig1_data["weight"][selection1]
      #sig1_mT = sig1_data["mT_jj"][sig1_loss>score_cut]
      #sig1_weight = sig1_data["weight"][sig1_loss>score_cut]
      y,_,_ =plt.hist(sig1_mT, bins=bins, density=False, histtype='step', weights=sig1_weight)
      y_total = np.sum(y)

      sb_values[dsid] = {"s_sqrtb_Inclusive": my_metric(y_total,y0_total), "s_sqrtb_Max": max(my_metric(y,y0))}
    except Exception as e:
      print(e)

  do_grid_plots(sb_values)

def main():
  #mT_shape_compare()
  #grid_scan()
  #grid_s_sqrt_b(1)
  cms_mT_plots()
  #score_cut_mT_plot()

if __name__ == '__main__':
  main()

