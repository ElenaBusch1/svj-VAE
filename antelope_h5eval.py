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
import h5py

def mT_shape_compare():
  with h5py.File("../v8.1/v8p1bkg.hdf5","r") as f:
    bkg_data = f.get('qcd')[:]

  with h5py.File("../v8.1/v8p1_PFN_QCDskim.hdf5","r") as f:
    pfn_data = f.get('data')[:]

  selection0 = (bkg_data["rT"] > 0.15) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deta_12"] < 1.5)
  bkg_loss = bkg_data["score"]
  bkg20 = np.percentile(bkg_loss,99)

def cms_mT_plots():
  with h5py.File("../v8.1/v8p1bkg.hdf5","r") as f:
    bkg_data = f.get('qcd')[:]
  
  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  #bkg20 = np.percentile(bkg_loss,99.5)
  
  with h5py.File("../v8.1/v8p1_515495.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_515498.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_515518.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  selection0 = (bkg_data["rT"] <= 0.25) & (bkg_data["rT"] > 0.15) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deta_12"] < 1.5)
  selection1 = (sig1_data["rT"] <= 0.25) & (sig1_data["rT"] > 0.15) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deta_12"] < 1.5)
  selection2 = (sig2_data["rT"] <= 0.25) & (sig2_data["rT"] > 0.15) & (sig2_data["dphi_min"] < 0.8) & (sig2_data["deta_12"] < 1.5)
  selection3 = (sig3_data["rT"] <= 0.25) & (sig3_data["rT"] > 0.15) & (sig3_data["dphi_min"] < 0.8) & (sig3_data["deta_12"] < 1.5)
  selection4 = (sig4_data["rT"] <= 0.25) & (sig4_data["rT"] > 0.15) & (sig4_data["dphi_min"] < 0.8) & (sig4_data["deta_12"] < 1.5)
  
  print(variables)
  
  w0 = 100*bkg_data["weight"][selection0] 
  w1 = sig1_data["weight"][selection1] 
  w2 = sig2_data["weight"][selection2] 
  w3 = sig3_data["weight"][selection3] 
  w4 = sig4_data["weight"][selection4] 
  
  w = [w0,w1,w2,w3,w4]
  
  labels = ["QCD", "1500 GeV,0.2", "1500 GeV,0.8", "4000 GeV,0.2", "4000 GeV,0.8"]
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
    plot_single_variable(d,w,labels, var, logy=True) 
    plot_ratio(d,w,labels, var, logy=True) 


def score_cut_mT_plot():
  with h5py.File("../v8.1/v8p1_PFN_QCDskim.hdf5","r") as f:
    bkg_data = f.get('data')[:]
  
  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg_loss = bkg_data["score"]
  bkg20 = np.percentile(bkg_loss,99)
  
  with h5py.File("../v8.1/v8p1_PFN_515495.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFN_515498.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFN_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFN_515518.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  sig1_loss = sig1_data["score"]
  sig2_loss = sig2_data["score"]
  sig3_loss = sig3_data["score"]
  sig4_loss = sig4_data["score"]
  
  print(variables)
  
  w0 = 100*bkg_data["weight"][bkg_loss>bkg20] 
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
  with h5py.File("../v8.1/v8p1_PFN_QCDskim.hdf5","r") as f:
    bkg_data = f.get('data')[:]
  
  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg_loss = bkg_data["score"]
  print("bkg events", len(bkg_loss))
  bkg20 = np.percentile(bkg_loss, 80)
  
  sic_values = {}
  
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    print()
    try:
      with h5py.File("../v8.1/v8p1_PFN_"+str(dsid)+".hdf5","r") as f:
        sig1_data = f.get('data')[:]
      sig1_loss = sig1_data["score"]
      bkg1_loss = bkg_loss[:len(sig1_loss)]
      sic_vals = do_roc(bkg1_loss, sig1_loss, str(dsid), False)
      sic_values[dsid] = sic_vals
    except Exception as e:
      print(e)
    #sig1_cut = sig1_loss[sig1_loss>bkg20]
    #cut = len(sig1_cut)/total
    #print(dsid, f'{cut:.0%}') 
  
  print("bkg events: ", len(bkg_loss))
  
  do_grid_plots(sic_values)

def main():
  cms_mT_plots()

if __name__ == '__main__':
  main()

