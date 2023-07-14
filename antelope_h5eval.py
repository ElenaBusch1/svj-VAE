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
#from models_archive import *
from eval_helper import *
import matplotlib.pyplot as plt
import h5py

def my_metric(s,b):
    return np.sqrt(2*((s+b)*np.log(1+s/b)-s))

def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

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
  with h5py.File("../v8.1/v8p1_CMS_QCDskim1.hdf5","r") as f:
    bkg_data = f.get('data')[:]
   
  with h5py.File("../v8.1/v8p1_CMSskim1_515503.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_CMSskim1_515506.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_CMSskim1_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_CMSskim1_515518.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  selection0 = (bkg_data["rT"] > 0.25) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deltaY_12"] < 1.5)
  selection1 = (sig1_data["rT"] > 0.25) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deltaY_12"] < 1.5)
  selection2 = (sig2_data["rT"] > 0.25) & (sig2_data["dphi_min"] < 0.8) & (sig2_data["deltaY_12"] < 1.5)
  selection3 = (sig3_data["rT"] > 0.25) & (sig3_data["dphi_min"] < 0.8) & (sig3_data["deltaY_12"] < 1.5)
  selection4 = (sig4_data["rT"] > 0.25) & (sig4_data["dphi_min"] < 0.8) & (sig4_data["deltaY_12"] < 1.5)
   
  w0 = 50*bkg_data["weight"][selection0] 
  w1 = sig1_data["weight"][selection1] 
  w2 = sig2_data["weight"][selection2] 
  w3 = sig3_data["weight"][selection3] 
  w4 = sig4_data["weight"][selection4] 
  
  w = [w0,w1,w2,w3,w4]
  
  labels = ["QCD", "2500 GeV,0.2", "2500 GeV,0.8", "4000 GeV,0.2", "2500 GeV,0.8"]
  for var in ["mT_jj"]:
    if (var=="weight" or var=="mcEventWeight"): continue
    bkg = bkg_data[var][selection0]
    sig1 = sig1_data[var][selection1] 
    sig2 = sig2_data[var][selection2] 
    sig3 = sig3_data[var][selection3] 
    sig4 = sig4_data[var][selection4] 
    labels[0] += " ({0:.1e}, {1:.1e})".format(len(bkg),np.sum(w0))
    labels[1] += " ({0:.1e}, {1:.1e})".format(len(sig1),np.sum(w1))
    labels[2] += " ({0:.1e}, {1:.1e})".format(len(sig2),np.sum(w2))
    labels[3] += " ({0:.1e}, {1:.1e})".format(len(sig3),np.sum(w3))
    labels[4] += " ({0:.1e}, {1:.1e})".format(len(sig4),np.sum(w4))
    d = [bkg, sig1, sig2, sig3, sig4]
    #plot_single_variable(d,w,labels, var, logy=True) 
    plot_ratio(d,w,labels, var, logy=True) 

def score_cut_mT_plot():
  with h5py.File("../v8.1/v8p1_PFNv3_QCDskim3.hdf5","r") as f:
    bkg_data = f.get('data')[:]

  bkg20 = 0.92
  bkg_loss = bkg_data["score"]
  
  with h5py.File("../v8.1/v8p1_PFNv3_515503.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv3_515506.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv3_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv3_515518.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  sig1_loss = sig1_data["score"]
  sig2_loss = sig2_data["score"]
  sig3_loss = sig3_data["score"]
  sig4_loss = sig4_data["score"]
   
  w0 = 5*bkg_data["weight"][bkg_loss>bkg20] 
  w1 = sig1_data["weight"][sig1_loss>bkg20] 
  w2 = sig2_data["weight"][sig2_loss>bkg20] 
  w3 = sig3_data["weight"][sig3_loss>bkg20] 
  w4 = sig4_data["weight"][sig4_loss>bkg20] 
  
  w = [w0,w1,w2,w3,w4]
  
  labels = ["QCD", "2500 GeV,0.2", "2500 GeV,0.8", "4000 GeV,0.2", "4000 GeV,0.8"]
  for var in ["mT_jj"]:
    if (var=="weight" or var=="mcEventWeight"): continue
    bkg = bkg_data[var][bkg_loss>bkg20]
    sig1 = sig1_data[var][sig1_loss>bkg20] 
    sig2 = sig2_data[var][sig2_loss>bkg20] 
    sig3 = sig3_data[var][sig3_loss>bkg20] 
    sig4 = sig4_data[var][sig4_loss>bkg20] 
    labels[0] += " ({0:.1e}, {1:.1e})".format(len(bkg),np.sum(w0))
    labels[1] += " ({0:.1e}, {1:.1e})".format(len(sig1),np.sum(w1))
    labels[2] += " ({0:.1e}, {1:.1e})".format(len(sig2),np.sum(w2))
    labels[3] += " ({0:.1e}, {1:.1e})".format(len(sig3),np.sum(w3))
    labels[4] += " ({0:.1e}, {1:.1e})".format(len(sig4),np.sum(w4))
    d = [bkg, sig1, sig2, sig3, sig4]
    #plot_single_variable(d,w,labels, var, logy=True) 
    plot_ratio(d,w,labels, var, logy=True) 


def grid_scan(title):
  #with h5py.File("../v8.1/v8p1_PFNv1_QCDskim.hdf5","r") as f:
  #  bkg_data = f.get('data')[:]
  dir_all='/nevis/katya01/data/users/kpark/svj-vae/results/07_12_23_08_47/' # change
  h5dir=dir_all+'h5dir/'
  bkgpath=h5dir+"v8p1_QCDskim.hdf5"
  with h5py.File(bkgpath,"r") as f:
    bkg_data1 = f.get('data')[:]
#  with h5py.File("../v8.1/v8p1_PFNv3_QCDskim3_2.hdf5","r") as f:
#    bkg_data2 = f.get('data')[:]
  #with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_3.hdf5","r") as f:
  #  bkg_data3 = f.get('data')[:]
  bkg_data=bkg_data1
  #bkg_data = np.concatenate((bkg_data1,bkg_data2))
  #bkg_data = np.concatenate((bkg_data1,bkg_data2, bkg_data3))

  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg_loss = bkg_data["score"]
  bkg_weights = np.reshape(bkg_data["weight"],len(bkg_data["weight"]))
  print("bkg events", len(bkg_loss))
  
  sic_values = {}
  
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    sigpath=h5dir+"v8p1_"+str(dsid)+".hdf5"
    print()
    try:
      with h5py.File(sigpath,"r") as f:
      #with h5py.File("../v8.1/v8p1_PFNv3_"+str(dsid)+".hdf5","r") as f:
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
  cprint(f'{sic_values=}', 'green') 
  do_grid_plots(sic_values, title)

def grid_s_sqrt_b(score_cut, bkg_file, bkg_scale, sig_file_prefix, title, cms=False):
  with h5py.File("../v8.1/"+bkg_file,"r") as f:
    bkg_data = f.get('data')[:]
  
  ## CMS selections
  if cms:
    selection0 = (bkg_data["rT"] > 0.25) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deltaY_12"] < 1.5)
    bkg_mT = bkg_data["mT_jj"][selection0]
    bkg_weight = bkg_data["weight"][selection0]
    bkg_weight = bkg_scale*bkg_weight

  ## ML selection
  else:
    bkg_loss = bkg_data["score"]
    bkg_mT = bkg_data["mT_jj"][bkg_loss>score_cut]
    bkg_weight = bkg_data["weight"][bkg_loss>score_cut]
    bkg_weight = bkg_scale*bkg_weight

  y0_total = np.sum(bkg_weight)
  sb_values = {}
  
  with open("dsid_masses.json", "r") as f:
    dsid_mass = json.load(f)
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    try:
      with h5py.File("../v8.1/"+sig_file_prefix+str(dsid)+".hdf5","r") as f:
        sig1_data = f.get('data')[:]

      ## CMS selections
      if cms:
        selection1 = (sig1_data["rT"] > 0.25) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deltaY_12"] < 1.5)
        sig1_mT = sig1_data["mT_jj"][selection1]
        sig1_weight = sig1_data["weight"][selection1]

      ## ML selection
      else:
        sig1_loss = sig1_data["score"]
        sig1_weight = sig1_data["weight"][sig1_loss>score_cut]
        sig1_mT = sig1_data["mT_jj"][sig1_loss>score_cut]

      y_total = np.sum(sig1_weight) #inclusive total

      sig1_mass_window = (sig1_mT < 6500) & (sig1_mT > 1500)
      bkg_mass_window = (bkg_mT < 6500) & (bkg_mT > 1500)
      sig1_restricted_mT = sig1_mT[sig1_mass_window]
      bkg_restricted_mT = bkg_mT[bkg_mass_window]
      sig1_restricted_weight = sig1_weight[sig1_mass_window]
      bkg_restricted_weight = bkg_weight[bkg_mass_window]

      sig1_mass = dsid_mass[str(dsid)]
      sig_perc_below = np.sum(sig1_restricted_weight[sig1_restricted_mT<sig1_mass])/np.sum(sig1_restricted_weight)
      if sig_perc_below > 0.6: 
        sig1_mT_low_cut = weighted_percentile(sig1_restricted_mT[sig1_restricted_mT<sig1_mass], sig1_restricted_weight[sig1_restricted_mT<sig1_mass],sig_perc_below-0.6)
        print("mass window: ", sig1_mT_low_cut, " - ", sig1_mass)
        sig1_mT_window = (sig1_restricted_mT>sig1_mT_low_cut) & (sig1_restricted_mT<sig1_mass)
        bkg_mT_window = (bkg_restricted_mT>sig1_mT_low_cut) & (bkg_restricted_mT<sig1_mass)
        y_mT = np.sum(sig1_restricted_weight[sig1_mT_window])
        y0_mT = np.sum(bkg_restricted_weight[bkg_mT_window])
        sb_values[dsid] = {"sensitivity_Inclusive": my_metric(y_total,y0_total), "sensitivity_mT": my_metric(y_mT,y0_mT)}
      else:
        print("Cannot evaluate masspoint", sig1_mass)
        sb_values[dsid] = {"sensitivity_Inclusive": my_metric(y_total,y0_total), "sensitivity_mT": 0}

    except Exception as e:
      print(e)
      sb_values[dsid] = {"sensitivity_Inclusive": 0, "sensitivity_mT": 0}

  do_grid_plots(sb_values,title)
  return sb_values

def compare_s_sqrt_b():
  v2Inclusive = grid_s_sqrt_b(0.99, "v8p1_PFNv2_QCDskim1.hdf5", 50, "v8p1_PFNv2_", "PFN_NoPresel", False)
  v3MET = grid_s_sqrt_b(0.92, "v8p1_PFNv3_QCDskim3.hdf5", 5, "v8p1_PFNv3_", "PFN_METPresel", False)
  cms = grid_s_sqrt_b(0, "v8p1_CMS_QCDskim1.hdf5", 50, "v8p1_CMSskim1_", "CMS", cms=True)
  v2_compare = {}
  v3_compare = {}
  for dsid in v2Inclusive.keys():
    v2incl = v2Inclusive[dsid]["sensitivity_Inclusive"]
    v3incl = v3MET[dsid]["sensitivity_Inclusive"]
    cmsincl = cms[dsid]["sensitivity_Inclusive"]
    v2mT = v2Inclusive[dsid]["sensitivity_mT"]
    v3mT = v3MET[dsid]["sensitivity_mT"]
    cmsmT = cms[dsid]["sensitivity_mT"]
    if (cmsmT != 0):
      v2_compare[dsid] = {"sensitivity_Inclusive": v2incl/cmsincl, "sensitivity_mT": v2mT/cmsmT, "mT_over_Incl": v2mT/v2incl}
      if (v3incl != 0):
        v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": v3mT/cmsmT, "mT_over_Incl": v3mT/v3incl}
      else:
        v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": v3mT/cmsmT, "mT_over_Incl": 0}
    else:
      v2_compare[dsid] = {"sensitivity_Inclusive": v2incl/cmsincl, "sensitivity_mT": 0, "mT_over_Incl": v2mT/v2incl}
      if (v3incl != 0):
        v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": 0, "mT_over_Incl": v3mT/v3incl}
      else:
        v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": 0, "mT_over_Incl": 0}
  do_grid_plots(v2_compare, "v2_compare")  
  do_grid_plots(v3_compare, "v3_compare")  

def main():
  #mT_shape_compare()
  #grid_scan("METPresel")
  #compare_s_sqrt_b()
  #grid_s_sqrt_b(0.99)
  cms_mT_plots()
  #score_cut_mT_plot()

if __name__ == '__main__':
  main()


