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

def zero_div(n,d):
  return n/d if d else 0

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

def correlation_plots():
  with h5py.File("../v8.1/v8p1_PFNv6_allSignal.hdf5","r") as f:
    qcd = f.get('data')[:]
  selection = qcd["score"]>-1
  score = qcd["score"][selection]
  mT_jj = qcd["mT_jj"][selection]
  jet2_Width = qcd["jet2_Width"][selection]

  bin_dict = {"mT_jj": np.arange(1000,6000,100), "score": np.arange(0,1.0,0.02), "jet2_Width": np.arange(0,0.3, 0.006)}

  correlation_plot(score, mT_jj, "score", "mT_jj", bin_dict, "All Signals")
  correlation_plot(score, jet2_Width, "score", "jet2_Width", bin_dict, "All Signals")
  correlation_plot(mT_jj, jet2_Width, "mT_jj", "jet2_Width", bin_dict, "All Signals")

def get_sig_contamination():
  with h5py.File("../v9.1/v9p1_PFNv6_totalBkgALL_skim0.hdf5","r") as f:
    bkg_data = f.get('data')[:]
  bkg_loss = bkg_data["score"]
  w0 = 5*bkg_data["weight"]
  bkg_jet2 = bkg_data["jet2_Width"]
  bkg_SR = (bkg_jet2 > 0.05) & (bkg_loss>0.6)
  bkg_CR = (bkg_jet2 <= 0.05) & (bkg_loss>0.0)
  bkg_VR = (bkg_jet2 > 0.05) & (bkg_loss<0.6)
  bkg_CR_count = sum(w0[bkg_CR])
  bkg_VR_count = sum(w0[bkg_VR])
  bkg_SR_count = sum(w0[bkg_SR])
  
  dsids = range(515487,515527)
  sig_eff = {}
  for dsid in dsids:
    with h5py.File("../v8.1/v8p1_PFNv6_"+str(dsid)+".hdf5","r") as f:
      sig_data = f.get('data')[:]
    sig_loss = sig_data["score"]
    w = sig_data["weight"]
    sig_jet2 = sig_data["jet2_Width"]
    sig_SR = (sig_jet2 > 0.05) & (sig_loss>0.6)
    sig_CR = (sig_jet2 <= 0.05) & (sig_loss>0.0)
    sig_VR = (sig_jet2 > 0.05) & (sig_loss<0.6)
    sig_CR_count = sum(w[sig_CR])
    sig_VR_count = sum(w[sig_VR])
    sig_SR_count = sum(w[sig_SR])
    sig_eff[dsid] = {"Signal Contamination (%)":100*sig_CR_count/bkg_CR_count, "Signal Effiency in SR":sig_SR_count/sum(w), "Signal Efficiency in CR": sig_CR_count/sum(w)}

  do_grid_plots(sig_eff, "CR Study")  

def test_transform():
  with h5py.File("../kp/v8p1_vANTELOPE_relu_QCDskim.hdf5","r") as f:
    qcd = f.get('data')[:]
  with h5py.File("../kp/v8p1_vANTELOPE_relu_515506.hdf5","r") as f:
    sig1 = f.get('data')[:]

  selection1 = sig1["jet1_pt"] > 0.0
  sig1_loss = sig1["mse"][selection1]

  selection0 = qcd["jet1_pt"] > 0.0
  bkg_loss = qcd["mse"][selection0]
  bkg_weights = np.reshape(qcd["weight"][selection0],len(qcd["weight"][selection0]))
  bkg_idx = get_weighted_elements_h5(np.abs(bkg_weights),len(sig1_loss))
  bkg1_loss = bkg_loss[bkg_idx]
  sic_vals = do_roc(bkg1_loss, sig1_loss, "515506", True)

def mT_shape_compare():
  with h5py.File("../v9.1/v9p1_PFNv6_totalBkgALL_skim0.hdf5","r") as f:
    qcd = f.get('data')[:]
  #with h5py.File("../v9.1/v9p1_PFNv4_Znunuskim0_w300k.hdf5","r") as f:
  #  znunu = f.get('data')[:]
  #with h5py.File("../v9.1/v9p1_PFNv4_Wjetsskim0_w300k.hdf5","r") as f:
  #  wjets = f.get('data')[:]
  with h5py.File("../v9.2/v9p2_PFNv6_dataAll.hdf5","r") as f:
    data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv6_515503.hdf5","r") as f:
    sig1 = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv6_515506.hdf5","r") as f:
    sig2 = f.get('data')[:]

  #variables = [f_name for (f_name,f_type) in qcd.dtype.descr]
  variables = ["mT_jj"]
  for var in variables:
    #if var == "mT_jj": continue
    selectionDT = data["jet2_Width"]<0.05
    selectionCR = qcd["jet2_Width"]<0.05
    #selectionVR = (qcd["jet2_Width"]>0.05) & (qcd["score"]<0.6)
    #selectionSR = (qcd["jet2_Width"]>0.05) & (qcd["score"]>0.6)
    data_loss = data[var][selectionDT]
    qcd_lossCR = qcd[var][selectionCR]
    #qcd_lossVR = qcd[var][selectionVR]
    #qcd_lossSR = qcd[var][selectionSR]
    #print("bkg shape: ", qcd_loss.shape)
    #znunu_loss = znunu[var]
    #wjets_loss = wjets[var]
    #data_loss = data[var]
    #total_bkg = np.concatenate((qcd_loss, znunu_loss, wjets_loss))
    selection1 = sig1["score"]>0.6
    selection2 = sig2["score"]>0.6
    sig1_loss = sig1[var][selection1]
    sig2_loss = sig2[var][selection2]
    qcd_weightsCR = np.reshape(5*qcd["weight"][selectionCR],len(qcd_lossCR))
    #qcd_weightsVR = np.reshape(5*qcd["weight"][selectionVR],len(qcd_lossVR))
    #qcd_weightsSR = np.reshape(5*qcd["weight"][selectionSR],len(qcd_lossSR))
    #print("CR:",sum(qcd_weightsCR))
    #print("VR:",sum(qcd_weightsVR))
    #print("SR:",sum(qcd_weightsSR))
    #znunu_weights = 9.17*np.reshape(znunu["weight"],len(znunu["weight"]))
    #wjets_weights = 35.97*np.reshape(wjets["weight"],len(wjets["weight"]))
    #total_weights = np.concatenate((qcd_weights,znunu_weights,wjets_weights))
    #d = [qcd_lossCR, qcd_lossVR, qcd_lossSR]#, sig1_loss, sig2_loss]
    #w = [qcd_weightsCR, qcd_weightsVR, qcd_weightsSR]#np.ones(len(sig1_loss)), np.ones(len(sig2_loss))]
    #labels = ["MC - CR", "MC - VR", "MC - SR"]#, "2500 GeV 0.8"]
    d = [data_loss, qcd_lossCR]
    w = [np.ones(len(data_loss)), qcd_weightsCR]
    labels = ["Data - CR", "MC - CR"]#"2000 GeV 0.2", "2000 GeV 0.8"]
    #labels = [l+str(len(ds)) for l,ds in zip(lab,d)]
    #plot_single_variable(d,w,labels, var, logy=True) 
    plot_simple_ratio(d,w,labels, var, logy=True) 
    

def cms_mT_plots():
  with h5py.File("../v9.1/v9p1_CMS_totalBkgALL_skim0.hdf5","r") as f:
    bkg_data = f.get('data')[:]
   
  with h5py.File("../v8.1/v8p1_CMSskim1_515499.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_CMSskim1_515507.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_CMSskim1_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_CMSskim1_515519.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  selection0 = (bkg_data["rT"] > 0.25) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deltaY_12"] < 1.5) 
  selection1 = (sig1_data["rT"] > 0.25) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deltaY_12"] < 1.5)
  selection2 = (sig2_data["rT"] > 0.25) & (sig2_data["dphi_min"] < 0.8) & (sig2_data["deltaY_12"] < 1.5)
  selection3 = (sig3_data["rT"] > 0.25) & (sig3_data["dphi_min"] < 0.8) & (sig3_data["deltaY_12"] < 1.5)
  #selection4 = (sig4_data["rT"] > 0.25) & (sig4_data["dphi_min"] < 0.8) & (sig4_data["deltaY_12"] < 1.5)
   
  w0 = 0.82*5*bkg_data["weight"][selection0] 
  w1 = sig1_data["weight"][selection1] 
  w2 = sig2_data["weight"][selection2] 
  w3 = sig3_data["weight"][selection3] 
  print(w1)
  #w4 = sig4_data["weight"][selection4] 
  
  w = [w0,w1,w2,w3]#,w4]
  
  labels = ["QCD", "2000 GeV,0.2", "3000 GeV,0.2", "4000 GeV,0.2"]#, "5000 GeV,0.2"]
  for var in ["mT_jj"]:
    if (var=="weight" or var=="mcEventWeight"): continue
    bkg = bkg_data[var][selection0]
    sig1 = sig1_data[var][selection1] 
    sig2 = sig2_data[var][selection2] 
    sig3 = sig3_data[var][selection3] 
    #sig4 = sig4_data[var][selection4] 
    labels[0] += " ({0:.1e}, {1:.1e})".format(len(bkg),np.sum(w0))
    labels[1] += " ({0:.1e}, {1:.1e})".format(len(sig1),np.sum(w1))
    labels[2] += " ({0:.1e}, {1:.1e})".format(len(sig2),np.sum(w2))
    labels[3] += " ({0:.1e}, {1:.1e})".format(len(sig3),np.sum(w3))
    #labels[4] += " ({0:.1e}, {1:.1e})".format(len(sig4),np.sum(w4))
    d = [bkg, sig1, sig2, sig3]#, sig4]
    plot_single_variable(d,w,labels, var, logy=True) 
    #plot_ratio(d,w,labels, var, logy=True) 

def score_cut_mT_plot():
  with h5py.File("../v8.1/v8p1_PFNv6_allSignal.hdf5","r") as f:
    bkg_data = f.get('data')[:]

  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  bkg20 = 0.6
  bkg_loss = bkg_data["score"]
  
  with h5py.File("../v8.1/v8p1_PFNv6_515503.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv6_515505.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv6_515516.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv6_515518.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  sig1_loss = sig1_data["score"]
  sig2_loss = sig2_data["score"]
  sig3_loss = sig3_data["score"]
  sig4_loss = sig4_data["score"]
   
  #w0 = 5*bkg_data["weight"][bkg_loss>bkg20] 
  w0 = bkg_data["weight"]#[bkg_loss>bkg20]
  w1 = sig1_data["weight"][sig1_loss>bkg20] 
  w2 = sig2_data["weight"][sig2_loss>bkg20] 
  w3 = sig3_data["weight"][sig3_loss>bkg20] 
  w4 = sig4_data["weight"][sig4_loss>bkg20] 
  
  w = [w0,w1,w2,w3,w4]
  
  #print(sum(w0[bkg_loss>bkg20]))
  bkg_jet2 = bkg_data["jet2_Width"]
  jet2_SR = (bkg_jet2 > 0.05) & (bkg_loss>0.6)
  jet2_CR = (bkg_jet2 < 0.05) & (bkg_loss>0.0)
  jet2_VR = (bkg_jet2 > 0.05) & (bkg_loss<0.6)
  print("SR", sum(w0[jet2_SR]))
  print("CR", sum(w0[jet2_CR]))
  print("VR", sum(w0[jet2_VR]))
  quit()

  #for var in variables: #["mT_jj", "deta_12", ""]:
  for var in ["jet2_Width"]:
    labels = ["Background", "2500 GeV,0.2", "2500 GeV,0.6", "4000 GeV,0.4", "4000 GeV,0.8"]
    if (var=="weight" or var=="mcEventWeight"): continue
    bkg = bkg_data[var][bkg_loss>bkg20]
    sig1 = sig1_data[var][sig1_loss>bkg20] 
    sig2 = sig2_data[var][sig2_loss>bkg20] 
    sig3 = sig3_data[var][sig3_loss>bkg20] 
    sig4 = sig4_data[var][sig4_loss>bkg20] 
    #labels[0] += " ({0:.1e}, {1:.1e})".format(len(bkg),np.sum(w0))
    #labels[1] += " ({0:.1e}, {1:.1e})".format(len(sig1),np.sum(w1))
    #labels[2] += " ({0:.1e}, {1:.1e})".format(len(sig2),np.sum(w2))
    #labels[3] += " ({0:.1e}, {1:.1e})".format(len(sig3),np.sum(w3))
    #labels[4] += " ({0:.1e}, {1:.1e})".format(len(sig4),np.sum(w4))
    d = [bkg, sig1, sig2, sig3, sig4]
    #plot_single_variable(d,w,labels, var, logy=True) 
    plot_ratio(d,w,labels, var, logy=True,cumsum=True) 

def grid_scan(title):
  with h5py.File("../v9.1/v9p1_PFNv6_1_totalBkgALL_skim0.hdf5","r") as f:
    bkg_data = f.get('data')[:]

  variables = [f_name for (f_name,f_type) in bkg_data.dtype.descr]
  selection0 = bkg_data["jet2_Width"] > 0.0
  bkg_loss = bkg_data["score"][selection0]
  bkg_weights = np.reshape(bkg_data["weight"][selection0],len(bkg_data["weight"][selection0]))
  print("bkg events", len(bkg_loss))
  
  sic_values = {}
  
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    print()
    try:
      with h5py.File("../v8.1/v8p1_PFNv6_1_"+str(dsid)+"_skim1.hdf5","r") as f:
        sig1_data = f.get('data')[:]
      selection1 = sig1_data["jet2_Width"] > 0.0
      sig1_loss = sig1_data["score"][selection1]
      bkg_idx = get_weighted_elements_h5(np.abs(bkg_weights),len(sig1_loss))
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
  
  do_grid_plots(sic_values, title)

def grid_s_sqrt_b(score_cut, bkg_file, bkg_scale, sig_file_prefix, title, cms=False):
  with h5py.File("../v9.1/"+bkg_file,"r") as f:
    bkg_data = f.get('data')[:]
  
  ## CMS selections
  if cms:
    selection0 = (bkg_data["rT"] > 0.25) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deltaY_12"] < 1.5)
    bkg_mT = bkg_data["mT_jj"][selection0]
    bkg_weight = bkg_data["weight"][selection0]
    bkg_weight = bkg_scale*bkg_weight

  ## ML selection
  else:
    selection1 = (bkg_data["score"] > score_cut) #& (bkg_data["jet2_Width"] > 0.09)
    if (title == "PFN_SR"): selection1 = (bkg_data["score"] > score_cut) & (bkg_data["jet2_Width"] > 0.05)
    bkg_mT = bkg_data["mT_jj"][selection1]
    bkg_weight = bkg_data["weight"][selection1]
    bkg_weight = bkg_scale*bkg_weight

  y0_total = np.sum(bkg_weight)
  sb_values = {}
  
  with open("dsid_masses.json", "r") as f:
    dsid_mass = json.load(f)
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    try:
      with h5py.File("../v8.1/"+sig_file_prefix+str(dsid)+"_"+title+".hdf5","r") as f:
        sig1_data = f.get('data')[:]

      ## CMS selections
      if cms:
        selection1 = (sig1_data["rT"] > 0.25) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deltaY_12"] < 1.5) 
        sig1_mT = sig1_data["mT_jj"][selection1]
        sig1_weight = sig1_data["weight"][selection1]

      ## ML selection
      else:
        selection1 = (sig1_data["score"] > score_cut)# & (sig1_data["jet2_Width"] > 0.07)
        if (title == "PFN_SR"): selection1 = (sig1_data["score"] > score_cut) & (sig1_data["jet2_Width"] > 0.05)
        sig1_weight = sig1_data["weight"][selection1]
        sig1_mT = sig1_data["mT_jj"][selection1]

      y_total = np.sum(sig1_weight) #inclusive total

      sig1_mass_window = (sig1_mT < 6500) & (sig1_mT > 1000)
      bkg_mass_window = (bkg_mT < 6500) & (bkg_mT > 1000)
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
  #v2Inclusive = grid_s_sqrt_b(0.92, "v8p1_PFNv3_QCDskim3.hdf5", 5, "v8p1_PFNv3_", "PFN_PreBugFix", False)
  v3MET = grid_s_sqrt_b(0.6, "v9p1_PFNv6_1_totalBkgALL_skim0.hdf5", 5, "v8p1_PFNv6_1_", "skim0", False)
  cms = grid_s_sqrt_b(0.6, "v9p1_PFNv6_1_totalBkgALL_skim0.hdf5", 5, "v8p1_PFNv6_1_", "skim1", False)
  #cms = grid_s_sqrt_b(0.6, "v9p1_CMS_totalBkgALL_skim0.hdf5", 5, "v8p1_CMSskim1_", "CMS", cms=True)
  v2_compare = {}
  v3_compare = {}
  for dsid in v3MET.keys():
    #v2incl = v2Inclusive[dsid]["sensitivity_Inclusive"]
    v3incl = v3MET[dsid]["sensitivity_Inclusive"]
    cmsincl = cms[dsid]["sensitivity_Inclusive"]
    #v2mT = v2Inclusive[dsid]["sensitivity_mT"]
    v3mT = v3MET[dsid]["sensitivity_mT"]
    cmsmT = cms[dsid]["sensitivity_mT"]
    v3_compare[dsid] = {"sensitivity_Inclusive": zero_div(v3incl,cmsincl), "sensitivity_mT": zero_div(v3mT,cmsmT), "mT_over_Incl": zero_div(v3mT,v3incl)}
    #if (cmsmT != 0):
    #  #v2_compare[dsid] = {"sensitivity_Inclusive": v2incl/cmsincl, "sensitivity_mT": v2mT/cmsmT, "mT_over_Incl": v2mT/v2incl}
    #  if (v3incl != 0):
    #    v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": v3mT/cmsmT, "mT_over_Incl": v3mT/v3incl}
    #  else:
    #    v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": v3mT/cmsmT, "mT_over_Incl": 0}
    #else:
    #  #v2_compare[dsid] = {"sensitivity_Inclusive": v2incl/cmsincl, "sensitivity_mT": 0, "mT_over_Incl": v2mT/v2incl}
    #  if (v3incl != 0):
    #    v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": 0, "mT_over_Incl": v3mT/v3incl}
    #  else:
    #    v3_compare[dsid] = {"sensitivity_Inclusive": v3incl/cmsincl, "sensitivity_mT": 0, "mT_over_Incl": 0}
  #do_grid_plots(v2_compare, "v2_compare")  
  do_grid_plots(v3_compare, "k_fold")  

def main():
  mT_shape_compare()
  #grid_scan("PFNv6_1_skim1")
  #compare_s_sqrt_b()
  #correlation_plots()
  #get_sig_contamination()
  #grid_s_sqrt_b(0.99)
  #cms_mT_plots()
  #score_cut_mT_plot()
  #test_transform()

if __name__ == '__main__':
  main()

