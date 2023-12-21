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
from models_archive import *
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

def check_yield(title, all_dir, file_prefix, filename,  key):
  h5dir=all_dir+'applydir/'
  plot_dir=h5dir+'/plots/'
  if not os.path.exists(plot_dir):os.mkdir(plot_dir)
  path=h5dir+'/'+'hdf5_jet2_width'+'/'+f"{file_prefix}{filename}"
  with h5py.File(path,"r") as f:
#  with h5py.File("../v8.1/v8p1_PFNv6_allSignal.hdf5","r") as f:
    data_var = f.get('data')[:]
  text=''
  hists=[data_var['mT_jj'], data_var['mT_jj'], data_var['mT_jj'], data_var['mT_jj']]
  h_names=['(All)','(CR)', '(VR)', ' (SR)']
  weights_ls_each=[]
  if data_var['weight'].any(): # if array contains some element other than 0 
    weights_ls_each.append(data_var['weight'])
    print('weights does not already contain only zeros')
  else:#if array contains only zeros
    print(np.ones(data_var['weight'].shape).shape)
    print(data_var['weight'].shape)
    weights_ls_each.append(np.ones(data_var['weight'].shape))
    print('weights does  already contain only zeros -> made array of 1s' )

  method=key
  weights_ls=[weights_ls_each[0],weights_ls_each[0], weights_ls_each[0], weights_ls_each[0]]
  hists_cut=[[data_var['jet2_Width'], data_var[method]],[data_var['jet2_Width'], data_var[method]],[ data_var['jet2_Width'], data_var[method]],[ data_var['jet2_Width'], data_var[method]]]
  cut_ls=[[0,0],[0.05, 0], [0.05, 0.7],[0.05, 0.7]]
  cut_operator = [[True, True],[False,True], [True,  False],[True, True]]
  method_cut=[['jet2_Width', method],['jet2_Width', method], ['jet2_Width', method], ['jet2_Width', method]]  

  '''
  bool_ratio has to be False for the  
  '''
  text=plot_single_variable_ratio(hists=hists, h_names=h_names, weights_ls=weights_ls, title=title,hists_cut=hists_cut, cut_ls=cut_ls, cut_operator=cut_operator, method_cut=method_cut, bool_ratio=True, bool_plot=False, bin_min=1000, bin_max= 5000, logy=True)
  with open(all_dir+f'yield_{file_prefix}{filename.split(".hdf5")[-2]}.txt', 'w') as fw:
    fw.write(text) 

def correlation_plots(title, all_dir, file_prefix, filename,  key):
  h5dir=all_dir+'applydir/'
  plot_dir=h5dir+'/plots/'
  if not os.path.exists(plot_dir):os.mkdir(plot_dir)
  path=h5dir+'/'+'hdf5_jet2_width'+'/'+f"{file_prefix}{filename}"
  with h5py.File(path,"r") as f:
#  with h5py.File("../v8.1/v8p1_PFNv6_allSignal.hdf5","r") as f:
    qcd = f.get('data')[:]
  selection = qcd[key]>-1
  score = qcd[key][selection]
  mT_jj = qcd["mT_jj"][selection]
  jet2_Width = qcd["jet2_Width"][selection]

  bin_dict = {"mT_jj": np.arange(1000,6000,100), key: np.arange(0.62,0.96,0.01), "jet2_Width": np.arange(0,0.3, 0.006)}
  #bin_dict = {"mT_jj": np.arange(1000,6000,100), key: np.arange(0,1.0,0.02), "jet2_Width": np.arange(0,0.3, 0.006)}
  tag_file=file_prefix+filename.split('.hdf5')[-2]
  correlation_plot(score, mT_jj, key, "mT_jj", bin_dict, title, tag_file=tag_file, plot_dir=plot_dir)
  correlation_plot(score, jet2_Width, key, "jet2_Width", bin_dict, title, tag_file=tag_file, plot_dir=plot_dir)
  correlation_plot(mT_jj, jet2_Width, "mT_jj", "jet2_Width", bin_dict, title, tag_file=tag_file, plot_dir=plot_dir)

def mT_shape_compare(key):
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
  bkg_lossv1 = bkg_datav1[key]
  bkg_lossv2 = bkg_datav2[key]
  bkg1_weights = np.reshape(bkg_datav1["weight"],len(bkg_datav1["weight"]))
  bkg2_weights = np.reshape(bkg_datav2["weight"],len(bkg_datav2["weight"]))
  dsids = [515518,515522,515523,515515]
  for dsid in dsids:
    with h5py.File("../v8.1/v1_hdf5/v8p1_PFN_"+str(dsid)+".hdf5","r") as f:
      sigv1_data = f.get('data')[:]
    with h5py.File("../v8.1/v8p1_PFNv2_"+str(dsid)+".hdf5","r") as f:
      sigv2_data = f.get('data')[:]
    sigv1_loss = sigv1_data[key]
    sigv2_loss = sigv2_data[key]
    #bkgv1_idx = get_weighted_elements_h5(bkg1_weights,len(sigv1_loss))
    #bkgv2_idx = get_weighted_elements_h5(bkg2_weights,len(sigv2_loss))
    bkgv1_loss = bkg_lossv1[:len(sigv1_loss)] 
    bkgv2_loss = bkg_lossv2[:len(sigv2_loss)]
    d = [bkgv1_loss, bkgv2_loss, sigv1_loss, sigv2_loss]
    w = [np.ones(len(x)) for x in d]
    lab = ["v1 BKG (bad sel)", "v2 BKG (bad sel)", "v1 Signal", "v2 Signal"]
    labels = [l+str(len(ds)) for l,ds in zip(lab,d)]
    plot_single_variable(d,w,labels, key+str(dsid), logy=False) 
    

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

def score_cut_mT_plot(key):
  with h5py.File("../v8.1/v8p1_PFNv3_QCDskim3.hdf5","r") as f:
    bkg_data = f.get('data')[:]

  bkg20 = 0.92
  bkg_loss = bkg_data[key]
  
  with h5py.File("../v8.1/v8p1_PFNv3_515503.hdf5","r") as f:
    sig1_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv3_515506.hdf5","r") as f:
    sig2_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv3_515515.hdf5","r") as f:
    sig3_data = f.get('data')[:]
  with h5py.File("../v8.1/v8p1_PFNv3_515518.hdf5","r") as f:
    sig4_data = f.get('data')[:]
  
  sig1_loss = sig1_data[key]
  sig2_loss = sig2_data[key]
  sig3_loss = sig3_data[key]
  sig4_loss = sig4_data[key]
   
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


def grid_scan(title, outputdir, sig_prefix, bkg_prefix,bkg_file, key='multi_reco'): # or 'score'
  # if this doesn't work try changing bkgpath
  #with h5py.File("../v8.1/v8p1_PFNv1_QCDskim.hdf5","r") as f:
  #  bkg_data = f.get('data')[:]
  plot_dir=outputdir+'/plots/'
  if not os.path.exists(plot_dir):os.mkdir(plot_dir)
  bkgpath=outputdir+f"{bkg_prefix}{bkg_file}"
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
  bkg_loss = bkg_data[key]
  if bkg_data['weight'].any(): 
    bkg_data_weights=bkg_data['weight']
  else:  bkg_data_weights=np.ones(bkg_data['weight'].shape)
  bkg_weights = np.reshape(bkg_data_weights,len(bkg_data_weights))
  #bkg_weights = np.reshape(bkg_data["weight"],len(bkg_data["weight"]))
  print("bkg events", len(bkg_loss))
  print(bkg_data1['mT_jj'],bkg_data1['weight']) 
  print(bkg_data1['mT_jj'].shape,bkg_data1['weight'].shape) 
  sic_values = {}
  
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    sigpath=outputdir+f"{sig_prefix}{dsid}_log10"+".hdf5"
    print(f'{sigpath=}') 
    try:
      with h5py.File(sigpath,"r") as f:
      #with h5py.File("../v8.1/v8p1_PFNv3_"+str(dsid)+".hdf5","r") as f:
        sig1_data = f.get('data')[:]
      sig1_loss = sig1_data[key]
      bkg_idx = get_weighted_elements_h5(bkg_weights,len(sig1_loss))
      #bkg1_loss = bkg_loss[:len(sig1_loss)]
      bkg1_loss = bkg_loss[bkg_idx]
      #plot_single_variable([bkg1_loss,sig1_loss],[np.ones(len(bkg1_loss)),np.ones(len(sig1_loss))],["bkg","sig"], key+str(dsid), logy=True) 
      sic_vals = do_roc(bkg1_loss, sig1_loss, tag_file=f'{key}_'+str(dsid), tag_title=f'{key} '+str(dsid), make_transformed_plot=False,plot_dir=plot_dir )
      sic_values[dsid] = sic_vals
      cprint(f"{dsid}, sig events, {len(sig1_loss)}", )
    except Exception as e:
      cprint(e,'red')
    #sig1_cut = sig1_loss[sig1_loss>bkg20]
    #cut = len(sig1_cut)/total
    #print(dsid, f'{cut:.0%}') 
  
  print("bkg events: ", len(bkg_loss))
  print(f'grid_scan in {plot_dir}')
  do_grid_plots(sic_values, tag_title=f'{key} '+title, tag_file=f'{key}_'+title,plot_dir=plot_dir)

def grid_s_sqrt_b(score_cut,outputdir, bkg_scale, sig_prefix, bkg_prefix,bkg_file,title, cms=False, key="multi_reco"): #all_dir # bkg_scale = 5
  # if can't read the file try changing sigpath or outputdir
  plot_dir=outputdir+'/plots/'
  if not os.path.exists(plot_dir):os.mkdir(plot_dir)
  bkgpath=outputdir+f"{bkg_prefix}{bkg_file}"
  with h5py.File(bkgpath,"r") as f:
    bkg_data = f.get('data')[:]
  
  ## CMS selections
  if cms:
    selection0 = (bkg_data["rT"] > 0.25) & (bkg_data["dphi_min"] < 0.8) & (bkg_data["deltaY_12"] < 1.5)
    bkg_mT = bkg_data["mT_jj"][selection0]
    bkg_weight = bkg_data["weight"][selection0]
    bkg_weight = bkg_scale*bkg_weight

  ## ML selection
  else:
    bkg_loss = bkg_data[key]
    bkg_mT = bkg_data["mT_jj"][bkg_loss>score_cut]
    if bkg_data['weight'].any(): 
      bkg_data_weights=bkg_data['weight']
    else:  bkg_data_weights=np.ones(bkg_data['weight'].shape)

    bkg_weight = bkg_data_weights[bkg_loss>score_cut]
    #bkg_weight = bkg_data["weight"][bkg_loss>score_cut]
    bkg_weight = bkg_scale*bkg_weight

  y0_total = np.sum(bkg_weight)
  sb_values = {}
  
  with open("/nevis/katya01/data/users/ebusch/SVJ/autoencoder/svj-vae/dsid_masses.json", "r") as f:
    dsid_mass = json.load(f)
  dsids = range(515487,515527) #,515499,515502,515507,515510,515515,515518,515520,515522]
  for dsid in dsids:
    sigpath=outputdir+f"{sig_prefix}{dsid}_log10"+".hdf5"
    # sigpath="../v8.1/"+sig_prefix+str(dsid)+".hdf5"
    try:
      with h5py.File(sigpath,"r") as f:
        sig1_data = f.get('data')[:]

      ## CMS selections
      if cms:
        selection1 = (sig1_data["rT"] > 0.25) & (sig1_data["dphi_min"] < 0.8) & (sig1_data["deltaY_12"] < 1.5)
        sig1_mT = sig1_data["mT_jj"][selection1]
        sig1_weight = sig1_data["weight"][selection1]

      ## ML selection
      else:
        sig1_loss = sig1_data[key]
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
      cprint(e, 'red')
      sb_values[dsid] = {"sensitivity_Inclusive": 0, "sensitivity_mT": 0}

  print(f'grid_s_sqrt_b in {plot_dir}')
#  do_grid_plots(sb_values, title+f'_score_cut={score_cut}',plot_dir=plot_dir)
  do_grid_plots(sb_values, tag_title=f'{key} score cut = {score_cut} '+title, tag_file=f'{key}_score_cut={score_cut}'+title,plot_dir=plot_dir)
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
  #grid_scan("METPresel") -> before  8/10
  grid_scan("METPresel_PFNv3") #-> after 8/10
  #compare_s_sqrt_b()
  #grid_s_sqrt_b(0.99)
  cms_mT_plots()
  #score_cut_mT_plot()

if __name__ == '__main__':
  main()

