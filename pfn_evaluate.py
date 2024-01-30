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

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
pfn_models = ['PFNv6']
arch_dir = "architectures_saved/"
#pfn_model = 'PFNv6'
x_events = -1
myCernID = "ebusch"

## evaluate bkg
for pfn_model in pfn_models:

  ## ---------- Load graph model ----------
  graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
  graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
  graph.compile()
  
  ## Load classifier model
  classifier = keras.models.load_model(arch_dir+pfn_model+'_classifier_arch')
  classifier.load_weights(arch_dir+pfn_model+'_classifier_weights.h5')
  classifier.compile()

  ## parse input files
  with open("../v11.1/v11p1_test.txt", "r") as f:
    files = []
    for line in f:
      line = line.strip()
      files.append(line)

  ## evaluate all files
  for myFile in files:
    print("-------> Evaluating", myFile)
    dsid = myFile[myFile.find(myCernID)+len(myCernID)+1:myFile.find(".root")]
    my_variables = ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "deltaY_12", "dphi_12", "weight", "mcEventWeight"]
    try:
      bkg2,mT_bkg = getTwoJetSystem(x_events,"../v11.1/"+myFile, my_variables, False)
    except:
      continue
    scaler = load(arch_dir+pfn_model+'_scaler.bin')
    bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
    
    phi_bkg = graph.predict(bkg2)
     
    pred_phi_bkg = classifier.predict(phi_bkg)
    
    ## Classifier loss
    bkg_loss = pred_phi_bkg[:,1]
    
    my_variables.insert(0,"score")
    save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)
    #print(save_bkg)
    ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
    rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
    
    with h5py.File("v9p2_PFNv6_"+dsid+".hdf5","w") as h5f:
      dset = h5f.create_dataset("data",data=rec_bkg)
    print("Saved hdf5 for", dsid)

  # ## evaluate bkg
  # ##for chunk in range(73):
  # for dataYear in range(15,19): 
  #   ## Load testing data
  #   x_events = -1 ## -1 for all events

  #   #idx_range = range(chunk*100000, (chunk+1)*100000)
  #   #if chunk == 72: idx_range = range(chunk*100000,7299018)
  #   my_variables = ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "deltaY_12", "dphi_12", "weight", "mcEventWeight"]
  #   bkg2,mT_bkg = getTwoJetSystem(x_events,"../v9.2/user.ebusch.data"+str(dataYear)+".root", my_variables, False)
  #   #bkg2,mT_bkg = getTwoJetSystem(x_events,"../v9.1/skim0.user.ebusch.totalBkgALL.root", my_variables, False, idx_range)
  #   scaler = load(arch_dir+pfn_model+'_scaler.bin')
  #   bkg2,_ = apply_StandardScaling(bkg2,scaler,False) 
  #   phi_bkg = graph.predict(bkg2)
  #   pred_phi_bkg = classifier.predict(phi_bkg)
  #   # ## Classifier loss
  #   bkg_loss = pred_phi_bkg[:,1]
  #   my_variables.insert(0,"score")
  #   print(my_variables)
  #   save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)
  #   #print(save_bkg)
  #   ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
  #   rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
  #   #with h5py.File("v9p1_"+pfn_model+"_totalBkgALL_skim0.hdf5","w") as h5f:
  #   with h5py.File("v9p2_"+pfn_model+"_data"+str(dataYear)+".hdf5","w") as h5f:
  #     dset = h5f.create_dataset("data",data=rec_bkg)
  #   print("Saved hdf5 for "+pfn_model+" data"+str(dataYear))



####### - VERSION FOR PARALLEL EVAL ON SMALL QCD FILES - ########

##  ## Load testing data
##  x_events = -1 ## -1 for all events
##  bkg_arrays = []
##  with open("test.txt", "r") as f:
##    files = []
##    for line in f:
##      line = line.strip()
##      files.append(line)
##  i=0
##  for bkg_file in files:
##    print(bkg_file)
##    my_variables = ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]
##    
##    ## evaluate bkg
##    bkg2,mT_bkg = getTwoJetSystem(x_events,"../v8.1/skim1/"+bkg_file, my_variables, False)
##    scaler = load(arch_dir+pfn_model+'_scaler.bin')
##    bkg2,_ = apply_StandardScaling(bkg2,scaler,False) 
##    phi_bkg = graph.predict(bkg2)
##    pred_phi_bkg = classifier.predict(phi_bkg)
##    # ## Classifier loss
##    bkg_loss = pred_phi_bkg[:,1]
##    my_variables.insert(0,"score")
##    all_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)
##    save_bkg = all_bkg[all_bkg[:,0]>0.95]
##    bkg_arrays.append(save_bkg)
##    print("Mini bkg: ", save_bkg.shape)
##    print()
##    if (i % 10 == 0 and i != 0):
##      tmp_bkg = np.concatenate(bkg_arrays[i-10:i],axis=0)
##      ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
##      rec_tmp_bkg = np.rec.array(tmp_bkg, dtype=ds_dt)
##      with h5py.File("v8p1_"+pfn_model+"_skim1_tmp"+str(i)+".hdf5","w") as h5f:
##        dset = h5f.create_dataset("data",data=rec_tmp_bkg)
##      print("Saved hdf5 for tmp"+str(i))
##    i+=1
##  
##  total_bkg = np.concatenate(bkg_arrays,axis=0)
##  print("Total bkg: ", total_bkg.shape)
##  #print(save_bkg)
##  ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
##  rec_bkg = np.rec.array(total_bkg, dtype=ds_dt)
##  with h5py.File("v8p1_"+pfn_model+"_skim1.hdf5","w") as h5f:
##    dset = h5f.create_dataset("data",data=rec_bkg)
##  print("Saved hdf5 for skim1")

