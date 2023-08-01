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
pfn_model = 'PFNv4'
arch_dir = "architectures_saved/"

## ---------- Load graph model ----------
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## Load classifier model
classifier = keras.models.load_model(arch_dir+pfn_model+'_classifier_arch')
classifier.load_weights(arch_dir+pfn_model+'_classifier_weights.h5')
classifier.compile()

## Load history
# with open(arch_dir+ae_model+"_history.json", 'r') as f:
#     h = json.load(f)
# print(h)
# print(type(h))

print ("Loaded model")

## Load testing data
x_events = -1 ## -1 for all events
my_variables = ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "deltaY_12", "dphi_12", "weight", "mcEventWeight"]

## evaluate bkg
bkg2,mT_bkg = getTwoJetSystem(x_events,"../v9.1/user.ebusch.data16.root", my_variables, False)
scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg2,scaler,False) 
phi_bkg = graph.predict(bkg2)
pred_phi_bkg = classifier.predict(phi_bkg)
# ## Classifier loss
bkg_loss = pred_phi_bkg[:,1]
my_variables.insert(0,"score")
print(my_variables)
save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)
#print(save_bkg)
ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
with h5py.File("v9p1_PFNv4_data16_100k.hdf5","w") as h5f:
  dset = h5f.create_dataset("data",data=rec_bkg)
print("Saved hdf5 for QCDskim")
quit()

## evaluate signals
dsids = [515499, 515502]
#dsids = range(515486,515527)
for dsid in dsids:
  my_variables = ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "deltaY_12", "dphi_12", "weight", "mcEventWeight"]
  try:
    bkg2,mT_bkg = getTwoJetSystem(x_events,"../v8.1/skim3.user.ebusch."+str(dsid)+".root", my_variables, False)
  except:
    continue
  scaler = load(arch_dir+pfn_model+'_scaler.bin')
  bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
  
  phi_bkg = graph.predict(bkg2)
   
  pred_phi_bkg = classifier.predict(phi_bkg)
  
  # ## Classifier loss
  bkg_loss = pred_phi_bkg[:,1]
  
  my_variables.insert(0,"score")
  print(my_variables)
  save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)
  #print(save_bkg)
  ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
  rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
  
  with h5py.File("v8p1_PFNv4_"+str(dsid)+".hdf5","w") as h5f:
    dset = h5f.create_dataset("data",data=rec_bkg)
  print("Saved hdf5 for ", dsid)


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

