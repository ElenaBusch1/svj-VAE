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
import h5py
import sys
from termcolor import cprint
from helper import Label
from antelope_h5eval import *
###########functions
def extract_tag(filename):
  filename=filename.split(".")
  if 'QCDskim' in filename:
    tag='bkg'
  elif 'SIGskim' in filename:
    tag='sig'
  elif filename[0]!='user':
    tag=''+filename[3]
  else: tag=''+filename[2]
  return tag

# ---------- Load graph model ----------
def call_functions(bkg_events, tag, bool_weight, bkg_file,extraVars, dsid, h5dir,h5path, bool_pt):
  cprint(f'{extraVars=}', 'red')
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
  


#change here -> file and other changes

  track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e","jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
  track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e","jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
  jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"] # order is important in apply_JetScalingRotation
 
  seed=0
  
  plot_dir=h5dir+'/plots_dsid/'
  if not os.path.exists(plot_dir):
      
    os.mkdir(plot_dir)

  cprint(f'{extraVars=}', 'magenta')
  bkg2, mT_bkg, bkg_sel, jet_bkg = getTwoJetSystem(nevents=bkg_events,input_file=bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight,  extraVars=extraVars, plot_dir=plot_dir,seed=seed, bool_pt=bool_pt)


  scaler = load(arch_dir+pfn_model+'_scaler.bin')
  bkg2,_ = apply_StandardScaling(bkg2,scaler,False)

# plot_vectors(bkg2,sig2,"PFN")

  phi_bkg = graph.predict(bkg2)


# each event has a pfn score 
  pred_phi_bkg = classifier.predict(phi_bkg)


# write on html


## Classifier loss
  bkg_loss = pred_phi_bkg[:,1]
  newVars=['score']
  newVars+=extraVars
  
#  newVars=extraVars.insert(0,"score")
  save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)



  ds_dt = np.dtype({'names':newVars,'formats':[(float)]*len(newVars)})
  rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)

  with h5py.File(h5path,"w") as h5f:
    dset = h5f.create_dataset("data",data=rec_bkg)

  """
  with h5py.File(h5path,"r") as f:
   #with h5py.File("../v8.1/v8p1_PFNv2_"+str(dsid)+".hdf5","r") as f:
    sigv1_data = f.get('data')[:]
  sigv1_loss = sigv1_data["score"]
  print(sigv1_loss)
  """

  return rec_bkg
  #return rec_bkg, rec_sig

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
title='July12'
myVars= ["mT_jj", "weight"]# if this is empty
pfn_model = 'PFN'
## Load testing data
sig_events = 10000000000
bkg_events = 10000000000
bool_weight=True
if bool_weight:weight_tag='ws'
else:weight_tag='nws'
#tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
#bool_rewrite=False
bool_rewrite=True

bool_pt=True

dir_all='/nevis/katya01/data/users/kpark/svj-vae/results/07_12_23_08_47/' # change
h5dir=dir_all+'h5dir/'
if not os.path.exists(h5dir):
  os.mkdir(h5dir)
arch_dir=dir_all+'architectures_saved/'
dsids=list(range(515487,515527))
corrupt_files=[515508, 515511,515493]
dsids=[x for x in dsids if x not in corrupt_files ]
file_ls=[]
for dsid in dsids:
  file_ls.append("skim3.user.ebusch."+str(dsid)+".root")

filetag_ls=[extract_tag(filename=fl) for fl in file_ls]

for fl in file_ls:
  dsid=fl.split('.')[-2]
  print('*'*30)
  print(fl) 
  h5path=h5dir+'/'+"v8p1_"+str(dsid)+".hdf5" 
  cprint(f'{dsid=}, {h5path=}', 'green')
 # my_variables= ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]
  tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
  cprint(fl,'blue')
   
  cprint(f'{dsid=},', 'green')
  rec_bkg=call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=fl,extraVars=myVars, dsid=dsid,h5dir=h5dir, h5path=h5path, bool_pt=bool_pt)
#for background 
bkg_file="skim3.user.ebusch.QCDskim.root"
tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
   
rec_bkg=call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=bkg_file,extraVars=myVars, dsid=bkg_file.split('.')[-2],h5dir=h5dir, h5path=h5path, bool_pt=bool_pt)
#"""
grid_scan(title)
