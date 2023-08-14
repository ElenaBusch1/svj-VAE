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
#h5_dir, max_track
# ---------- Load graph model ----------
def call_functions(bkg_events, tag, bool_weight, bkg_file,extraVars, dsid, applydir,h5path, bool_pt, max_track, h5_dir,read_dir,pfn_model,vae_model=''):
  cprint(f'{extraVars=}, {pfn_model=}, {vae_model=}', 'red')
  print(arch_dir+pfn_model+'_graph_arch')
  graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
  graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
  graph.compile()

## Load classifier model
  classifier = keras.models.load_model(arch_dir+pfn_model+'_classifier_arch')
  classifier.load_weights(arch_dir+pfn_model+'_classifier_weights.h5')
  classifier.compile()

  if vae_model !='':
    encoder = keras.models.load_model(arch_dir+vae_model+'_encoder_arch')
    decoder = keras.models.load_model(arch_dir+vae_model+'_decoder_arch')
    vae = VAE(encoder,decoder)
    
    vae.get_layer('encoder').load_weights(arch_dir+vae_model+'_encoder_weights.h5')
    vae.get_layer('decoder').load_weights(arch_dir+vae_model+'_decoder_weights.h5')
    
    vae.compile(optimizer=keras.optimizers.Adam())
 
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
  
  plot_dir=applydir+'/plots_dsid/'
  if not os.path.exists(plot_dir):
      
    os.mkdir(plot_dir)

  cprint(f'{extraVars=}', 'magenta')
  bkg2, mT_bkg, bkg_sel, jet_bkg, _, _ = getTwoJetSystem(nevents=bkg_events,input_file=bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight,  extraVars=extraVars, plot_dir=plot_dir,seed=seed,max_track=max_track, bool_pt=bool_pt, h5_dir=h5_dir, bool_select_all=True, read_dir=read_dir)


  scaler = load(arch_dir+pfn_model+'_scaler.bin')
  bkg2,_ = apply_StandardScaling(bkg2,scaler,False)

# plot_vectors(bkg2,sig2,"PFN")
  phi_bkg = graph.predict(bkg2)

# each event has a pfn score
## Classifier loss
  if vae_model =='': # if PFN 
    pred_phi_bkg = classifier.predict(phi_bkg)
    bkg_loss = pred_phi_bkg[:,1]
  else: # if PFN + AE
    ## Scale phis - values from v1 training
    eval_min = 0.0
    eval_max = 109.87523
    #eval_max = np.amax(phi_bkg)
    #eval_min = np.amin(phi_bkg)
    phi_bkg = (phi_bkg - eval_min)/(eval_max-eval_min)
    
    pred_phi_bkg = vae.predict(phi_bkg)['reconstruction']
    
    # ## AE loss
    bkg_loss = np.array(keras.losses.mse(phi_bkg, pred_phi_bkg))


  newVars=['score']
  newVars+=extraVars
  
#  newVars=extraVars.insert(0,"score")
  save_bkg = np.concatenate((bkg_loss[:,None], mT_bkg),axis=1)

  ds_dt = np.dtype({'names':newVars,'formats':[(float)]*len(newVars)})
  rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)

  with h5py.File(h5path,"w") as f:
    dset = f.create_dataset("data",data=rec_bkg)

  return rec_bkg
  #return rec_bkg, rec_sig

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
myVars= ["mT_jj", "weight"]# if this is empty
pfn_model = 'PFNv6'
#vae_model = 'ANTELOPE'
vae_model = 'vANTELOPE'

## Load testing data
sig_events =100000
#bkg_events =10
bkg_events = -1
bool_weight=True
if bool_weight:weight_tag='ws'
else:weight_tag='nws'
#tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
#bool_rewrite=False
bool_rewrite=True

bool_pt=False
#max_track=15# CHECK THIS
max_track=80# CHECK THIS
h5_dir='/nevis/katya01/data/users/kpark/svj-vae/h5dir/jul28/'

#all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/paramscan_new/07_24_23_07_11/' # change

all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/antelope/' # change
applydir=all_dir+'applydir/'
if not os.path.exists(applydir):
  os.mkdir(applydir)
arch_dir=all_dir+'architectures_saved/'
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
  if vae_model=='':
    sig_file_prefix='v8p1_'
  else:sig_file_prefix=f'v8p1_{vae_model}_'
  h5path=applydir+'/'+sig_file_prefix+str(dsid)+".hdf5" 
  cprint(f'{dsid=}, {h5path=}', 'green')
 # my_variables= ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]
  tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
  cprint(fl,'blue')
   
  if  os.path.exists(h5path): # and (dsid !=515429):
    with h5py.File(h5path,"r") as f:
      dset = f.get('data')[:]
  else:    rec_bkg=call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=fl,extraVars=myVars, dsid=dsid,applydir=applydir, h5path=h5path,bool_pt=bool_pt, max_track=max_track, h5_dir=h5_dir, read_dir='/data/users/ebusch/SVJ/autoencoder/v8.1/', pfn_model=pfn_model, vae_model=vae_model)

bkg_file="skim3.user.ebusch.QCDskim.root"
tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
dsid=bkg_file.split('.')[-2]
h5path=applydir+'/'+"v8p1_"+str(dsid)+".hdf5"
cprint(h5path, 'magenta')
if  os.path.exists(h5path):
  with h5py.File(h5path,"r") as f:
    dset = f.get('data')[:]

else: rec_bkg=call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=bkg_file,extraVars=myVars, dsid=dsid,applydir=applydir, h5path=h5path, bool_pt=bool_pt, max_track=max_track, h5_dir=h5_dir, read_dir='/data/users/ebusch/SVJ/autoencoder/v9.1/',pfn_model=pfn_model, vae_model=vae_model)
title=f'track={max_track}'
grid_s_sqrt_b(score_cut=0.97, bkg_scale=5, sig_file_prefix=sig_file_prefix,bkg_file_prefix=bkg_file_prefix, title=title, all_dir=all_dir,cms=False)
grid_scan(title, all_dir=all_dir, sig_file_prefix=sig_file_prefix,bkg_file_prefix=bkg_file_prefix)
