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
def call_functions(bkg_events, tag, bool_weight, bkg_file,extraVars, dsid, applydir,h5path, bool_pt, max_track, h5_dir,read_dir,pfn_model,file_dir,vae_model='', bool_no_scaling=False, bool_transformed=True, arch_dir_pfn=''):
  cprint(f'{extraVars=}, {pfn_model=}, {vae_model=}', 'red')
  print(arch_dir_pfn+pfn_model+'_graph_arch')
  graph = keras.models.load_model(arch_dir_pfn+pfn_model+'_graph_arch')
  graph.load_weights(arch_dir_pfn+pfn_model+'_graph_weights.h5')
  graph.compile()

## Load classifier model
  classifier = keras.models.load_model(arch_dir_pfn+pfn_model+'_classifier_arch')
  classifier.load_weights(arch_dir_pfn+pfn_model+'_classifier_weights.h5')
  classifier.compile()

  if vae_model !='':
    encoder = keras.models.load_model(arch_dir+vae_model+'_encoder_arch')
    decoder = keras.models.load_model(arch_dir+vae_model+'_decoder_arch')
    vae = VAE(encoder,decoder, kl_loss_scalar=1)
    
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
  bkg2, mT_bkg, _, _, _, _ = getTwoJetSystem(nevents=bkg_events,input_file=bkg_file,
  #bkg2, mT_bkg, bkg_sel, jet_bkg, _, _ = getTwoJetSystem(nevents=bkg_events,input_file=bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight,  extraVars=extraVars, plot_dir=plot_dir,seed=seed,max_track=max_track, bool_pt=bool_pt, h5_dir=h5_dir, bool_select_all=True, read_dir=read_dir)

  print(mT_bkg.shape)
  scaler = load(arch_dir_pfn+pfn_model+'_scaler.bin')
  bkg2,_ = apply_StandardScaling(bkg2,scaler,False)

  plot_vectors(bkg2,bkg2,tag_file="ANTELOPE_"+str(dsid)+'_', tag_title=f" (ANTELOPE) {str(dsid)}", plot_dir=plot_dir, bool_sig_on=False, labels=[str(dsid)])# change
  plot_single_variable([mT_bkg[:,2]],h_names= [bkg_file],weights_ls=[mT_bkg[:,1]], tag_title= f'leading jet pT  {str(dsid)}', plot_dir=plot_dir,logy=True, tag_file='jet1_pt_'+str(dsid))
  plot_single_variable([mT_bkg[:,0]],h_names= [bkg_file],weights_ls=[mT_bkg[:,1]], tag_title= f'{extraVars[0]} {str(dsid)} (weighted)', plot_dir=plot_dir,logy=True, tag_file='mT_jj_'+str(dsid))

# plot_vectors(bkg2,sig2,"PFN")
  phi_bkg = graph.predict(bkg2)
  vae_min_dict, vae_max_dict, pfn_min_dict,pfn_max_dict={},{},{},{}
  # technically these are loss transformed!
  vae_min_dict['09_26_23_10_38']= {'mse': -12.634254306189314, 'multi_reco':-12.545995242335483, 'multi_kl': -20.211301490367624 , 'multi_mse': -12.542977457162644}
  vae_max_dict['09_26_23_10_38']= {'mse': -3.456886217152505, 'multi_reco':-3.4590470421545785, 'multi_kl':-9.330598758710815, 'multi_mse': -3.458533554054478}
  vae_max_dict['09_27_23_01_32']= {'mse': 4.051665855814887 ,'multi_reco': 4.14657246457546, 'multi_kl':6.57444001122076, 'multi_mse':6.6017456361377675 }
  vae_max_dict['09_27_23_01_32']= {'mse': 4.051665855814887 ,'multi_reco': 2.986, 'multi_kl':6.57444001122076, 'multi_mse':6.35 }
  #wrong
  vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.166914871440587, 'multi_kl':0.9695256492112576, 'multi_mse':1.0634667425682687 }
  vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.41, 'multi_kl':0.9695256492112576, 'multi_mse':0.9376 }
#  vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.166914871440587, 'multi_kl':0.9695256492112576, 'multi_mse':-3.166914871440587 }
  """
  vae_min_dict['09_26_23_01_32']= {'mse': ,'multi_reco':, 'multi_kl':, 'multi_mse': }
  vae_min_dict['09_26_23_10_38']= {'mse': -12.740280963806276, 'multi_reco':12.681404701687859, 'multi_kl': -20.29521939559835 , 'multi_mse': -12.67853443999128}
  vae_max_dict['09_26_23_10_38']= {'mse': -3.483815150531365, 'multi_reco': -3.48506892905624, 'multi_kl':-10.672584950009908, 'multi_mse': -3.48466478371146}
 
  vae_min_dict['09_26_23_01_32']= {'mse': -3.4774327321494654,'multi_reco':-3.411389944447053, 'multi_kl':0.7924726076345564, 'multi_mse':0.9376798510229637 }
  vae_max_dict['09_26_23_01_32']= {'mse':2.992416483473239 ,'multi_reco':2.989450855262472, 'multi_kl':6.328397109439065, 'multi_mse': 6.3575135820668995 }

  """
  pfn_min_dict['09_26_23_10_38']= 0
  pfn_max_dict['09_26_23_10_38']= 204.44198608398438
# each event has a pfn score
## Classifier loss
  if vae_model =='': # if PFN 
    pred_phi_bkg = classifier.predict(phi_bkg)
    bkg_loss = pred_phi_bkg[:,1]
  else: # if PFN + AE
    ## Scale phis - values from v1 training
    phi_bkg = phi_bkg.astype('float64')
    eval_min = 0.0
    eval_max = 109.87523
    #eval_max = np.amax(phi_bkg)
    #eval_min = np.amin(phi_bkg)

    if not bool_no_scaling :# scaling 
      eval_min = pfn_min_dict[file_dir]
      eval_max = pfn_max_dict[file_dir]
      phi_bkg = (phi_bkg - eval_min)/(eval_max-eval_min)
    
    plot_phi(phi_bkg,tag_file="PFN_phi_input_"+str(dsid), tag_title=f"{str(dsid)} Input", plot_dir=plot_dir)
    pred_phi_bkg = vae.predict(phi_bkg)['reconstruction']
    bkg_loss={}
    # ## AE loss
    bkg_loss['mse'] = np.array(keras.losses.mse(phi_bkg, pred_phi_bkg))
    bkg_loss['multi_reco'], bkg_loss['multi_kl'],  bkg_loss['multi_mse']=get_multi_loss_each(vae, phi_bkg)
    
    methods=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
    new_methods=[]
    if bool_transformed:
      old_methods=methods.copy() # essential that new_methods and methods are separate b/c otherwise, will loop through methods that are already transformed
      for method in old_methods:
        new_method=f'{method}_transformed_log_sig'
        #new_method=f'{method}_transformed'
        print(f'{method=}, {new_method=}')
        loss=np.log(bkg_loss[method])
#        max_loss=np.max(loss)
#        min_loss=np.min(loss)
        """
        min_loss = vae_min_dict[file_dir][method]
        max_loss = vae_max_dict[file_dir][method]
        print(f'{max_loss=}, {min_loss=}, {loss[:5]}')
        loss_transformed_bkg = (loss - min_loss)/(max_loss -min_loss)
        """
        loss_transformed_bkg = 1/(1 + np.exp(-loss))
        bkg_loss[new_method] =loss_transformed_bkg
        new_methods.append(new_method)
        # xlog=False plots
  for method in new_methods: # transformed
    plot_score(bkg_loss[method], np.array([]), False, xlog=False, tag_file=vae_model+f'_{method}_{str(dsid)}', tag_title=vae_model+f' {method} {str(dsid)}', plot_dir=plot_dir, bool_pfn=False, labels=[str(dsid)]) # anomaly score 

  for method in old_methods:
    
    plot_score(bkg_loss[method][bkg_loss[method]>0], np.array([]),False, xlog=True, tag_file=vae_model+'_pos'+f'_{method}_{str(dsid)}', tag_title=vae_model + ' (score > 0)'+f' {method} {str(dsid)}', plot_dir=plot_dir, bool_pfn=False, labels=[str(dsid)]) # anomaly score
#    except: print('skip b/c it failed for some reason')
  newVars=['mse', 'multi_reco', 'multi_kl', 'multi_mse',
    'mse_transformed', 'multi_reco_transformed', 'multi_kl_transformed', 'multi_mse_transformed']
#  newVars=['score', 'score_total', 'score_kl', 'score_reco', ]
  newVars+=extraVars

#  newVars=extraVars.insert(0,"score")
  save_bkg = np.concatenate((
   bkg_loss['mse'][:,None], 
   bkg_loss['multi_reco'][:,None],
   bkg_loss['multi_kl'][:,None],
   bkg_loss['multi_mse'][:,None],


   bkg_loss['mse_transformed'][:,None],
   bkg_loss['multi_reco_transformed'][:,None],
   bkg_loss['multi_kl_transformed'][:,None],
   bkg_loss['multi_mse_transformed'][:,None],

   mT_bkg),axis=1)

  ds_dt = np.dtype({'names':newVars,'formats':[(float)]*len(newVars)})
  rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
 

  with h5py.File(h5path,"w") as f:
    dset = f.create_dataset("data",data=rec_bkg)
  return rec_bkg
  #return rec_bkg, rec_sig
def add_column(input_h5path, output_h5path, plot_dir, vae_model, dsid):
  
  # read hdf5 
  with h5py.File(input_h5path,"r") as f:
    bkg_loss = f.get('data')[:]
  # transform
  print(bkg_loss.dtype.names)
  print(bkg_loss['mse'])
  methods=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
  new_methods=[]
  bkg_loss_new={} 
  for method in methods:
    new_method=f'{method}_transformed_log_sig'
    print(f'{method=}, {new_method=}')
    loss=np.log(bkg_loss[method]).flatten()
#    print(loss.shape)
    loss_transformed_bkg = 1/(1 + np.exp(-loss))
    bkg_loss_new[new_method] =loss_transformed_bkg
    new_methods.append(new_method)
  # reevaluate from the columns from old methods

  print(bkg_loss['mse'].shape,bkg_loss_new['mse_transformed_log_sig'].shape) 
  print(bkg_loss['mse'][:,None].shape,bkg_loss_new['mse_transformed_log_sig'][:,None].shape) 
  print(bkg_loss_new[new_method].shape)
  save_bkg = np.concatenate(( # it is important that the data from the hdf5 file doesn't get expanded to a new shape ( no calling with [:, None] to be compatible with the new column datashapes)
   bkg_loss['mse'], 
   bkg_loss['multi_reco'],
   bkg_loss['multi_kl'],
   bkg_loss['multi_mse'],

   bkg_loss['mse_transformed'],
   bkg_loss['multi_reco_transformed'],
   bkg_loss['multi_kl_transformed'],
   bkg_loss['multi_mse_transformed'],

   bkg_loss['mT_jj'], 
   bkg_loss['weight'],
   bkg_loss['jet1_pt'],

   bkg_loss_new['mse_transformed_log_sig'][:,None],
   bkg_loss_new['multi_reco_transformed_log_sig'][:,None],
   bkg_loss_new['multi_kl_transformed_log_sig'][:,None],
   bkg_loss_new['multi_mse_transformed_log_sig'][:,None]
   ),axis=1)

  print(f"{bkg_loss.shape=} , {bkg_loss_new['mse_transformed_log_sig'].shape=}, {bkg_loss_new['multi_kl_transformed_log_sig'].shape=}, { bkg_loss_new['multi_kl_transformed_log_sig'].shape=},{bkg_loss_new['multi_mse_transformed_log_sig'].shape=}") 
  print(f"{save_bkg.shape}")
  keys =list(bkg_loss.dtype.names)
  keys+=list(bkg_loss_new.keys())
  print(len(keys))
  print(save_bkg.dtype.itemsize)
  #keys =list(bkg_loss_new.keys())
  print(keys)
  ds_dt = np.dtype({'names':keys,'formats':[(float)]*len(keys)})
 # print(ds_dt)
  rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
  with h5py.File(output_h5path,"w") as f:
    dset = f.create_dataset("data",data=rec_bkg)
  print(output_h5path)
  for method in new_methods: # transformed
    plot_score(rec_bkg[method], np.array([]), False, xlog=False, tag_file=vae_model+f'_{method}_{str(dsid)}', tag_title=vae_model+f' {method} {str(dsid)}', plot_dir=plot_dir, bool_pfn=False, labels=[str(dsid)]) # anomaly score 

  for method in methods:
    plot_score(rec_bkg[method][bkg_loss[method]>0], np.array([]),False, xlog=True, tag_file=vae_model+'_pos'+f'_{method}_{str(dsid)}', tag_title=vae_model + ' (score > 0)'+f' {method} {str(dsid)}', plot_dir=plot_dir, bool_pfn=False, labels=[str(dsid)]) # anomaly score
  # write to the file
  return rec_bkg

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
myVars= ["mT_jj", "weight", "jet1_pt"]# if this is empty
#myVars= ["mT_jj", "weight"]# if this is empty
pfn_model = 'PFNv6'
#vae_model = 'ANTELOPE'
vae_model = 'vANTELOPE'
#bkg_loss_type='bkg_loss'#'bkg_loss' 'bkg_total_loss', 'bkg_kl_loss', ''bkg_reco_loss', 
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
#h5_dir='/nevis/katya01/data/users/kpark/svj-vae/h5dir/jul28/'
h5_dir='/nevis/katya01/data/users/kpark/svj-vae/h5dir/antelope/aug17_jetpt'
#all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/paramscan_new/07_24_23_07_11/' # change
bool_transformed=True


# change
file_dir='09_26_23_10_38'
#file_dir='09_27_23_01_32'
all_dir=f'/nevis/katya01/data/users/kpark/svj-vae/results/grid_sept26/{file_dir}/' # change
#all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/grid_sept26/09_27_23_01_32/'
bool_no_scaling=True
#sigmoid function -> get_decoder in models.py or actually this might be remembered from the architecture 
#scaling

arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/'
#all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/antelope/' # change
applydir=all_dir+'applydir/'
plot_dir=applydir+'/plots/'
if not os.path.exists(applydir):
  os.mkdir(applydir)
  print('made ', applydir)
arch_dir=all_dir+'architectures_saved/'
dsids=list(range(515487,515527))
#dsids=list(range(515487,515527))
corrupt_files=[515508, 515511,515493]
dsids=[x for x in dsids if x not in corrupt_files ]
file_ls=[]
for dsid in dsids:
  file_ls.append("skim3.user.ebusch."+str(dsid)+".root")
#stdoutOrigin=sys.stdout
#sys.stdout = open(applydir+f'stdout.txt', 'w')
filetag_ls=[extract_tag(filename=fl) for fl in file_ls]
if vae_model=='': # if evaluating PFN
  sig_file_prefix='v8p1_'
else:sig_file_prefix=f'v8p1_{vae_model}_' # if evaluating ANTELOPE
if vae_model=='': # if evaluating PFN
  bkg_file_prefix='v9p1_'
else:bkg_file_prefix=f'v8p1_{vae_model}_' # if evaluating ANTELOPE
# HERE


  
  



for fl in file_ls:
  dsid=fl.split('.')[-2]
  print('*'*30)
  print(fl) 
  h5path=applydir+'/'+f'{sig_file_prefix}{dsid}'+".hdf5" 
  output_h5path=applydir+'/'+f'{sig_file_prefix}{dsid}_new'+".hdf5" 
  #h5path=applydir+'/'+f'{sig_file_prefix}{dsid}_{bkg_loss_type}'+".hdf5" 
  cprint(f'{dsid=}, {h5path=}', 'green')
 # my_variables= ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]
  tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
  cprint(fl,'blue')
  cprint(f'{h5path}', 'blue')
  """ 
  if  os.path.exists(h5path): # and (dsid !=515429):
    with h5py.File(h5path,"r") as f:
      dset = f.get('data')[:]
  else:    rec_bkg=call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=fl,extraVars=myVars, dsid=dsid,applydir=applydir, h5path=h5path,bool_pt=bool_pt, max_track=max_track, h5_dir=h5_dir, read_dir='/data/users/ebusch/SVJ/autoencoder/v8.1/', file_dir=file_dir,pfn_model=pfn_model, vae_model=vae_model, bool_no_scaling=bool_no_scaling, bool_transformed=bool_transformed, arch_dir_pfn=arch_dir_pfn)
  """
  add_column(input_h5path=h5path, output_h5path=output_h5path, plot_dir=plot_dir,vae_model=vae_model, dsid=dsid) 

bkg_file="skim0.user.ebusch.QCDskim.root"
tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
dsid=bkg_file.split('.')[-2]
h5path=applydir+'/'+f'{sig_file_prefix}{dsid}'+".hdf5" 
output_h5path=applydir+'/'+f'{sig_file_prefix}{dsid}_new'+".hdf5" 
#h5path=applydir+'/'+"v8p1_"+str(dsid)+".hdf5"
cprint(h5path, 'magenta')
""" 
if  os.path.exists(h5path):
  with h5py.File(h5path,"r") as f:
    dset = f.get('data')[:]

else: rec_bkg=call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=bkg_file,extraVars=myVars, dsid=dsid,applydir=applydir, h5path=h5path, bool_pt=bool_pt, max_track=max_track, h5_dir=h5_dir, read_dir='/data/users/ebusch/SVJ/autoencoder/v9.1/',file_dir=file_dir,pfn_model=pfn_model, vae_model=vae_model, bool_no_scaling=bool_no_scaling, bool_transformed=bool_transformed, arch_dir_pfn=arch_dir_pfn)
  """ 
add_column(input_h5path=h5path, output_h5path=output_h5path, plot_dir=plot_dir,vae_model=vae_model, dsid=dsid) 
#comment here
title=f'track={max_track}'
key='multi_kl_transformed_log_sig'
#key='multi_kl_transformed'
#score=getSignalSensitivityScore(bkg_loss, sig_loss)
#print("95 percentile score = ",score)
#grid_s_sqrt_b( bkg_scale=5, sig_file_prefix=sig_file_prefix,bkg_file_prefix=bkg_file_prefix, title=title, all_dir=all_dir,cms=False, key=key)
grid_s_sqrt_b(score_cut=0.97, bkg_scale=5, sig_file_prefix=sig_file_prefix,bkg_file_prefix=bkg_file_prefix, title=title, all_dir=all_dir,cms=False, key=key)
grid_scan(title, all_dir=all_dir, sig_file_prefix=sig_file_prefix,bkg_file_prefix=sig_file_prefix, key=key)
#sys.stdout =stdoutOrigin
"""
vae_min_dict, vae_max_dict, pfn_min_dict,pfn_max_dict={},{},{},{}

vae_min_dict['09_26_23_10_38']= {'mse': -12.634254306189314, 'multi_reco':-12.545995242335483, 'multi_kl': -20.211301490367624 , 'multi_mse': -12.542977457162644}
vae_max_dict['09_26_23_10_38']= {'mse': -3.456886217152505, 'multi_reco':-3.4590470421545785, 'multi_kl':-9.330598758710815, 'multi_mse': -3.458533554054478}
vae_max_dict['09_27_23_01_32']= {'mse': 4.051665855814887 ,'multi_reco': 4.14657246457546, 'multi_kl':6.57444001122076, 'multi_mse':6.6017456361377675 }
  #wrong
vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.166914871440587, 'multi_kl':0.9695256492112576, 'multi_mse':1.0634667425682687 }
dsids= ['515502']
"""





dsids= ['515502', '515499']
from helper import Label
keys=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
for method_scale in keys:
  hists=[]
  weight_ls=[]
  h_names=[]
  method=f'{method_scale}_transformed_log_sig'
  #method='multi_reco_transformed'
  bkgpath=applydir+f"{bkg_file_prefix}QCDskim_new.hdf5"
  with h5py.File(bkgpath,"r") as f:
    bkg_data = f.get('data')[:]
  
  """
  loss= np.log(bkg_data[method_scale])
  min_loss = vae_min_dict[file_dir][method_scale]
  max_loss = vae_max_dict[file_dir][method_scale]
  loss_fixed = (loss - min_loss)/(max_loss -min_loss)
  print(f'{file_dir=}, {min_loss=}, {max_loss=} {np.max(loss_fixed)=}, {np.min(loss_fixed)=}')
  """
  loss_fixed=bkg_data[method]
  hists.append(loss_fixed)
  weight_ls.append(bkg_data['weight'])
  h_names.append(f'QCD ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}]) ')
  #h_names.append(f'QCD (s.w. test: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}]) ')
  
  for dsid in dsids:
    sigpath=applydir+f"{sig_file_prefix}{dsid}_new"+".hdf5"
      # sigpath="../v8.1/"+sig_file_prefix+str(dsid)+".hdf5"
    with h5py.File(sigpath,"r") as f:
      sig1_data = f.get('data')[:]
    mass=Label(dsid).get_m(bool_num=True)
    print(mass)
    rinv=Label(dsid).get_rinv(bool_num=True)
  #  loss= np.log(sig1_data[method_scale])
    """
    min_loss = vae_min_dict[file_dir][method_scale]
    max_loss = vae_max_dict[file_dir][method_scale]
    loss_fixed = (loss - min_loss)/(max_loss -min_loss)
    print(f'{file_dir=}, {min_loss=}, {max_loss=}, {np.max(loss_fixed)=}, {np.min(loss_fixed)=}')
    """
    loss_fixed=sig1_data[method]
    hists.append(loss_fixed)
    weight_ls.append(sig1_data['weight'])
    #mass = dsid_mass[dsid]
    h_names.append(f'{mass} GeV {rinv} ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}])')
    #h_names.append(f'{mass} GeV {rinv} (s.w. test: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}])')
  """
  loss_all=np.concatenate((np.log(bkg_data[method_scale]), np.log(sig1_data[method_scale])))
  max_loss=np.max(loss_all)
  min_loss=np.min(loss_all)
  loss=bkg_data[method_scale]
  loss_log=np.log(loss)
  loss_fixed = (loss_log - min_loss)/(max_loss -min_loss)
  print(f'all {min_loss=}, {max_loss=} {np.max(loss_fixed)=}, {np.min(loss_fixed)=}')
  hists.append(loss_fixed)
  weight_ls.append(bkg_data['weight'])
  h_names.append(f'QCD (s.w. evaluation: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}])')
  
  loss=sig1_data[method_scale]
  loss_log=np.log(loss)
  loss_fixed = (loss_log - min_loss)/(max_loss -min_loss)
  print(f'all {min_loss=}, {max_loss=} {np.max(loss_fixed)=}, {np.min(loss_fixed)=}')
  hists.append(loss_fixed)
  weight_ls.append(sig1_data['weight'])
  h_names.append(f'{mass} GeV {rinv} (s.w. evaluation: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}])')
  """
  plot_single_variable_ratio(hists,h_names=h_names,weights_ls=weight_ls,plot_dir=plot_dir,logy=True, title= f'{method}_comparison', bool_ratio=False)
  plot_single_variable_ratio(hists,h_names=h_names,weights_ls=weight_ls,plot_dir=plot_dir,logy=True, title= f'{method}_comparison', bool_ratio=True)
  
  
  
  sys.exit()
