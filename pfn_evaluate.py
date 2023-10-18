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
def call_functions(bkg_events, tag, bool_weight, bkg_file,extraVars, dsid, applydir,h5path, bool_pt, max_track, h5_dir,read_dir,pfn_model,file_dir,vae_model='', bool_no_scaling=False, bool_transformed=True, arch_dir_pfn='', bool_float64=False):
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
  
  if mT_bkg[:,1].any():
    bool_weight=True
  else: bool_weight=False # if data, not drawing with weights

  plot_vectors(bkg2,bkg2,tag_file="ANTELOPE_"+str(dsid)+'_', tag_title=f" (ANTELOPE) {str(dsid)}", plot_dir=plot_dir, bool_sig_on=False, labels=[str(dsid)])# change
  plot_single_variable([mT_bkg[:,2]],h_names= [bkg_file],weights_ls=[mT_bkg[:,1]], tag_title= f'leading jet pT  {str(dsid)}', plot_dir=plot_dir,logy=True, tag_file='jet1_pt_'+str(dsid), bool_weight=bool_weight)
  plot_single_variable([mT_bkg[:,0]],h_names= [bkg_file],weights_ls=[mT_bkg[:,1]], tag_title= f'{extraVars[0]} {str(dsid)} (weighted)', plot_dir=plot_dir,logy=True, tag_file='mT_jj_'+str(dsid), bool_weight=bool_weight)

# plot_vectors(bkg2,sig2,"PFN")
  phi_bkg = graph.predict(bkg2)
  vae_min_dict, vae_max_dict, pfn_min_dict,pfn_max_dict={},{},{},{}
  """
  # technically these are loss transformed!
  vae_min_dict['09_26_23_10_38']= {'mse': -12.634254306189314, 'multi_reco':-12.545995242335483, 'multi_kl': -20.211301490367624 , 'multi_mse': -12.542977457162644}
  vae_max_dict['09_26_23_10_38']= {'mse': -3.456886217152505, 'multi_reco':-3.4590470421545785, 'multi_kl':-9.330598758710815, 'multi_mse': -3.458533554054478}
  vae_max_dict['09_27_23_01_32']= {'mse': 4.051665855814887 ,'multi_reco': 4.14657246457546, 'multi_kl':6.57444001122076, 'multi_mse':6.6017456361377675 }
  vae_max_dict['09_27_23_01_32']= {'mse': 4.051665855814887 ,'multi_reco': 2.986, 'multi_kl':6.57444001122076, 'multi_mse':6.35 }
  vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.166914871440587, 'multi_kl':0.9695256492112576, 'multi_mse':1.0634667425682687 }
  vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.41, 'multi_kl':0.9695256492112576, 'multi_mse':0.9376 }
  #wrong
#  vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.166914871440587, 'multi_kl':0.9695256492112576, 'multi_mse':-3.166914871440587 }
  """

  # manual 
  pfn_min_dict['09_26_23_10_38']= 0
  pfn_max_dict['09_26_23_10_38']= 204.44198608398438  # got from Final Max in /nevis/katya01/data/users/kpark/svj-vae/results/grid_sept26/09_26_23_10_38/stdout.txt
# each event has a pfn score
## Classifier loss
  if vae_model =='': # if PFN 
    pred_phi_bkg = classifier.predict(phi_bkg)
    bkg_loss = pred_phi_bkg[:,1]
  else: # if PFN + AE
    ## Scale phis - values from v1 training
    if bool_float64:
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

  return phi_bkg, pred_phi_bkg, mT_bkg, extraVars,dsid, h5path, vae_model, bool_transformed, vae, plot_dir, file_dir
 
def write_hdf5(phi_bkg, pred_phi_bkg, mT_bkg, extraVars, dsid, h5path, vae_model, bool_transformed, vae, plot_dir, file_dir, subset=0, n_events=0, bool_split=True):
  if vae_model =='':
    print('not writing hdf5 in write_hdf5 function because using VAE')
  else:
    print(phi_bkg.shape, pred_phi_bkg.shape, mT_bkg.shape)
    # only selects some indices of phi_bkg, pred_phi_bkg, mT_bkg to evaluate
    if bool_split:
      phi_bkg, pred_phi_bkg, mT_bkg= phi_bkg[subset*n_events: (subset+1)*n_events, :]  , pred_phi_bkg[subset*n_events: (subset+1)*n_events, :],   mT_bkg[subset*n_events: (subset+1)*n_events, :]
      print('after split',phi_bkg.shape, pred_phi_bkg.shape, mT_bkg.shape)
#    phi_bkg, pred_phi_bkg, mT_bkg = 
    bkg_loss={}
    # ## AE loss
    bkg_loss['mse'] = np.array(keras.losses.mse(phi_bkg, pred_phi_bkg))
    bkg_loss['multi_reco'], bkg_loss['multi_kl'],  bkg_loss['multi_mse']=get_multi_loss_each(vae, phi_bkg)
    
    methods=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
    new_methods=[]
    if bool_transformed:
      old_methods=methods.copy() # essential that new_methods and methods are separate b/c otherwise, will loop through methods that are already transformed
      for method in old_methods:
        new_method=f'{method}_transformed_log10_sig'
        #new_method=f'{method}_transformed'
        print(f'{method=}, {new_method=}')
        loss=np.log10(bkg_loss[method])
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
  newVars= methods+ new_methods
#  newVars=['mse', 'multi_reco', 'multi_kl', 'multi_mse',
#    'mse_transformed', 'multi_reco_transformed', 'multi_kl_transformed', 'multi_mse_transformed']
#  newVars=['score', 'score_total', 'score_kl', 'score_reco', ]
  newVars+=extraVars

#  newVars=extraVars.insert(0,"score")
  save_bkg = np.concatenate((
   bkg_loss['mse'][:,None], 
   bkg_loss['multi_reco'][:,None],
   bkg_loss['multi_kl'][:,None],
   bkg_loss['multi_mse'][:,None],


   bkg_loss['mse_transformed_log10_sig'][:,None],
   bkg_loss['multi_reco_transformed_log10_sig'][:,None],
   bkg_loss['multi_kl_transformed_log10_sig'][:,None],
   bkg_loss['multi_mse_transformed_log10_sig'][:,None],

   mT_bkg),axis=1)

  ds_dt = np.dtype({'names':newVars,'formats':[(float)]*len(newVars)})
  rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
 

  with h5py.File(f"{h5path.split('.hdf5')[-2]}_{subset}.hdf5","w") as f:
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
  ''' 
  for method in methods:
    new_method=f'{method}_transformed_log10_sig'
    print(f'{method=}, {new_method=}')
    loss=np.log10(bkg_loss[method]).flatten()
#    print(loss.shape)
    loss_transformed_bkg = 1/(1 + np.exp(-loss))
    bkg_loss_new[new_method] =loss_transformed_bkg
    new_methods.append(new_method)
  # reevaluate from the columns from old methods

  ''' 
  print(bkg_loss['mse'].shape,bkg_loss_new['mse_transformed_log10_sig'].shape) 
  print(bkg_loss['mse'][:,None].shape,bkg_loss_new['mse_transformed_log10_sig'][:,None].shape) 
  print(bkg_loss_new[new_method].shape)
  """
   bkg_loss['mse_transformed'],
   bkg_loss['multi_reco_transformed'],
   bkg_loss['multi_kl_transformed'],
   bkg_loss['multi_mse_transformed'],
    bkg_loss_new['mse_transformed_log10_sig'][:,None],
   bkg_loss_new['multi_reco_transformed_log10_sig'][:,None],
   bkg_loss_new['multi_kl_transformed_log10_sig'][:,None],
   bkg_loss_new['multi_mse_transformed_log10_sig'][:,None]
  """
  save_bkg = np.concatenate(( # it is important that the data from the hdf5 file doesn't get expanded to a new shape ( no calling with [:, None] to be compatible with the new column datashapes)
   bkg_loss['mse'], 
   bkg_loss['multi_reco'],
   bkg_loss['multi_kl'],
   bkg_loss['multi_mse'],
   bkg_loss['mse_transformed_log10_sig'],
   bkg_loss['multi_reco_transformed_log10_sig'],
   bkg_loss['multi_kl_transformed_log10_sig'],
   bkg_loss['multi_mse_transformed_log10_sig'],

   bkg_loss['mT_jj'], 
   bkg_loss['weight'],
   bkg_loss['jet1_pt'],

   bkg_loss_new['jet2_width'][:,None] 
   ),axis=1)

#  print(f"{bkg_loss.shape=} , {bkg_loss_new['mse_transformed_log_sig'].shape=}, {bkg_loss_new['multi_kl_transformed_log_sig'].shape=}, { bkg_loss_new['multi_kl_transformed_log_sig'].shape=},{bkg_loss_new['multi_mse_transformed_log_sig'].shape=}") 
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

def transform_dir_txt(file_dir):
  # e.g. v9p1 -> v9.1
 # print(file_dir.replace('p', '.'))
  return file_dir.replace('p', '.')

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
bool_weight=False
if bool_weight:weight_tag='ws'
else:weight_tag='nws'
#tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
#bool_rewrite=False
bool_rewrite=True

bool_pt=False
#max_track=15# CHECK THIS
max_track=80# CHECK THIS
#h5_dir='/nevis/katya01/data/users/kpark/svj-vae/h5dir/jul28/'
h5_dir='/nevis/katya01/data/users/kpark/svj-vae/h5dir/antelope/v9p2'
#h5_dir='/nevis/katya01/data/users/kpark/svj-vae/h5dir/antelope/aug17_jetpt'
#all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/paramscan_new/07_24_23_07_11/' # change
bool_transformed=True


# change
file_dir='10_08_23_04_08'
#file_dir='09_26_23_10_38'
bkg_file_dir='v9p2'
sig_file_dir='v9p2'
#sig_file_dir='v8p1'

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
  file_ls.append("user.ebusch."+str(dsid)+".root")
  #file_ls.append("skim3.user.ebusch."+str(dsid)+".root")
#stdoutOrigin=sys.stdout
#sys.stdout = open(applydir+f'stdout.txt', 'w')
filetag_ls=[extract_tag(filename=fl) for fl in file_ls]
if vae_model=='': # if evaluating PFN
  sig_file_prefix=f'{sig_file_dir}_'
else:sig_file_prefix=f'{sig_file_dir}_{vae_model}_' # if evaluating ANTELOPE
if vae_model=='': # if evaluating PFN
  bkg_file_prefix='{bkg_file_dir}_'
else:bkg_file_prefix=f'{bkg_file_dir}_{vae_model}_' # if evaluating ANTELOPE
sig_read_dir='/data/users/ebusch/SVJ/autoencoder/'+transform_dir_txt(sig_file_dir)+'/'
#sig_read_dir='/data/users/ebusch/SVJ/autoencoder/'+transform_dir_txt('v8.1')+'/'
bkg_read_dir='/data/users/ebusch/SVJ/autoencoder/'+transform_dir_txt(bkg_file_dir)+'/'
#bkg_read_dir='/data/users/kpark/SVJ/MicroNTuples/'+transform_dir_txt(bkg_file_dir)+'/'
#bkg_read_dir='/data/users/ebusch/SVJ/autoencoder/'+transform_dir_txt('v9.1')+'/'
# HERE
"""
Here we evaluate on signal files 
"""
'''
for dsid in dsids:
  mass=Label(str(dsid)).get_m(bool_num=True)
  rinv=Label(str(dsid)).get_rinv(bool_num=True)
  print(dsid, mass, rinv)
sys.exit()
'''
'''
for fl in file_ls:
  dsid=fl.split('.')[-2]
  print('*'*30)
  print(fl) 
  h5path=applydir+'/'+f'{sig_file_prefix}{dsid}_log10'+".hdf5" 
  #h5path=applydir+'/'+f'{sig_file_prefix}{dsid}'+".hdf5" 
  output_h5path=applydir+'/'+f'{sig_file_prefix}{dsid}_log10'+".hdf5" 
  #h5path=applydir+'/'+f'{sig_file_prefix}{dsid}_{bkg_loss_type}'+".hdf5" 
  cprint(f'{dsid=}, {h5path=}', 'green')
 # my_variables= ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]
  tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
  cprint(fl,'blue')
  cprint(f'{h5path}', 'blue')
  if  os.path.exists(h5path): # and (dsid !=515429):
    with h5py.File(h5path,"r") as f:
      dset = f.get('data')[:]
  else:    
    phi_bkg, pred_phi_bkg, mT_bkg,extraVars, dsid, h5path, vae_model, bool_transformed, vae, plot_dir, file_dir =  call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=fl,extraVars=myVars, dsid=dsid,applydir=applydir, h5path=h5path,bool_pt=bool_pt, max_track=max_track, h5_dir=h5_dir, read_dir=sig_read_dir, file_dir=file_dir,pfn_model=pfn_model, vae_model=vae_model, bool_no_scaling=bool_no_scaling, bool_transformed=bool_transformed, arch_dir_pfn=arch_dir_pfn)
    rec_bkg_each=write_hdf5(phi_bkg=phi_bkg, pred_phi_bkg=pred_phi_bkg, mT_bkg=mT_bkg, extraVars=extraVars, dsid=dsid, h5path= h5path, vae_model=vae_model, bool_transformed=bool_transformed, vae=vae, plot_dir=plot_dir, file_dir=file_dir)
  #Here you can add a column to a hdf5 file that was already processed and has new columns  
#  add_column(input_h5path=h5path, output_h5path=output_h5path, plot_dir=plot_dir,vae_model=vae_model, dsid=dsid) 
  #""" 
"""
Here we evaluate on background files 
"""
'''
def list_files(ls):
  ls=sorted(ls)
  min_ls=min(ls)
  max_ls=max(ls)
  assert ls== list(set(range(min_ls, max_ls+1))), "a missing integer between the minimum and the maximum element"
  return f'{min_ls}-{max_ls}' 
#bkg_file="user.ebusch.dataALL.root"
bkg_file="skim0.user.ebusch.bkgAll.root"
#bkg_file="user.ebusch.515487.root"
#bkg_file="skim0.user.ebusch.QCDskim.root"
tag= f'{pfn_model}_2jAvg_MM_{weight_tag}'
dsid=bkg_file.split('.')[-2]
h5path=applydir+'/'+f'{bkg_file_prefix}{dsid}_log10'+".hdf5" 
#h5path=applydir+'/'+f'{bkg_file_prefix}{dsid}'+".hdf5" 
output_h5path=applydir+'/'+f'{bkg_file_prefix}{dsid}_log10'+".hdf5" 
#h5path=applydir+'/'+"v8p1_"+str(dsid)+".hdf5"
cprint(h5path, 'magenta')
if  os.path.exists(h5path):
  with h5py.File(h5path,"r") as f:
    dset = f.get('data')[:]

  new_pt=dset['jet1_pt']
  new_weight=dset['weight'] 
  plot_single_variable([new_pt],h_names= [bkg_file],weights_ls=[new_pt], tag_title= f'leading jet pT  {str(dsid)}', plot_dir=applydir+'/plots_dsid/',logy=True, tag_file='new_jet1_pt_'+str(dsid), bool_weight=bool_weight)
else:
#  phi_bkg, pred_phi_bkg, mT_bkg, extraVars, dsid, h5path, vae_model, bool_transformed, vae, plot_dir, file_dir =  call_functions(bkg_events=bkg_events, tag=tag, bool_weight=bool_weight, bkg_file=bkg_file,extraVars=myVars, dsid=dsid,applydir=applydir, h5path=h5path, bool_pt=bool_pt, max_track=max_track, h5_dir=h5_dir, read_dir=bkg_read_dir,file_dir=file_dir,pfn_model=pfn_model, vae_model=vae_model, bool_no_scaling=bool_no_scaling, bool_transformed=bool_transformed, arch_dir_pfn=arch_dir_pfn)
  # initialize rec_bkg
  # decide how many loops
  # loop thru different QCD files
  n_events=100000
#  n_file = phi_bkg.shape[0]//n_events +1
  n_file=47
  ls_files=[]
  for subset in range(0,n_file):
  #for i in range(0,73):
  # check if the file exists; if so, read
    h5path_subset=f"{h5path.split('.hdf5')[-2]}_{subset}.hdf5"
    if os.path.exists(h5path_subset):
      with h5py.File(h5path_subset,"r") as f:
        rec_bkg_each = f.get('data')[:]

      ls_files.append(subset)
    else: break # if the subset hdf5 doesn't exist, then break right away 
    """
    else:
    # else, evaluate + write
      rec_bkg_each=write_hdf5(phi_bkg=phi_bkg, pred_phi_bkg=pred_phi_bkg, mT_bkg=mT_bkg,extraVars=extraVars, dsid=dsid, h5path= h5path, vae_model=vae_model, bool_transformed=bool_transformed, vae=vae, plot_dir=plot_dir, file_dir=file_dir,  subset=subset, n_events=n_events, bool_split=True)
    # concatenate and write
    """
    if subset==0:
      rec_bkg=rec_bkg_each
    else:
      print(rec_bkg.shape)
      rec_bkg= np.append(rec_bkg, np.array(rec_bkg_each , dtype=rec_bkg_each.dtype))
  print(rec_bkg_each.dtype)
  #ds_dt = np.dtype({'names':newVars,'formats':[(float)]*len(newVars)})
  #rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
  ls_files=list_files(ls_files)
  combined_h5path=f"{h5path.split('.hdf5')[-2]}_{ls_files}.hdf5"
  with h5py.File(combined_h5path,"w") as f:
    dset = f.create_dataset("data",data=rec_bkg[:,None]) # has to have [:, None] to be compatible with the original hdf5 format
"""
Here you can add a column to a hdf5 file that was already processed and has new columns  
""" 
""" 
add_column(input_h5path=h5path, output_h5path=output_h5path, plot_dir=plot_dir,vae_model=vae_model, dsid=dsid) 
""" 

title=f'track={max_track}'
key='multi_kl_transformed_log10_sig'
keys=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
keys=[f'{key}_transformed_log10_sig' for key in keys ]
score_cut_dict={}
#score_cut_dict['09_26_23_10_38']={'multi_kl_transformed_log_sig':2.11e-7, 'multi_mse_transformed_log_sig':1.02e-3}
#score_cut_dict['09_27_23_01_32']={'multi_kl_transformed_log_sig':9.37e-1, 'multi_mse_transformed_log_sig':6.54e-1}
#score_cut_dict['09_27_23_01_32']={'multi_kl_transformed_log10_sig':9.37e-1, 'multi_mse_transformed_log10_sig':6.54e-1, 'multi_reco_transformed_log10_sig': 9.45e-1}
score_cut_dict['09_26_23_10_38']={'multi_kl_transformed_log10_sig':1.06e-3, 'multi_mse_transformed_log10_sig':3.45e-2}
score_cut_dict['09_27_23_01_32']={'multi_kl_transformed_log10_sig':0.72, 'multi_mse_transformed_log10_sig':0.573}
score_cut_dict['10_08_23_04_08']={'multi_kl_transformed_log10_sig':0.72, 'multi_mse_transformed_log10_sig':0.573, 'multi_reco_transformed_log10_sig':0.73}


keys=list(score_cut_dict['10_08_23_04_08'].keys())
cprint(keys, 'red')
#key='multi_kl_transformed'
#score=getSignalSensitivityScore(bkg_loss, sig_loss)
#print("95 percentile score = ",score)
#grid_s_sqrt_b( bkg_scale=5, sig_file_prefix=sig_file_prefix,bkg_file_prefix=bkg_file_prefix, title=title, all_dir=all_dir,cms=False, key=key)
'''
bkg_file='dataALL_log10_0-44.hdf5'
for key in keys:
  print(file_dir,key)
  grid_scan(title, all_dir=all_dir, sig_file_prefix=sig_file_prefix,bkg_file_prefix=sig_file_prefix, bkg_file=bkg_file,key=key)
  grid_s_sqrt_b(score_cut_dict[file_dir][key], bkg_scale=5, sig_file_prefix=sig_file_prefix,bkg_file_prefix=bkg_file_prefix,bkg_file=bkg_file, title=title, all_dir=all_dir,cms=False, key=key)
#sys.stdout =stdoutOrigin
sys.exit()
'''
vae_min_dict, vae_max_dict, pfn_min_dict,pfn_max_dict={},{},{},{}
vae_min_dict['09_26_23_10_38']= {'mse': -12.634254306189314, 'multi_reco':-12.545995242335483, 'multi_kl': -20.211301490367624 , 'multi_mse': -12.542977457162644}
vae_max_dict['09_26_23_10_38']= {'mse': -3.456886217152505, 'multi_reco':-3.4590470421545785, 'multi_kl':-9.330598758710815, 'multi_mse': -3.458533554054478}
vae_max_dict['09_27_23_01_32']= {'mse': 4.051665855814887 ,'multi_reco': 4.14657246457546, 'multi_kl':6.57444001122076, 'multi_mse':6.6017456361377675 }
  #wrong
vae_min_dict['09_27_23_01_32']= {'mse': -3.2118662852702515,'multi_reco':-3.166914871440587, 'multi_kl':0.9695256492112576, 'multi_mse':1.0634667425682687 }
dsids= ['515502']

"""

#"""

#dsids= ['515502', '515499']
dsids= [ '515499', '515502', '515515', '515518']
from helper import Label
keys=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
for method_scale in keys:
  hists=[]
  hists_var=[]
  var='mT_jj'
  weight_ls=[]
  h_names=[]
#  method=method_scale
  method=f'{method_scale}_transformed_log10_sig'
  #method='multi_reco_transformed'
  bkgpath=applydir+f"{bkg_file_prefix}dataALL_log10.hdf5"
  dsid=bkgpath.split('.')[-2].split('_')[-2]
  #bkgpath=applydir+f"{bkg_file_prefix}QCDskim_log10.hdf5"
  with h5py.File(bkgpath,"r") as f:
    bkg_data = f.get('data')[:]
  
  loss_fixed=bkg_data[method]
  hists.append(loss_fixed)
  hists_var.append(bkg_data[var])
  if bkg_data['weight'].any(): # if array contains some element other than 0 
    weight_ls.append(bkg_data['weight'])
  else:#if array contains only zeros
    print(np.ones(bkg_data['weight'].shape).shape)
    print(bkg_data['weight'].shape)
    weight_ls.append(np.ones(bkg_data['weight'].shape))
  
  h_names.append(f'dataALL ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}]) ')
  #h_names.append(f'QCD ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}]) ')
  #h_names.append(f'QCD (s.w. test: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}]) ')
  '''
  for dsid in dsids:
    sigpath=applydir+f"{sig_file_prefix}{dsid}_log10"+".hdf5"
      # sigpath="../v8.1/"+sig_file_prefix+str(dsid)+".hdf5"
    with h5py.File(sigpath,"r") as f:
      sig1_data = f.get('data')[:]
    mass=Label(dsid).get_m(bool_num=True)
    print(mass)
    rinv=Label(dsid).get_rinv(bool_num=True)
  #  loss= np.log(sig1_data[method_scale])
    loss_fixed=sig1_data[method]
    hists.append(loss_fixed)
    weight_ls.append(sig1_data['weight'])
    #mass = dsid_mass[dsid]
    h_names.append(f'{mass} GeV {rinv} ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}])')
    #h_names.append(f'{mass} GeV {rinv} (s.w. test: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}])')
  '''
  bkgpath=applydir+f"{bkg_file_prefix}bkgAll_log10_0-46.hdf5"
  #bkgpath=applydir+f"{bkg_file_prefix}bkgAll_log10.hdf5"
  dsid=bkgpath.split('.')[-2].split('_')[-2]
  #bkgpath=applydir+f"{bkg_file_prefix}QCDskim_log10.hdf5"
  with h5py.File(bkgpath,"r") as f:
    bkg_data = f.get('data')[:]
  
  loss_fixed=bkg_data[method]
  hists.append(loss_fixed)
  hists_var.append(bkg_data[var])
  if bkg_data['weight'].any(): # if array contains some element other than 0 
    weight_ls.append(bkg_data['weight'])
  else:#if array contains only zeros
    print(np.ones(bkg_data['weight'].shape).shape)
    print(bkg_data['weight'].shape)
    weight_ls.append(np.ones(bkg_data['weight'].shape))
  
  h_names.append(f'bkgALL ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}]) ')
  print(hists[0].shape, hists[1].shape,hists_var[0].shape, hists_var[1].shape) 
 # plot_single_variable_ratio(hists,h_names=h_names,weights_ls=weight_ls,plot_dir=plot_dir,logy=True, title= f'{method}_comparison', bool_ratio=False)
  #plot_single_variable_ratio(hists,h_names=h_names,weights_ls=weight_ls,plot_dir=plot_dir,logy=True, title= f'{method}_comparison', bool_ratio=True)
  
#  plot_single_variable_ratio(hists,h_names=h_names,weights_ls=weight_ls,plot_dir=plot_dir,logy=True, title= f'{method}_comparison', bool_ratio=False)
  plot_single_variable_ratio([hists_var[-1],hists_var[-1]],h_names=['bkgALL', 'bkgALL'],weights_ls=[weight_ls[-1], weight_ls[-1]],plot_dir=plot_dir,logy=True, title= f'{var}_{method}_comparison', bool_ratio=True, hists_cut=[hists[-1], hists[-1]],cut_ls=[0.7,0.7], cut_operator = [True, False], method_cut=method)

