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
import time
###########functions
def list_files(ls):
  ls=sorted(ls)
  min_ls=min(ls)
  max_ls=max(ls)
  assert ls== list(set(range(min_ls, max_ls+1))), 'a missing integer between the minimum and the maximum element'
  return f'{min_ls}-{max_ls}' 

def transform_dir_txt(filedir):
  # e.g. v9p1 -> v9.1
 # print(filedir.replace('p', '.'))
  return filedir.replace('p', '.')

def extract_tag(filename):
  filename=filename.split('.')
  if 'QCDskim' in filename:
    tag='bkg'
  elif 'SIGskim' in filename:
    tag='sig'
  elif filename[0]!='user':
    tag=''+filename[3]
  else: tag=''+filename[2]
  return tag
def make_dsid(bool_print=False):
  dsids=list(range(515487,515527))
  corrupt_files=[515508, 515511,515493]
  dsids=[x for x in dsids if x not in corrupt_files ]
  file_ls=[]
  for dsid in dsids:
    file_ls.append('user.ebusch.'+str(dsid)+'.root')
    if bool_print:
      mass=Label(str(dsid)).get_m(bool_num=True)
      rinv=Label(str(dsid)).get_rinv(bool_num=True)
      print(dsid, mass, rinv)
    #file_ls.append('skim3.user.ebusch.'+str(dsid)+'.root')
  filetag_ls=[extract_tag(filename=fl) for fl in file_ls]
  return file_ls, filetag_ls

## ---------- USER PARAMETERS ----------
## Model options:   "AE", "VAE", "PFN_AE", "PFN_VAE"

track_array0 = ['jet0_GhostTrack_pt', 'jet0_GhostTrack_eta', 'jet0_GhostTrack_phi', 'jet0_GhostTrack_e','jet0_GhostTrack_z0', 'jet0_GhostTrack_d0', 'jet0_GhostTrack_qOverP']
track_array1 = ['jet1_GhostTrack_pt', 'jet1_GhostTrack_eta', 'jet1_GhostTrack_phi', 'jet1_GhostTrack_e','jet1_GhostTrack_z0', 'jet1_GhostTrack_d0', 'jet1_GhostTrack_qOverP']
jet_array = ['jet1_eta', 'jet1_phi', 'jet2_eta', 'jet2_phi'] # order is important in apply_JetScalingRotation

#stdoutOrigin=sys.stdout
#sys.stdout = open(self.applydir+f'stdout.txt', 'w')
# my_variables= ["mT_jj", "jet1_pt", "jet2_pt", "jet1_Width", "jet2_Width", "jet1_NumTrkPt1000PV", "jet2_NumTrkPt1000PV", "met_met", "mT_jj_neg", "rT", "maxphi_minphi", "dphi_min", "pt_balance_12", "dR_12", "deta_12", "dphi_12", "weight", "mcEventWeight"]
#sigmoid function -> get_decoder in models.py or actually this might be remembered from the architecture 
#scaling
class Param_evaluate():
  def __init__(self,
      filedir, # e.g. '12_05_23_13_05'
      extraVars= ['mT_jj', 'weight', 'jet1_pt', 'jet2_Width'],# if this is empty
      seed=0,   max_track=80, 
      bkg_version= 'v9p2', sig_version= 'v9p2', #'v8p1'
      bkg_nevents=-1, sig_nevents=-1,
      bool_pt=False, bool_transformed=True, bool_weight=False,bool_no_scaling=True,bool_float64=False, 
      pfn_model='PFNv6', 
      arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
      vae_model='vANTELOPE',
      h5_dir='/nevis/katya01/data/users/kpark/svj-vae/h5dir/antelope/v9p2', 
      ):
      self.filedir=filedir
      self.extraVars=extraVars
      self.seed=seed
      self.max_track=max_track
      self.bkg_version=bkg_version
      self.sig_version=sig_version
      self.bkg_nevents=bkg_nevents
      self.sig_nevents=sig_nevents

      self.bool_pt=bool_pt
      self.bool_transformed=bool_transformed
      self.bool_weight=bool_weight
      self.bool_no_scaling=bool_no_scaling
      self.bool_float64=bool_float64
      self.pfn_model=pfn_model
      self.arch_dir_pfn=arch_dir_pfn
      self.vae_model=vae_model
      self.h5_dir=h5_dir

      if bool_weight:self.weight_tag='ws'
      else:self.weight_tag='nws'

      '''
      sig_read_dir: directory of the file the root file is read from 
      h5_dir: from reading sig_read_dir or bkg_read_dir, it creates h5 files of np arrays
      all_dir: has PFN or VAE model and plots
      arch_dir: subfolder of all_dir; has PFN or VAE model 
      apply_dir: subfolder of all_dir; all the new files/plots from this class will be saved here
      plot_dir: subfolder of apply_dir; e.g. score, phi2D, inputs, hist_jet1_pt plots of each signal point/other files     
      '''
      
      if self.vae_model=='': # if evaluating PFN
        self.sig_prefix=f'{self.sig_version}_'
        self.bkg_prefix=f'{self.bkg_version}_'
      else:
        self.sig_prefix=f'{self.sig_version}_{self.vae_model}_' # if evaluating ANTELOPE
        self.bkg_prefix=f'{self.bkg_version}_{self.vae_model}_' # if evaluating ANTELOPE
      self.sig_read_dir='/data/users/ebusch/SVJ/autoencoder/'+transform_dir_txt(self.sig_version)+'/'
      self.bkg_read_dir='/data/users/ebusch/SVJ/autoencoder/'+transform_dir_txt(self.bkg_version)+'/' # also could be in microntuples folder
#      self.tag= f'{self.pfn_model}_2jAvg_MM_{self.weight_tag}'
      self.all_dir=f'/nevis/katya01/data/users/kpark/svj-vae/results/grid_sept26/{self.filedir}/' # change
      self.arch_dir_vae=self.all_dir+'architectures_saved/'
      self.applydir=self.all_dir+'applydir/'
      self.plot_dir=self.applydir+'/plots_dsid/' # applydir/
      dir_ls =[self.applydir, self.plot_dir] # h5_dir, arch_dir, sig_read_dir, bkg_read_dir should already exist
      for d in dir_ls:
        if not os.path.exists(d):
          os.mkdir(d)
          print(f'made a directory: {d}')

  
  def call_functions(self,nevents, bool_weight, input_file,read_dir, bool_select_all,dsid):
    cprint(f'{self.extraVars=}, {self.pfn_model=}, {self.vae_model=}', 'red')
    print(self.arch_dir_pfn+self.pfn_model+'_graph_arch')
    graph = keras.models.load_model(self.arch_dir_pfn+self.pfn_model+'_graph_arch')
    graph.load_weights(self.arch_dir_pfn+self.pfn_model+'_graph_weights.h5')
    graph.compile()
  
    ## Load classifier model
    classifier = keras.models.load_model(self.arch_dir_pfn+self.pfn_model+'_classifier_arch')
    classifier.load_weights(self.arch_dir_pfn+self.pfn_model+'_classifier_weights.h5')
    classifier.compile()
  
    # ---------- Load graph model ----------
    if self.vae_model !='':
      encoder = keras.models.load_model(self.arch_dir_vae+self.vae_model+'_encoder_arch')
      decoder = keras.models.load_model(self.arch_dir_vae+self.vae_model+'_decoder_arch')
      vae = VAE(encoder,decoder, kl_loss_scalar=1)
      vae.get_layer('encoder').load_weights(self.arch_dir_vae+self.vae_model+'_encoder_weights.h5')
      vae.get_layer('decoder').load_weights(self.arch_dir_vae+self.vae_model+'_decoder_weights.h5')
      vae.compile(optimizer=keras.optimizers.Adam())
   
    ## Load history
    # with open(arch_dir+ae_model+'_history.json', 'r') as f:
    #     h = json.load(f)
    # print(h, type(h))
    print ('Loaded model')
    cprint(f'{self.extraVars=}', 'magenta')
    bkg2, mT_bkg, _, _, _, _ = getTwoJetSystem(nevents=nevents,input_file=input_file,
        track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
        bool_weight=bool_weight,  extraVars=self.extraVars, plot_dir=self.plot_dir,seed=self.seed,max_track=self.max_track, bool_pt=self.bool_pt, h5_dir=self.h5_dir, bool_select_all=bool_select_all, read_dir=read_dir)
  
    print(mT_bkg.shape)
    scaler = load(self.arch_dir_pfn+self.pfn_model+'_scaler.bin')
    bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
    
    if mT_bkg[:,1].any():
      bool_weight=True
    else: bool_weight=False # if data, not drawing with weights
  
    plot_vectors(bkg2,bkg2,tag_file='ANTELOPE_'+str(dsid)+'_', tag_title=f' (ANTELOPE) {str(dsid)}', plot_dir=self.plot_dir, bool_sig_on=False, labels=[str(dsid)])# change
    plot_single_variable([mT_bkg[:,2]],h_names= [input_file],weights_ls=[mT_bkg[:,1]], tag_title= f'leading jet pT  {str(dsid)}', plot_dir=self.plot_dir,logy=True, tag_file='jet1_pt_'+str(dsid), bool_weight=bool_weight)
    plot_single_variable([mT_bkg[:,0]],h_names= [input_file],weights_ls=[mT_bkg[:,1]], tag_title= f'{self.extraVars[0]} {str(dsid)} (weighted)', plot_dir=self.plot_dir,logy=True, tag_file='mT_jj_'+str(dsid), bool_weight=bool_weight)
  
    phi_bkg = graph.predict(bkg2)
    vae_min_dict, vae_max_dict, pfn_min_dict,pfn_max_dict={},{},{},{}
  
    pfn_min_dict['09_26_23_10_38']= 0
    pfn_max_dict['09_26_23_10_38']= 204.44198608398438  # got from Final Max in /nevis/katya01/data/users/kpark/svj-vae/results/grid_sept26/09_26_23_10_38/stdout.txt
    # each event has a pfn score
    ## Classifier loss
    if self.vae_model =='': # if PFN 
      pred_phi_bkg = classifier.predict(phi_bkg)
      bkg_loss = pred_phi_bkg[:,1]
    else: # if PFN + AE
      ## Scale phis - values from v1 training
      if self.bool_float64:
        phi_bkg = phi_bkg.astype('float64')
      eval_min = 0.0
      eval_max = 109.87523
      #eval_max = np.amax(phi_bkg)
      #eval_min = np.amin(phi_bkg)
  
      if not self.bool_no_scaling :# scaling 
        eval_min = pfn_min_dict[self.filedir]
        eval_max = pfn_max_dict[self.filedir]
        phi_bkg = (phi_bkg - eval_min)/(eval_max-eval_min)
      
      plot_phi(phi_bkg,tag_file='PFN_phi_input_'+str(dsid), tag_title=f'{str(dsid)} Input', plot_dir=self.plot_dir)
      pred_phi_bkg = vae.predict(phi_bkg)['reconstruction']
    cprint(f'{mT_bkg.shape=} {self.extraVars=}', 'red')
    return phi_bkg, pred_phi_bkg, mT_bkg,  vae
   
  def write_hdf5(self,dsid,phi_bkg, pred_phi_bkg, mT_bkg, vae,  prefix,outputfolder, subset=0, split_nevents=0, bool_split=False):
    if bool_split: outputpath=self.applydir+outputfolder+f'{prefix}{dsid}_log10_{subset}'+'.hdf5' 
    else: outputpath=self.applydir+outputfolder+f'{prefix}{dsid}_log10'+'.hdf5' 

    # check if the file exists; if so, read
    if os.path.exists(outputpath):
      with h5py.File(outputpath,'r') as f:
        rec_bkg_each = f.get('data')[:]

      return rec_bkg_each

    # doesn't exit, so continue
    if self.vae_model =='':
      print('not writing hdf5 in write_hdf5 function because using VAE')
    else:
      print(phi_bkg.shape, pred_phi_bkg.shape, mT_bkg.shape)
      # only selects some indices of phi_bkg, pred_phi_bkg, mT_bkg to evaluate
      if bool_split:
        phi_bkg, pred_phi_bkg, mT_bkg= phi_bkg[subset*split_nevents: (subset+1)*split_nevents, :]  , pred_phi_bkg[subset*split_nevents: (subset+1)*split_nevents, :],   mT_bkg[subset*split_nevents: (subset+1)*split_nevents, :]
        print('after split',phi_bkg.shape, pred_phi_bkg.shape, mT_bkg.shape)
       
      bkg_loss={}
      # ## AE loss
      bkg_loss['mse'] = np.array(keras.losses.mse(phi_bkg, pred_phi_bkg))
      bkg_loss['multi_reco'], bkg_loss['multi_kl'],  bkg_loss['multi_mse']=get_multi_loss_each(vae, phi_bkg)
      methods=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
      new_methods=[]
      if self.bool_transformed:
        old_methods=methods.copy() # essential that new_methods and methods are separate b/c otherwise, will loop through methods that are already transformed
        for method in old_methods:
          new_method=f'{method}_transformed_log10_sig'
          print(f'{method=}, {new_method=}')
          loss=np.log10(bkg_loss[method])
          loss_transformed_bkg = 1/(1 + np.exp(-loss))
          bkg_loss[new_method] =loss_transformed_bkg
          new_methods.append(new_method)

    for method in new_methods: # transformed
      plot_score(bkg_loss[method], np.array([]), False, xlog=False, tag_file=self.vae_model+f'_{method}_{str(dsid)}', tag_title=self.vae_model+f' {method} {str(dsid)}', plot_dir=self.plot_dir, bool_pfn=False, labels=[str(dsid)]) # anomaly score 
  
    for method in old_methods:
      plot_score(bkg_loss[method][bkg_loss[method]>0], np.array([]),False, xlog=True, tag_file=self.vae_model+'_pos'+f'_{method}_{str(dsid)}', tag_title=self.vae_model + ' (score > 0)'+f' {method} {str(dsid)}', plot_dir=self.plot_dir, bool_pfn=False, labels=[str(dsid)]) # anomaly score

    newVars= methods+ new_methods
    newVars+=self.extraVars
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

    cprint(f'{newVars=}, {self.extraVars=},{mT_bkg.shape=}, {bkg_loss["mse"].shape=}, ', 'yellow') 
    ds_dt = np.dtype({'names':newVars,'formats':[(float)]*len(newVars)})
    rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)
    last_substring=f'{outputpath.split("/")[-1]}'
    new_dir=outputpath.replace(last_substring, '')
    if not os.path.exists(new_dir):
      os.mkdir(new_dir)
      print('made ', new_dir)
    
    with h5py.File(outputpath,'w') as f:
      dset = f.create_dataset('data',data=rec_bkg)
    return rec_bkg

  def add_column(self,dsid,outputfolder, newfolder, columns, nevents, input_file, bool_weight, read_dir, bool_split, input_tag,output_tag,prefix, bool_select_all):
    if output_tag!='':
       if output_tag[0]!='_': output_tag='_'+output_tag # if output_tag is not an empty string and doesn't contain '_' in the beginning, add it
#    if bool_split:
#      inputpath=self.applydir+outputfolder+f'{prefix}{dsid}_log10_{subset}'+'.hdf5' 
#      outputpath=self.applydir+newfolder+f'{prefix}{dsid}_log10_{subset}{filetag}'+'.hdf5' 
    inputpath=self.applydir+outputfolder+f'{prefix}{dsid}_log10{input_tag}'+'.hdf5' 
    outputpath=self.applydir+newfolder+f'{prefix}{dsid}_log10{output_tag}'+'.hdf5' 

    bkg2, mT_bkg, _, _, _, _ = getTwoJetSystem(nevents=nevents,input_file=input_file,
        track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
        bool_weight=bool_weight,  extraVars=self.extraVars+columns, plot_dir=self.plot_dir,seed=self.seed,max_track=self.max_track, bool_pt=self.bool_pt, h5_dir=self.h5_dir, bool_select_all=bool_select_all, read_dir=read_dir, h5tag='_new')
   
    cprint(mT_bkg.shape, 'yellow') 
    with h5py.File(inputpath,'r') as f:
      bkg_loss = f.get('data')[:]

    # check if the jet1_pt from this hdf5 and the jet1_pt from inputpath are the same
    cprint(f'{(bkg_loss["mT_jj"].flatten()==mT_bkg[:,0])=}, {bkg_loss["mT_jj"].flatten()}, {mT_bkg[:,0]=}', 'red')
    print(f'{mT_bkg[:,0][:,None].shape=}, {bkg_loss.shape=}')
    bkg_loss_names=list(bkg_loss.dtype.names)
    print(f'{bkg_loss_names=}')
    for i, key in enumerate(bkg_loss_names):
      if i==0:
        new_bkg=bkg_loss[key]
        if len(new_bkg.shape)<2:new_bkg=new_bkg[:,None] 
        #new_bkg=bkg_loss[key][:,None] # try this if there are dimension prob
        print(f'{new_bkg.shape=}')
      else:  
        print(f'{new_bkg.shape=}')
        new_bkg_each=bkg_loss[key]
        if len(new_bkg_each.shape)<2:new_bkg_each=new_bkg_each[:,None] 
        new_bkg=np.concatenate((new_bkg,new_bkg_each), axis=1)
    print(f'{new_bkg.shape=},{mT_bkg.shape=},{mT_bkg[:,0].shape=}, {mT_bkg[:,0][:,None].shape=},{bkg_loss_names}, {columns}, {new_bkg[:,-2]=}')
    for i, col in enumerate(columns): # i was adding mT_jj as jet2_Width
      new_bkg_each=mT_bkg[:,i+len(self.extraVars)]
      if len(new_bkg_each.shape)<2:new_bkg_each=new_bkg_each[:,None]
      new_bkg=np.concatenate((new_bkg,new_bkg_each), axis=1) 
      #new_bkg=np.concatenate((new_bkg,mT_bkg[:,i+len(self.extraVars)][:,None]), axis=1) 
   # plot_single_variable_ratio([new_bkg[:,-1]],h_names= [input_file],weights_ls=[new_bkg[:,-3]], title= f'{self.extraVars[-1]} {str(dsid)} (weighted)', plot_dir=self.plot_dir,logy=True, bool_ratio=False)
    all_keys=bkg_loss_names+columns
    ds_dt = np.dtype({'names':all_keys,'formats':[(float)]*len(all_keys)})
    save_bkg = np.rec.array(new_bkg, dtype=ds_dt)
    rec_bkg= save_bkg
      # it is important that the data from the hdf5 file doesn't get expanded to a new shape ( no calling with [:, None] to be compatible with the new column datashapes)
    print(f'{all_keys},{new_bkg.shape=}, {save_bkg.shape=}, {rec_bkg.shape=}, {rec_bkg.dtype.itemsize=}', 'yellow')
    with h5py.File(outputpath,'w') as f:
      dset = f.create_dataset('data',data=rec_bkg)
    print(f'{outputpath=}', 'make sure this is jet2_Width', f'{self.extraVars[-1]}')
    if 'Data' in dsid or 'data' in dsid or 'DATA' in dsid:
      bool_plot_weight=False
    else:  
      bool_plot_weight=True
    plot_single_variable([rec_bkg['mT_jj']],h_names= [f'{dsid}'],weights_ls=[rec_bkg['weight']], tag_title= f'mT_jj {str(dsid)} (weighted)', plot_dir=self.plot_dir,logy=True, tag_file=f'mT_jj_'+str(dsid), bool_weight=bool_plot_weight, bool_show=True) 
    plot_single_variable([rec_bkg['jet1_pt']],h_names= [f'{dsid}'],weights_ls=[rec_bkg['weight']], tag_title= f'jet1_pt {str(dsid)} (weighted)', plot_dir=self.plot_dir,logy=True, tag_file=f'jet1_pt_'+str(dsid), bool_weight=bool_plot_weight, bool_show=True) 
#    plot_single_variable([mT_bkg[:,-1]],h_names= [input_file],weights_ls=[mT_bkg[:,1]], tag_title= f'{self.extraVars[-1]} {str(dsid)} (weighted)', plot_dir=self.plot_dir,logy=True, tag_file=f'{self.extraVars[-1]}_'+str(dsid), bool_weight=bool_weight)
    return rec_bkg

  """
  Here we evaluate on signal files 
  """
  def eval_sig(self, bool_split=False, columns= ['jet2_Width'], outputfolder='/hdf5_orig/',bool_select_all=True, bool_add=False):
    #bool_select_all= True for most of the times
    file_ls,filetag_ls=make_dsid()
    for fl in file_ls:
      dsid=fl.split('.')[-2]
      print('*'*30)
      cprint(f'{fl},{dsid=}', 'green')
      """
        Here you can add a column to a hdf5 file that was already processed and has new columns  
      """ 
      if bool_add:
        newfolder='/hdf5_jet2_width/'
        input_tag=''
        output_tag='_jet2_width'
        self.add_column(dsid=dsid,outputfolder=outputfolder,newfolder=newfolder, columns=columns, nevents=self.sig_nevents, input_file=fl, bool_weight=self.bool_weight, read_dir=self.sig_read_dir,bool_split=bool_split, input_tag=input_tag,output_tag=output_tag, prefix=self.sig_prefix, bool_select_all=bool_select_all) 
        return 
      h5path1=self.applydir+outputfolder+f'{self.sig_prefix}{dsid}_log10'+'.hdf5' 
      h5path2=self.applydir+'/hdf5_jet2_width/'+f'{self.sig_prefix}{dsid}_log10'+'.hdf5' 
      if  os.path.exists(h5path1 ): # and (dsid !=515429):
        with h5py.File(h5path1,'r') as f:
          dset = f.get('data')[:]
      elif os.path.exists(h5path2):
        with h5py.File(h5path2,'r') as f:
          dset = f.get('data')[:]
      else:   
        phi_bkg, pred_phi_bkg, mT_bkg, vae =  self.call_functions(nevents=self.sig_nevents,  bool_weight=self.bool_weight, input_file=fl,read_dir=self.sig_read_dir, bool_select_all=bool_select_all,dsid=dsid)
        rec_bkg_each=self.write_hdf5(dsid=dsid, phi_bkg=phi_bkg, pred_phi_bkg=pred_phi_bkg, mT_bkg=mT_bkg,  vae=vae, prefix=self.sig_prefix, bool_split=bool_split, outputfolder=outputfolder)
    return 
  """
  Here we evaluate on background files 
  #output_self.h5path=self.applydir+'/'+'hdf5_jet2_width'+'/'+f'{bkg_prefix}{dsid}_log10_0-67_jet2_width'+".hdf5" 
  """
  def eval_bkg(self, bool_split=True, columns= ['jet2_Width'], outputfolder='/hdf5_orig/', bkg_file='skim0.user.ebusch.bkgAll.root', n_file=68,bool_select_all=True, bool_add=False):
    #bkg_file='user.ebusch.dataAll.root' #bkg_file='user.ebusch.515503.root'#bkg_file='skim0.user.ebusch.bkgAll.root' #bkg_file='skim0.user.ebusch.QCDskim.root'
    dsid=bkg_file.split('.')[-2]
    #check in original folder (self.outputfolder)
    h5path1=self.applydir+outputfolder+f'{self.bkg_prefix}{dsid}_log10'+'.hdf5' 
    h5path2=self.applydir+'/hdf5_jet2_width/'+f'{self.bkg_prefix}{dsid}_log10'+'.hdf5' 
    if 'Data' in dsid or 'data' in dsid or 'DATA' in dsid:
      bool_plot_weight=False      
    else:  bool_plot_weight=True
    if bool_add:
      newfolder='/hdf5_jet2_width/'
      input_tag='_0-67'
      output_tag='_0-67_jet2_width'
      self.add_column(dsid=dsid,outputfolder=outputfolder,newfolder=newfolder, columns=columns, nevents=self.sig_nevents, input_file=bkg_file, bool_weight=self.bool_weight, read_dir=self.bkg_read_dir, bool_split=bool_split,input_tag=input_tag,output_tag=output_tag, prefix=self.bkg_prefix, bool_select_all=bool_select_all)
      return 
    if  os.path.exists(h5path1):
      with h5py.File(h5path1,'r') as f:
        dset = f.get('data')[:]
    
      new_pt=dset['jet1_pt']
      new_weight=dset['weight']

      plot_single_variable([new_pt],h_names= [bkg_file],weights_ls=[new_weight], tag_title= f'leading jet pT  {str(dsid)}', plot_dir=self.applydir+'/plots_dsid/',logy=True, tag_file='new_jet1_pt_'+str(dsid), bool_weight=bool_plot_weight)
    elif  os.path.exists(h5path2):
      with h5py.File(h5path2,'r') as f:
        dset = f.get('data')[:]

    
      plot_single_variable([new_pt],h_names= [bkg_file],weights_ls=[new_weight], tag_title= f'leading jet pT  {str(dsid)}', plot_dir=self.applydir+'/plots_dsid/',logy=True, tag_file='new_jet1_pt_'+str(dsid), bool_weight=bool_plot_weight)
    else:
    # if it doesn't exist, 
      # decide how many loops by choosing n_file and how many events per each file for split_nevents
      split_nevents=100000
      #n_file=13 #  n_file = phi_bkg.shape[0]//split_nevents +1  #  n_file=47 #n_file=1 
      #n_file=68
      ls_files=[]
      bool_new=True
      for subset in range(0,n_file):
        outputpath=self.applydir+outputfolder+f'{self.bkg_prefix}{dsid}_log10_{subset}'+'.hdf5'
        if os.path.exists(outputpath):
          with h5py.File(outputpath,'r') as f:
            rec_bkg_each = f.get('data')[:]
        else: 
          if bool_new: 
            phi_bkg, pred_phi_bkg, mT_bkg,  vae =  self.call_functions(nevents=self.bkg_nevents,  bool_weight=self.bool_weight, input_file=bkg_file, read_dir=self.bkg_read_dir, bool_select_all=bool_select_all, dsid=dsid)
            bool_new=False
          rec_bkg_each=self.write_hdf5(dsid=dsid, phi_bkg=phi_bkg, pred_phi_bkg=pred_phi_bkg, mT_bkg=mT_bkg,  vae=vae, prefix=self.sig_prefix, bool_split=bool_split, outputfolder=outputfolder)
          rec_bkg_each=self.write_hdf5(dsid=dsid,phi_bkg=phi_bkg, pred_phi_bkg=pred_phi_bkg, mT_bkg=mT_bkg,  vae=vae,prefix=self.bkg_prefix, bool_split=bool_split,  outputfolder=outputfolder,subset=subset, split_nevents=split_nevents)
        # concatenate and write
        if subset==0:
          rec_bkg=rec_bkg_each
        else:
          rec_bkg= np.append(rec_bkg, np.array(rec_bkg_each , dtype=rec_bkg_each.dtype))
    
        ls_files.append(subset)
    
      ls_files=list_files(ls_files)
      cprint(ls_files, 'yellow')
      combined_outputpath=f"{h5path1.split('.hdf5')[-2]}_{ls_files}.hdf5"
      with h5py.File(combined_outputpath,'w') as f:
        dset = f.create_dataset('data',data=rec_bkg[:,None]) # has to have [:, None] to be compatible with the original hdf5 format

      print(f'{combined_outputpath=}')  
      """
      Here you can add a column to a hdf5 file that was already processed and has new columns  
      """ 
    return
  def scan(self, bkg_file='bkgAll_log10_0-67_jet2_width.hdf5',outputfolder='/hdf5_orig/'):
    title=f'track={self.max_track}'
    #bkg_file='bkgAll_log10_0-67_jet2_width.hdf5'#bkg_file='dataAll_log10_jet2_width.hdf5'
    #bkg_file='dataAll_log10_0-12.hdf5' #bkg_file='bkgAll_log10_0-67_jet2_width.hdf5'#bkg_file='dataAll_log10_jet2_width.hdf5'
    score_cut_dict={}
    '''
    keys=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
    keys=[f'{key}_transformed_log10_sig' for key in keys ]
    '''
    score_cut_dict[self.filedir]={'multi_kl_transformed_log10_sig':0.72, 'multi_mse_transformed_log10_sig':0.573, 'multi_reco_transformed_log10_sig':0.7}
    keys=list(score_cut_dict[self.filedir].keys())
    for key in keys:
      print(self.filedir,key)
      #check_yield( title='Background', all_dir=all_dir, bkg_prefix=self.bkg_prefix, filename=bkg_file,key=key)
    #  correlation_plots( title='Background', all_dir=all_dir, bkg_prefix=bkg_prefix, filename=bkg_file,key=key)
      outputdir=self.applydir+outputfolder # or choose newfolder if used add_columns()

      grid_scan(title, outputdir=outputdir, sig_prefix=self.sig_prefix,bkg_prefix=self.bkg_prefix, bkg_file=bkg_file,key=key)
      grid_s_sqrt_b(score_cut_dict[self.filedir][key], outputdir=outputdir,bkg_scale=5, sig_prefix=self.sig_prefix,bkg_prefix=self.bkg_prefix,bkg_file=bkg_file, title=title,cms=False, key=key)


      #score=getSignalSensitivityScore(bkg_loss, sig_loss)
      #print('95 percentile score = ',score)

    return
  
  def compare_hist(self,method):
    """
    Here, you can plot histograms that compare CR/VR/SR and other histograms
    """
    '''
    filedir_ls=[]
    read bkg files from filedir_ls
    bkgpath for each
    '''
    hs={}
    #var_ls=[method, 'weight']
    var_ls=[method, 'mT_jj', 'jet2_Width', 'weight']
    #for dsid in dsids:
    path=self.applydir+'/'+'hdf5_jet2_width'+'/'+f"{self.bkg_prefix}bkgAll_log10_0-67_jet2_width.hdf5"
    with h5py.File(path,"r") as f:
      data = f.get('data')[:]

        # concatenate here
    for var in var_ls:    
      if var =='weight' and not( data['weight'].any()): # if it's weight,  
        h=data['weight']
      else: #if array contains only zeros e.g. if it's weight of data
        h=np.ones(data['weight'].shape) 
             
      hs[var]=h

    #plot_single_variable_ratio(hists, h_names, weights_ls,title,density_top=True, logy=False, len_ls=[],  plot_dir="", bool_ratio=True, hists_cut=[], cut_ls=[], cut_operator=[], method_cut=[], bin_min=-999,bin_max=-999, bool_plot=True)
    var='multi_reco_transformed_log10_sig'
    plot_single_variable_ratio([hs[var][0],hs[var][1],hs[var][2]],h_names=['(0%)', '(1%)', '(10%)'], weights_ls=[hs['weight'][0],hs['weight'][-1], hs['weight'][2]],plot_dir=self.plot_dir,logy=True, title= f'mT_jj_{method}_signal_injection', bool_ratio=True, bin_min=0.6, bin_max= 1)
    '''
    plot mT_jj for all/ CR/VR/SR 
    plot_single_variable_ratio([hs['mT_jj'][-1],hs['mT_jj'][-1],hs['mT_jj'][-1], hs['mT_jj'][-1]],h_names=['(All)','(CR)', '(VR)', '(SR)'],weights_ls=[hs['weight'][-1],hs['weight'][-1], hs['weight'][-1], hs['weight'][-1]],plot_dir=self.plot_dir,logy=True, title= f'mT_jj_{method}_comparison_region', bool_ratio=True, hists_cut=[[hs['jet2_Width'][-1], hs[method][-1]],[hs['jet2_Width'][-1], hs[method][-1]],[ hs['jet2_Width'][-1], hs[method][-1]],[ hs['jet2_Width'][-1], hs[method][-1]]],cut_ls=[[0,0],[0.05, 0], [0.05, 0.7],[0.05, 0.7]], cut_operator = [[True, True],[False,True], [True,  False],[True, True]] , method_cut=[['jet2_Width', method],['jet2_Width', method], ['jet2_Width', method], ['jet2_Width', method]], bin_min=1000, bin_max= 5000) 
    '''
if __name__=="__main__":
   # change
  #filedir='12_05_23_13_05' # trained with data and signal injection (515503) 10%
  #filedir='12_02_23_09_19' # trained with data and signal injection (515503) 1%
  filedir='10_08_23_04_08_cp' # trained with data
  #filedir='09_26_23_10_38'
  #filedir='09_27_23_01_32'
  # filedir=
  method='multi_reco_transformed_log10_sig'
  param1=Param_evaluate(filedir=filedir, extraVars=['mT_jj', 'weight', 'jet1_pt'])
  start= time.time()
  #param1.eval_sig(bool_add=True)
  #param1.eval_bkg(bool_add=True)
#  param1.scan(outputfolder='/hdf5_jet2_width/')
#  param1.scan(bkg_file='bkgAll_log10_0-67.hdf5')
  param1.scan(bkg_file='dataAll_log10.hdf5')
  param1.compare_hist(method)
  end= time.time()
  print('time elapsed:', f' {round(end-start, 2)}s or  {round((end-start)/3600,2)}h')


''' 
errors
1)     new = obj.view(dtype)
ValueError: When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.
problem: the dimension of the arrays that go into hdf5 do not match with the number of variables given
solution: go to eval_helper.py and check getTwoJet... function and see if the hdf5 it's reading it from has the expected dimension as expected
2)   File "mtrand.pyx", line 937, in numpy.random.mtrand.RandomState.choice
ValueError: probabilities are not non-negative
problem: negative values in weight -> get_weighted_elements_h5 in antelope_h5eval.py causes issues
'''
