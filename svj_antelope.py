import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import json
from joblib import dump, load
from models import *
from root_to_numpy import *
from plot_helper import *
from eval_helper import *
import time
from svj_pfn import Param

import pandas as pd
# Example usage
class Param_ANTELOPE(Param):
  def __init__(self,  
      arch_dir_pfn,
      arch_dir="architectures_saved/",print_dir='',plot_dir='plots/',h5_dir='h5dir/jul28/',
      pfn_model='PFN', vae_model='vANTELOPE', bkg_events=200000, sig_events=20000,
      num_elements=100, element_size=7, encoding_dim=32, latent_dim=12, phi_dim=64, nepochs=50, n_neuron=75, learning_rate=0.001,
      nlayer_phi=3, nlayer_F=3,
      max_track=80,
      batchsize_pfn=512,
#      batchsize_pfn=500,
      batchsize_vae=32, # batchsize_pfn=500 -> 512 or any power of 2
      bool_pt=False,
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim0.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'],seed=0 ):
      #changeable: encoding_dim,latent_dim, nepochs, learning_rate, bkg_events, sig_events
    """
    arch_dir,print_dir,plot_dir,h5_dir,
      pfn_model, vae_model, bkg_events, sig_events,
      num_elements, element_size, encoding_dim, latent_dim, phi_dim, nepochs, n_neuron, learning_rate,
      nlayer_phi, nlayer_F,
      max_track,
      batchsize_pfn,
      batchsize_vae, 
      bool_pt,
      sig_file, bkg_file,  bool_weight, extraVars,seed, arch_dir_pfn):
    """
      
    """
    super().__init__( arch_dir="architectures_saved/",print_dir='',plot_dir='plots/',h5_dir='h5dir/jul28/',
      pfn_model='PFN', vae_model='vANTELOPE', bkg_events=500000, sig_events=500000,
      num_elements=100, element_size=7, encoding_dim=32, latent_dim=12, phi_dim=64, nepochs=100, n_neuron=75, learning_rate=0.001,
      nlayer_phi=3, nlayer_F=3,
      max_track=80,
      batchsize_pfn=512,
#      batchsize_pfn=500,
      batchsize_vae=32, # batchsize_pfn=500 -> 512 or any power of 2
      bool_pt=False,
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim3.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=[],seed=0)

    """
    super().__init__( arch_dir,print_dir,plot_dir,h5_dir,
      pfn_model, vae_model, bkg_events, sig_events,
      num_elements, element_size, encoding_dim, latent_dim, phi_dim, nepochs, n_neuron, learning_rate,
      nlayer_phi, nlayer_F,
      max_track,
      batchsize_pfn,
      batchsize_vae,
      bool_pt,
      sig_file, bkg_file,  bool_weight, extraVars,seed)

    self.arch_dir_pfn=arch_dir_pfn

  def save_info(self, bool_csv=True): # always saves info.txt -> info.csv is optional
    if bool_csv: # save in csv
      info_dict=[self.__dict__]
      print(info_dict) # print all the attributes as a dictionary
      print('printing in', self.print_dir)
      df= pd.DataFrame.from_dict(info_dict)
      df.to_csv(self.print_dir+f'info.csv', index=False) 
     # save in textfile
    text=f'{vars(param1)}' # print all attributes of the class as dictionary
    print(text)
    print('printing in', self.print_dir)
    with open(self.print_dir+f'info.txt', 'w') as f: 
      f.write(text)
#    return f'saved info in {self.print_dir}\n {text}'

    return f'saved info in {self.print_dir}\n {df}'


  def train_vae(self):
    cprint(self.arch_dir_pfn+self.pfn_model+'_graph_arch', 'yellow')
    graph = keras.models.load_model(self.arch_dir_pfn+self.pfn_model+'_graph_arch')
    graph.load_weights(self.arch_dir_pfn+self.pfn_model+'_graph_weights.h5')
    graph.compile()

    track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e","jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
    track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e","jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
    jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"] # order is important in apply_JetScalingRotation

    bool_weight_sig=False

    
    sig, mT_sig, sig_sel, jet_sig, sig_in0, sig_in1 = getTwoJetSystem(nevents=self.sig_events,input_file=self.sig_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight_sig,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed,max_track=self.max_track, bool_pt=self.bool_pt,h5_dir=self.h5_dir,       read_dir='/data/users/ebusch/SVJ/autoencoder/v8.1/')    
    bkg, mT_bkg, bkg_sel, jet_bkg,bkg_in0, bkg_in1 = getTwoJetSystem(nevents=self.bkg_events,input_file=self.bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=self.bool_weight,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed, max_track=self.max_track,bool_pt=self.bool_pt,h5_dir=self.h5_dir,
      read_dir='/data/users/ebusch/SVJ/autoencoder/v9.1/') 
      #read_dir='/data/users/ebusch/SVJ/autoencoder/v9.1/') 
    
    scaler = load(self.arch_dir_pfn+self.pfn_model+'_scaler.bin')
    bkg2,_ = apply_StandardScaling(bkg,scaler,False) # change
    sig2,_ = apply_StandardScaling(sig,scaler,False) # change
    plot_vectors(bkg2,sig2,tag_file="ANTELOPE", tag_title=" (ANTELOPE)", plot_dir=self.plot_dir)# change
    
    phi_bkg = graph.predict(bkg2)
    phi_sig = graph.predict(sig2)
 
    print(phi_bkg.shape, mT_bkg.shape)
    plot_1D_phi(phi_bkg, phi_sig,labels=['QCD', 'sig'], plot_dir=self.plot_dir, tag_file=self.pfn_model, tag_title=self.pfn_model)
    print(len(phi_bkg))
    #plot_score(phi_bkg[:,11], phi_sig[:,11], False, False, "phi_11_raw")
    
    #phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0], random_state=42) # no info on labels e.g. y_train or y_test
    phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0])
    print('before',phi_evalb.shape, phi_testb.shape) 
    phi_evalb_idx, phi_testb_idx, _, _ = train_test_split(np.arange(len(phi_bkg)), phi_bkg, test_size=sig2.shape[0])
    phi_evalb, phi_testb, mT_evalb, mT_testb= phi_bkg[phi_evalb_idx, :], phi_bkg[phi_testb_idx, :], mT_bkg[phi_evalb_idx,:], mT_bkg[phi_testb_idx, :]
    print('idx',phi_evalb_idx, phi_testb_idx)
    print('after',phi_evalb.shape, phi_testb.shape,  mT_evalb.shape, mT_testb.shape)
    
    plot_single_variable([mT_evalb[:,2], mT_testb[:,2]],h_names= ['training and validation', 'test'],weights_ls=[mT_evalb[:,1], mT_testb[:,1]], tag_title= 'leading jet pT (QCD)', plot_dir=self.plot_dir,logy=True, tag_file='jet1_pt')
    plot_single_variable([mT_evalb[:,3], mT_testb[:,3]],h_names= ['training and validation', 'test'],weights_ls=[mT_evalb[:,1], mT_testb[:,1]], tag_title= 'subleading jet pT (QCD)', plot_dir=self.plot_dir, logy=True, tag_file='jet2_pt')
    plot_phi(phi_evalb,tag_file="PFN_phi_train_raw",tag_title="Train") # change
    plot_phi(phi_testb,tag_file="PFN_phi_test_raw",tag_title="Test")
    plot_phi(phi_sig,tag_file="PFN_phi_sig_raw", tag_title="Signal")
    
    eval_max = np.amax(phi_evalb)
    eval_min = np.amin(phi_evalb)
    sig_max = np.amax(phi_sig)
    print("Min: ", eval_min)
    print("Max: ", eval_max)
    if (sig_max > eval_max): eval_max = sig_max
    print("Final Max: ", eval_max)

    
    phi_evalb = (phi_evalb - eval_min)/(eval_max-eval_min)
    phi_testb = (phi_testb - eval_min)/(eval_max-eval_min)
    phi_sig = (phi_sig - eval_min)/(eval_max-eval_min)
    
    #phi_evalb, phi_scaler = apply_StandardScaling(phi_evalb)
    #phi_testb, _ = apply_StandardScaling(phi_testb,phi_scaler,False)
    #phi_sig, _ = apply_StandardScaling(phi_sig,phi_scaler,False)
    
    plot_phi(phi_evalb,tag_file="PFN_phi_train_scaled",tag_title="Train Scaled") # change
    plot_phi(phi_testb,tag_file="PFN_phi_test_scaled",tag_title="Test Scaled")
    plot_phi(phi_sig,tag_file="PFN_phi_sig_scaled", tag_title="Signal Scaled")
    
    
    vae = get_vae(self.phi_dim,self.encoding_dim,self.latent_dim, self.learning_rate)
   
    # validation data set manually
    # Prepare the training dataset
    idx = np.random.choice( np.arange(len(phi_evalb)), size= round(.2 *len(phi_evalb)) , replace=False) # IMPT that replace=False so that event is picked only once
    idx = np.sort(idx) 
    # Prepare the validation dataset
    phi_evalb_val = phi_evalb[idx, :] 
    phi_evalb_train = np.delete(phi_evalb, idx) # doesn't modify input array 
    print(f'{phi_evalb_val.shape=}, {phi_evalb_train=}')     
    mT_evalb_val = mT_evalb[idx, :] 
    mT_evalb_train = np.delete(mT_evalb, idx, axis=0) # doesn't modify input array 
    plot_single_variable([mT_evalb_train[:,2], mT_evalb_val[:,2]],h_names= ['training', 'validation'],weights_ls=[mT_evalb_train[:,1], mT_evalb_val[:,1]], tag_title= 'leading jet pT (QCD)', plot_dir=self.plot_dir, logy=True, tag_file='jet1_pt')
    plot_single_variable([mT_evalb_train[:,3], mT_evalb_val[:,3]],h_names= ['training', 'validation'],weights_ls=[mT_evalb_train[:,1], mT_evalb_val[:,1]], tag_title= 'subleading jet pT (QCD)', plot_dir=self.plot_dir, logy=True, tag_file='jet2_pt')
    
    phi_evalb_train, phi_evalb_val, _, _ = train_test_split(phi_evalb, phi_evalb, test_size=round(.2 *len(phi_evalb)))
    h2 = vae.fit(phi_evalb_train, 
    #h2 = vae.fit(phi_evalb, 
        epochs=self.nepochs,
        batch_size=self.batchsize_vae,
        validation_data=(phi_evalb_val, phi_evalb_val),
        #validation_split=0.2,
        verbose=1)
    
    # # simple ae
    #ae.save(arch_dir+vae_model)
    #print("saved model"+ arch_dir+vae_model)
    
    
    #complex ae
    vae.get_layer('encoder').save_weights(self.arch_dir+self.vae_model+'_encoder_weights.h5')
    vae.get_layer('decoder').save_weights(self.arch_dir+self.vae_model+'_decoder_weights.h5')
    vae.get_layer('encoder').save(self.arch_dir+self.vae_model+'_encoder_arch')
    vae.get_layer('decoder').save(self.arch_dir+self.vae_model+'_decoder_arch')
    #with open(arch_dir+vae_model+"8.1_history.json", "w") as f:
    #    json.dump(h2.history, f)
    latent_bkg_test=vae.get_layer('encoder').predict(phi_testb)
    latent_bkg_train=vae.get_layer('encoder').predict(phi_evalb_train)
    latent_bkg_val=vae.get_layer('encoder').predict(phi_evalb_val)
    latent_sig=vae.get_layer('encoder').predict(phi_sig)

    #latent_bkg_test is a list but latent_bkg_test[0] is a numpy array
    latent_bkg_test, latent_bkg_train, latent_bkg_val, latent_sig=np.array(latent_bkg_test), np.array(latent_bkg_train), np.array(latent_bkg_val), np.array(latent_sig)
    try: 
      print('print 3')
      print(f'{latent_bkg_test.shape=}')
    except: 
      print('print 4')
      print(f'{len(latent_bkg_test)}')
    
    for k in range(len(latent_bkg_test)):

      plot_1D_phi(latent_bkg_test[k,:,:], latent_sig[k, :,:], labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_{k}', tag_title=self.vae_model)
      plot_1D_phi(latent_bkg_train[k,:,:], latent_bkg_val[k,:,:], labels=['train QCD', 'validation QCD'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'train_val_{k}', tag_title=self.vae_model)



#    plot_1D_phi(latent_bg_test, latent_sig labels=['test QCD' 'SIG'] plot_dir=self.plot_dir tag_file=self.vae_model+f'qcd_sig' tag_title=self.vae_model)
#    plot_1D_phi(latent_bkg_train, latent_bkg_val, labels=['train QCD', 'validation QCD'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'train_val', tag_title=self.vae_model)

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch 
    plot_loss(h2, loss='loss', tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
    
    #plot_loss(h2, vae_model, "kl_loss")
    #plot_loss(h2, vae_model, "reco_loss")
    
    #2. Get loss
    #bkg_loss, sig_loss = get_single_loss(ae, phi_testb, phi_sig)
#    """
    pred_phi_bkg = vae.predict(phi_testb)['reconstruction']
    pred_phi_sig = vae.predict(phi_sig)['reconstruction']
    bkg_loss = keras.losses.mse(phi_testb, pred_phi_bkg)
    sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)

#    plot_1D_phi(bkg_loss, sig_loss, plot_dir=self.plot_dir, tag_file=self.vae_model, tag_title=self.vae_model)
    plot_score(bkg_loss, sig_loss, False, True, tag_file=self.vae_model+'_single_loss', tag_title=self.vae_model+'(single loss)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
#    """
    #start = time.time()
    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig)
    #end = time.time()
#    print("Elapsed (with get_multi_loss) = %s" % (end - start))
    try:cprint(f'{min(bkg_loss)}, {min(sig_loss)}, {max(bkg_loss)}, {max(sig_loss)}', 'yellow')
    except: cprint(f'{np.min(bkg_loss)}, {np.min(sig_loss)},{np.max(bkg_loss)}, {np.max(sig_loss)}', 'blue')
    print('\n')
    plot_score(bkg_loss, sig_loss, False, True, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, True, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+"_KLD", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, True, tag_file=self.vae_model+"_Reco", tag_title=self.vae_model+"_Reco", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_loss, sig_loss, False, False, tag_file=self.vae_model+'_nolog', tag_title=self.vae_model+'_nolog', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, tag_file=self.vae_model+'_nolog'+"_KLD", tag_title=self.vae_model+'_nolog'+"_KLD", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, tag_file=self.vae_model+'_nolog'+"_Reco", tag_title=self.vae_model+'_nolog'+"_Reco", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    
    plot_score(bkg_loss, sig_loss, False, True, tag_file=self.vae_model+'_nolog', tag_title=self.vae_model+'_nolog', plot_dir=self.plot_dir, bool_pfn=False, bool_neg=True) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, True, tag_file=self.vae_model+'_nolog'+"_KLD", tag_title=self.vae_model+'_nolog'+"_KLD", plot_dir=self.plot_dir, bool_pfn=False, bool_neg=True) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, True, tag_file=self.vae_model+'_nolog'+"_Reco", tag_title=self.vae_model+'_nolog'+"_Reco", plot_dir=self.plot_dir, bool_pfn=False, bool_neg=True) # anomaly score
    """
    """
    # # 3. Signal Sensitivity Score
    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    sic_vals=do_roc(bkg_loss, sig_loss, tag_file=self.vae_model, tag_title=self.vae_model,make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss, sig_kl_loss, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+"_KLD",make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_reco=do_roc(bkg_reco_loss, sig_reco_loss, tag_file=self.vae_model+"_Reco", tag_title=self.vae_model+"_Reco",make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    auc={sic_vals['auc'], sic_vals_kl['auc'], sic_vals_reco['auc']}
    bkg_events_num,sig_events_num=len(phi_bkg), len(phi_sig) 
    #do_roc(bkg_reco_loss, sig_reco_loss, vself.vae_model+"_Reco", True)
    #do_roc(bkg_kl_loss, sig_kl_loss, vself.vae_model+"_KLD", True)
    
    print("Taking log of score...")
    """
    bkg_loss = np.log(bkg_loss)
    sig_loss = np.log(sig_loss)
    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    do_roc(bkg_loss, sig_loss, tag_file=self.vae_model+'log', tag_title=self.vae_model+'log',make_transformed_plot= True, plot_dir=self.plot_dir,  bool_pfn=False)
    
    """
    # ## get predictions on test data
    # preds = pfn.predict(x_test)
    # ## get ROC curve
    # pfn_fp, pfn_tp, threshs = roc_curve(y_test[:,1], preds[:,1])
    # #
    # # get area under the ROC curve
    # auc = roc_auc_score(y_test[:,1], preds[:,1])
    # print()
    # print('PFN AUC:', auc)
    # print()
    # 
    # # plot distributions
    # bkg_score = preds[:,1][y_test[:,1] == 0]
    # sig_score = preds[:,1][y_test[:,1] == 1]
    # plot_score(bkg_score, sig_score, False, False, 'PFN')
    return self.all_dir, auc, bkg_events_num,sig_events_num

## AE events
#sig_events = 20000
#bkg_events = 200000
pfn_model='PFNv6'
"""
sig_events=1151555
bkg_events=3234186
ls_sig=[100000]
ls_bkg=[500000]
"""
ls_sig=[20000]
ls_bkg=[200000]
for sig_events, bkg_events in zip(ls_sig, ls_bkg):
  param1=Param_ANTELOPE(pfn_model=pfn_model,bkg_events=bkg_events, sig_events=sig_events, h5_dir='h5dir/antelope/aug17_jetpt/', arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
    learning_rate=0.00001, extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'])
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train_vae()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin))
  print(param1.save_info())
"""
for learning_rate in [0.00001, 0.0001, 0.001]:
  param1=Param_ANTELOPE(pfn_model=pfn_model,h5_dir='h5dir/antelope/aug10/', arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
    learning_rate=learning_rate)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train_vae()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin))
  print(param1.save_info())
  sys.exit()
encoding_dim = 32
latent_dim = 12
phi_dim = 64
#nepochs=20
nepochs=50
batchsize_vae=32

#pfn_model = 'PFNv1'
pfn_model = 'PFNv6'
vae_model = 'vANTELOPE' # vae change
#vae_model = 'ANTELOPE'
#arch_dir = "architectures_saved/"
arch_dir='/nevis/katya01/data/users/kpark/svj-vae/results/antelope/architectures_saved/'
#arch_dir = "/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/"

#  all_dir='/nevis/katya01/data/users/kpark/svj-vae/'
#    plot_dir=all_dir+'results/antelope/plots_vae/' # vae change
################### Train the AE or VAE ###############################

"""
"""
Troubleshooting
1) OSError: SavedModel file does not exist at: /data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/PFN_graph_arch/{saved_model.pbtxt|saved_model.pb}
-> pfn_model is wrong
"""
