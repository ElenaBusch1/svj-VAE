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
#import imageio
#import IPython 
#import cv2 

import pandas as pd
# Example usage
class Param_ANTELOPE(Param):
  def __init__(self,  
      arch_dir_pfn, arch_dir_vae='',kl_loss_scalar=100,
      arch_dir="architectures_saved/",print_dir='',plot_dir='plots/',h5_dir='h5dir/jul28/',
      pfn_model='PFN', vae_model='vANTELOPE', bkg_events=200000, sig_events=20000,
      num_elements=100, element_size=7, encoding_dim=32, latent_dim=12, phi_dim=64, nepochs=50, n_neuron=75, learning_rate=0.00001,
      nlayer_phi=3, nlayer_F=3,
      max_track=80,
      batchsize_pfn=512,
#      batchsize_pfn=500,
      batchsize_vae=32, # batchsize_pfn=500 -> 512 or any power of 2
      bool_pt=False,
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim0.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'],seed=0 ):
      #changeable: encoding_dim,latent_dim, nepochs, learning_rate, bkg_events, sig_events

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
    self.arch_dir_vae=arch_dir_vae
    self.kl_loss_scalar=kl_loss_scalar

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

  def transform_sigma(self, arr):
    # arr is z_log_var  
    # z_log_var= log(z_sig ^2)=2* log (z_sig) 
    # z_sig = exp(.5 *z_log_var) 
    return np.exp(.5 * arr) # sigma

  def load_vae(self):
    print('trying to load', self.arch_dir_vae+self.vae_model+'_encoder_arch') 
    encoder = keras.models.load_model(self.arch_dir_vae+self.vae_model+'_encoder_arch')
    decoder = keras.models.load_model(self.arch_dir_vae+self.vae_model+'_decoder_arch')
    vae = VAE(encoder,decoder)

    vae.get_layer('encoder').load_weights(self.arch_dir_vae+self.vae_model+'_encoder_weights.h5')
    vae.get_layer('decoder').load_weights(self.arch_dir_vae+self.vae_model+'_decoder_weights.h5')

    vae.compile(optimizer=keras.optimizers.Adam())
   
    return vae

  def load_pfn(self):
    
    cprint(self.arch_dir_pfn+self.pfn_model+'_graph_arch', 'yellow')
    graph = keras.models.load_model(self.arch_dir_pfn+self.pfn_model+'_graph_arch')
    graph.load_weights(self.arch_dir_pfn+self.pfn_model+'_graph_weights.h5')
    graph.compile()

    scaler = load(self.arch_dir_pfn+self.pfn_model+'_scaler.bin')

    return graph, scaler
  
  def sample_flatten(self, arr):
    if arr.ndim != 3.:
      print('not 3 D array: check sample_flatten')
      sys.exit()
    return arr.reshape(arr.shape[0], -1)


  def shuffle_two(self, arr1, arr2):
    shuffler = np.random.permutation(len(arr1))
    arr1_shuffled = arr1[shuffler]
    arr2_shuffled = arr2[shuffler]
    return arr1, arr2 

  def prepare_test_ECG(self):
    # apply scaling # is it already applied
    # plot vectors
# just the VAE
# choose validation and train
# vae fitting
# vae save
# reconstruct

    dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
    raw_data = dataframe.values
    dataframe.head()
    y=raw_data[:,-1] # label
    x=raw_data[:, 0:-1]
    
   # scale
    
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    print('before',type(x), x.shape)  
    x = (x - min_val) / (max_val - min_val)
    x = tf.cast(x, tf.float32)
    x= x.numpy()
    print('after converting',type(x), x.shape)  
    # split 
    x_evalb,  x_testb, y_evalb, y_testb = train_test_split(x, y, test_size=0.2)

    print(f'{x_evalb.shape=}, {x_testb.shape=}, {y_evalb.shape=}, {y_testb.shape=}')
    # shuffle before flattening!
    x_evalb, y_evalb=self.shuffle_two(x_evalb, y_evalb)
    x_testb, y_testb=self.shuffle_two(x_testb, y_testb)
    cprint(f'before{x_evalb.shape=}', 'yellow')
    cprint(f'before{type(x_evalb[0])}', 'yellow')
    cprint(f'before{type(y_evalb[0])}', 'yellow')
    print(y_evalb, y_testb)

    x_evalb, x_testb= x_evalb.astype('float32'), x_testb.astype('float32') 
    y_evalb, y_testb= y_evalb.astype('float32'), y_testb.astype('float32') 
    print(x_evalb.shape, x_testb.shape)
    cprint(f'after{type(x_evalb[0])}', 'yellow')
    cprint(f'after{type(y_evalb[0])}', 'yellow')

    # separate the normal rhythms from abnormal rhythms
    test_labels = y_testb.astype(bool)
    normal_test = x_testb[test_labels]
    anomalous_test = x_testb[~test_labels]

    # plot test input
    # select 15 samples since there are 10000
    nsample=15
    x_testb_plt= anomalous_test[:nsample]
    # reshape
    fig = plt.figure(figsize=(15, 10))
 
    for i in range(nsample):
      ax = fig.add_subplot(5, 5, i+1)
      ax.set(xlabel='Time', ylabel='Voltage')
      ax.plot(np.arange(140),x_testb_plt[i]) 
      ax.grid()

    fig.suptitle('Input Anomalous ECG')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'input_anomalous.png')
#    plt.show()
    plt.clf()

    
    nsample=15
    x_testb_plt= normal_test[:nsample]
    # reshape
    fig = plt.figure(figsize=(15, 10))
 
    for i in range(nsample):
      ax = fig.add_subplot(5, 5, i+1)
      ax.set(xlabel='Time', ylabel='Voltage')
      ax.plot(np.arange(140),x_testb_plt[i]) 
      ax.grid()

    fig.suptitle('Input Normal ECG')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'input_normal.png')
#    plt.show()
    plt.clf()
    
    print(f'after{x_evalb.shape=}', 'yellow')

    #x_testb = x_testb.astype('float32')
#    cprint(f'after{(x_evalb[0])}', 'yellow')
    print(f'{np.max(x_evalb)=}')
    # validation data set manually
    # Prepare the training dataset
    idx = np.random.choice( np.arange(len(x_evalb)), size= round(.2 *len(x_evalb)) , replace=False) # IMPT that replace=False so that event is picked only once
    idx = np.sort(idx) 
    # Prepare the validation dataset
    x_evalb_val = x_evalb[idx, :] 
    x_evalb_train = np.delete(x_evalb, idx) # doesn't modify input array 
    print(f'{x_evalb_val.shape=}, {x_evalb_train=}')     
    
    x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val = train_test_split(x_evalb, y_evalb, test_size=round(.2 *len(x_evalb)))
    #phi_evalb_train, phi_evalb_val, _, _ = train_testb_split(phi_evalb, phi_evalb, testb_size=round(.2 *len(phi_evalb)))
    return  x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val


  def prepare_test(self, bool_flat=False):
    # apply scaling # is it already applied
    # plot vectors
# just the VAE
# choose validation and train
# vae fitting
# vae save
# reconstruct
    (x_evalb, y_evalb), (x_testb, y_testb) = tf.keras.datasets.fashion_mnist.load_data()

    # shuffle before flattening!
    x_evalb, y_evalb=self.shuffle_two(x_evalb, y_evalb)
    x_testb, y_testb=self.shuffle_two(x_testb, y_testb)
    cprint(f'before{x_evalb.shape=}', 'yellow')
    cprint(f'before{type(x_evalb[0])}', 'yellow')
    cprint(f'before{type(y_evalb[0])}', 'yellow')
    print(y_evalb, y_testb)

    #flatten
    if bool_flat:
      x_evalb, x_testb= self.sample_flatten(x_evalb), self.sample_flatten(x_testb)

    else:   x_evalb, x_testb= x_evalb.reshape(x_evalb.shape[0], 28,28, 1), x_testb.reshape(x_testb.shape[0], 28,28, 1)
    x_evalb, x_testb= x_evalb.astype('float32'), x_testb.astype('float32') 
    y_evalb, y_testb= y_evalb.astype('float32'), y_testb.astype('float32') 
    print(x_evalb.shape, x_testb.shape)
    #x_evalb = x_evalb.reshape(x_evalb.shape[0], 28, 28, 1).astype('float32')
    cprint(f'after{type(x_evalb[0])}', 'yellow')
    cprint(f'after{type(y_evalb[0])}', 'yellow')


    # plot test input
    # select 15 samples since there are 10000
    nsample=10
    x_testb_plt= x_testb[:nsample]
    # reshape
    if bool_flat:
      x_testb_plt=x_testb.reshape(x_testb.shape[0], 28,28, 1) # should be (x, 28, 28,1)
    cprint(f'reshape {x_testb.shape=}')
    fig = plt.figure(figsize=(15, 10))
 
    for i in range(nsample):
      ax = fig.add_subplot(5, 5, i+1)
      ax.axis('off')
#      ax.text(0.5, -0.15, str(label_dict[y_test[i]]), fontsize=10, ha='center', transform=ax.transAxes)
     
      ax.imshow(x_testb_plt[i, :,:,0]*255, cmap = 'gray') 


    fig.suptitle('Input Image')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'input.png')
#    plt.show()
    plt.clf()
    
    print(f'after{x_evalb.shape=}', 'yellow')

    #x_testb = x_testb.astype('float32')
#    cprint(f'after{(x_evalb[0])}', 'yellow')
    print(f'{np.max(x_evalb)=}')
    x_evalb = x_evalb / 255.
    x_testb = x_testb / 255.
    # validation data set manually
    # Prepare the training dataset
    idx = np.random.choice( np.arange(len(x_evalb)), size= round(.2 *len(x_evalb)) , replace=False) # IMPT that replace=False so that event is picked only once
    idx = np.sort(idx) 
    # Prepare the validation dataset
    x_evalb_val = x_evalb[idx, :] 
    x_evalb_train = np.delete(x_evalb, idx) # doesn't modify input array 
    print(f'{x_evalb_val.shape=}, {x_evalb_train=}')     
    
    x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val = train_test_split(x_evalb, y_evalb, test_size=round(.2 *len(x_evalb)))
    #phi_evalb_train, phi_evalb_val, _, _ = train_testb_split(phi_evalb, phi_evalb, testb_size=round(.2 *len(phi_evalb)))
    return  x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val

  def prepare_pfn(self, graph,scaler):
    # scale values
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
    print('hola') 
      #read_dir='/data/users/ebusch/SVJ/autoencoder/v9.1/') 
    bkg2,_ = apply_StandardScaling(bkg,scaler,False) # change
    sig2,_ = apply_StandardScaling(sig,scaler,False) # change
    plot_vectors(bkg2,sig2,tag_file="ANTELOPE", tag_title=" (ANTELOPE)", plot_dir=self.plot_dir)# change
    
    phi_bkg = graph.predict(bkg2)
    phi_sig = graph.predict(sig2)
 
    plot_1D_phi(phi_bkg, phi_sig,labels=['QCD', 'sig'], plot_dir=self.plot_dir, tag_file=self.pfn_model, tag_title=self.pfn_model)
    
    #phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0], random_state=42) # no info on labels e.g. y_train or y_test
    phi_evalb_idx, phi_testb_idx, _, _ = train_test_split(np.arange(len(phi_bkg)), phi_bkg, test_size=sig2.shape[0])
    phi_evalb, phi_testb, mT_evalb, mT_testb= phi_bkg[phi_evalb_idx, :], phi_bkg[phi_testb_idx, :], mT_bkg[phi_evalb_idx,:], mT_bkg[phi_testb_idx, :]
    print('idx',phi_evalb_idx, phi_testb_idx)
    print('after',phi_evalb.shape, phi_testb.shape,  mT_evalb.shape, mT_testb.shape)
    
    plot_single_variable([mT_evalb[:,2], mT_testb[:,2]],h_names= ['training and validation', 'test'],weights_ls=[mT_evalb[:,1], mT_testb[:,1]], tag_title= 'leading jet pT (QCD)', plot_dir=self.plot_dir,logy=True, tag_file='jet1_pt')
    plot_single_variable([mT_evalb[:,3], mT_testb[:,3]],h_names= ['training and validation', 'test'],weights_ls=[mT_evalb[:,1], mT_testb[:,1]], tag_title= 'subleading jet pT (QCD)', plot_dir=self.plot_dir, logy=True, tag_file='jet2_pt')
    """
    plot_phi(phi_evalb,tag_file="PFN_phi_train_raw",tag_title="Train",plot_dir=self.plot_dir) # change
    plot_phi(phi_testb,tag_file="PFN_phi_test_raw",tag_title="Test", plot_dir=self.plot_dir)
    plot_phi(phi_sig,tag_file="PFN_phi_sig_raw", tag_title="Signal", plot_dir=self.plot_dir)
    """
    
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
    
    plot_phi(phi_evalb,tag_file="PFN_phi_train_scaled",tag_title="Train Scaled", plot_dir=self.plot_dir) # change
    plot_phi(phi_testb,tag_file="PFN_phi_test_scaled",tag_title="Test Scaled", plot_dir=self.plot_dir)
    plot_phi(phi_sig,tag_file="PFN_phi_sig_scaled", tag_title="Signal Scaled", plot_dir=self.plot_dir)
    
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

    return   phi_bkg,phi_testb, phi_evalb_train, phi_evalb_val, phi_sig
 

  def train_vae(self, phi_evalb_train, phi_evalb_val, y_phi_evalb_train=[], y_phi_evalb_val=[]):

    vae = get_vae(self.phi_dim,self.encoding_dim,self.latent_dim, self.learning_rate, self.kl_loss_scalar, bool_test=False)
      
    h2 = vae.fit(phi_evalb_train, 
    #h2 = vae.fit(phi_evalb, 
        epochs=self.nepochs,
        batch_size=self.batchsize_vae,
      #  validation_data=(phi_evalb_val, phi_evalb_val),
        validation_split=0.2,
        verbose=1)
    # # simple ae
    vae.get_layer('encoder').save_weights(self.arch_dir+self.vae_model+'_encoder_weights.h5')
    vae.get_layer('decoder').save_weights(self.arch_dir+self.vae_model+'_decoder_weights.h5')
    vae.get_layer('encoder').save(self.arch_dir+self.vae_model+'_encoder_arch')
    vae.get_layer('decoder').save(self.arch_dir+self.vae_model+'_decoder_arch')
    print('successful in saving model' + self.arch_dir+self.vae_model)

    return vae, h2


  def evaluate_test_ECG(self):
    x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val= self.prepare_test_ECG()
    print('prepare_test_ECG')
      
    try: vae = self.load_vae()
    except: 
      print('loading vae not successful so will start the training process')
      vae,h2 = self.train_vae( x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val)
      print('training successful')

    latent_test=vae.get_layer('encoder').predict(x_testb)
    latent_train=vae.get_layer('encoder').predict(x_evalb_train)
    latent_val=vae.get_layer('encoder').predict(x_evalb_val)


    #latent_test is a list but latent_test[0] is a numpy array
    latent_test, latent_train, latent_val=np.array(latent_test), np.array(latent_train), np.array(latent_val)
    print(f'{latent_test.shape=}')

    plot_pca(latent_test[0,:,:], latent_label=np.array(y_testb), nlabel=2,n_components=2, tag_file=self.vae_model+'_test', tag_title=self.vae_model+' Test', plot_dir=self.plot_dir)
    plot_pca(latent_train[0,:,:],latent_label=np.array(y_evalb_train), nlabel=2, n_components=2, tag_file=self.vae_model+'_train', tag_title=self.vae_model+' Train', plot_dir=self.plot_dir)

    # reconstruct output
    nsample=10
    test_labels = y_testb.astype(bool)
    normal_test = x_testb[test_labels]
    anomalous_test = x_testb[~test_labels]
    normal_latent_test=latent_test[:,test_labels,:]
    anomalous_latent_test=latent_test[:,~test_labels,:]
    
    print(f'{normal_latent_test.shape=}')
    print(f'{anomalous_latent_test.shape=}')

#    print(f'{latent_test.shape=}')
    #latent_test_recon = vae.get_layer('decoder').predict(latent_test[2,:,:])
    normal_latent_test_recon = vae.get_layer('decoder').predict(normal_latent_test[2,:,:])
    anomalous_latent_test_recon = vae.get_layer('decoder').predict(anomalous_latent_test[2,:,:])
    print(f'{normal_latent_test_recon.shape=}')
    print(f'{anomalous_latent_test_recon.shape=}')
    # select 15 samples since there are 10000
    # reshape
    
    fig = plt.figure(figsize=(15, 10))
 
    for i in range(nsample):
      ax = fig.add_subplot(5, 5, i+1)
      plt.plot(normal_test[i], 'b', label='Input')
      plt.plot(normal_latent_test_recon[i], 'r', label='Reconstruction')
      ax.fill_between(np.arange(140),normal_latent_test_recon[i], normal_test[i], color='lightcoral', label='Error') 
      ax.set(xlabel='Time', ylabel='Voltage')
      ax.grid()
    #ax.imshow(latent_test_recon[i, :,:,0]*255, cmap = 'gray') 
    fig.suptitle('Output Normal ECG')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'output_normal.png')
#    plt.show()
    plt.clf()


    fig = plt.figure(figsize=(15, 10))
 
    for i in range(nsample):
      ax = fig.add_subplot(5, 5, i+1)
      plt.plot(anomalous_test[i], 'b', label='Input')
      plt.plot(anomalous_latent_test_recon[i], 'r', label='Reconstruction')
      ax.fill_between(np.arange(140),anomalous_latent_test_recon[i], anomalous_test[i], color='lightcoral', label='Error') 
      ax.set(xlabel='Time', ylabel='Voltage')
      ax.grid()
    #ax.imshow(latent_test_recon[i, :,:,0]*255, cmap = 'gray') 
    fig.suptitle('Output Anomalous ECG')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'output_anomalous.png')
#    plt.show()
    plt.clf()

    normal_latent_test_sigma, anomalous_latent_test_sigma = self.transform_sigma(normal_latent_test[1,:,:]), self.transform_sigma(anomalous_latent_test[1, :,:])

#    for k in range(len(latent_test)):
    plot_1D_phi(normal_latent_test[0,:,:],anomalous_latent_test[0,:,:] , labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_mu', tag_title=self.vae_model +r" $\mu$", ylog=True)
    plot_1D_phi(normal_latent_test_sigma,anomalous_latent_test_sigma , labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sigma', tag_title=self.vae_model +r" $\sigma$", ylog=True)

    plot_1D_phi(normal_latent_test[0,:,:],anomalous_latent_test[0,:,:] , labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_mu_custom', tag_title=self.vae_model +r" $\mu$",  bins=np.linspace(-0.0001,0.0001, num=50))
    plot_1D_phi(normal_latent_test_sigma,anomalous_latent_test_sigma , labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sigma', tag_title=self.vae_model +r" $\sigma$", bins=np.linspace(0.9998,1.0002,num=50))

    plot_1D_phi(normal_latent_test[2,:,:],anomalous_latent_test[2,:,:] , labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sampling', tag_title=self.vae_model +" Sampling", bool_norm=True)

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch 
    try:plot_loss(h2, loss='loss', tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
    except: print('loading vae_model so cannot draw regular loss plot -> no h2')
    #plot_loss(h2, vae_model, "kl_loss")
    #plot_loss(h2, vae_model, "reco_loss")
    
    #2. Get loss
#    """
    pred_x_test = vae.predict(x_testb)['reconstruction']
    bkg_loss_mse = keras.losses.mse(x_testb, pred_x_test)

#    plot_score(bkg_loss_mse, np.array([]), False, True, tag_file=self.vae_model+'_single_loss', tag_title=self.vae_model+' (MSE)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
#    """
    #start = time.time()
    step_size=self.batchsize_vae
    bkg_loss,  bkg_kl_loss,  bkg_reco_loss  = get_multi_loss_each(vae, x_testb, step_size=step_size)
    bkg_loss,  bkg_kl_loss,  bkg_reco_loss = np.array(bkg_loss),  np.array( bkg_kl_loss), np.array(bkg_reco_loss)
    #end = time.time()
#    print("Elapsed (with get_multi_loss) = %s" % (end - start))
#    try:cprint(f'{min(bkg_loss)}, {min(sig_loss)}, {max(bkg_loss)}, {max(sig_loss)}', 'yellow')
#    except: cprint(f'{np.min(bkg_loss)}, {np.min(sig_loss)},{np.max(bkg_loss)}, {np.max(sig_loss)}', 'blue')
    print('\n')

    # xlog=True plots
    sig_loss, sig_kl_loss, sig_reco_loss=np.array([]), np.array([]), np.array([]) 
    plot_score(bkg_loss[bkg_loss>0], sig_loss[sig_loss>0], False, True, tag_file=self.vae_model+'_pos', tag_title=self.vae_model + ' (score > 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], False, True, tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss>0], sig_reco_loss[sig_reco_loss>0], False, True, tag_file=self.vae_model+"_MSE_pos", tag_title=self.vae_model+" MSE (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog=False plots
    plot_score(bkg_loss, sig_loss, False, False, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog= False plots plot only points less than 0 
    plot_score(bkg_loss[bkg_loss<=0], sig_loss[sig_loss<=0], False, False, tag_file=self.vae_model+'_neg', tag_title=self.vae_model+' (score <= 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss<=0], sig_kl_loss[sig_kl_loss<=0], False, False, tag_file=self.vae_model+"_KLD_neg", tag_title=self.vae_model+" KLD (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss<=0], sig_reco_loss[sig_reco_loss<=0], False, False, tag_file=self.vae_model+"_MSE_neg", tag_title=self.vae_model+" MSE (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    # # 3. Signal Sensitivity Score

    """
    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    sic_vals=do_roc(bkg_loss, sig_loss, tag_file=self.vae_model, tag_title=self.vae_model+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss, sig_kl_loss, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_reco=do_roc(bkg_reco_loss, sig_reco_loss, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    auc={sic_vals['auc'], sic_vals_kl['auc'], sic_vals_reco['auc']}
    """
    auc= {np.nan, np.nan, np.nan}
    bkg_events_num,sig_events_num=np.nan, np.nan 
    
  
    return self.all_dir, auc, bkg_events_num,sig_events_num

  def evaluate_test(self):
    test_labels = y_evalb.astype(bool)
    normal_test = x_evalb[test_labels]
    anomalous_test = x_evalb[~test_labels]
    x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val= self.prepare_test(bool_flat=True)
    print('prepare_test')
      
    try: vae = self.load_vae()
    except: 
      print('loading vae not successful so will start the training process')
      vae,h2 = self.train_vae( x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val)
      print('training successful')

    latent_test=vae.get_layer('encoder').predict(x_testb)
    latent_train=vae.get_layer('encoder').predict(x_evalb_train)
    latent_val=vae.get_layer('encoder').predict(x_evalb_val)


    #latent_test is a list but latent_test[0] is a numpy array
    latent_test, latent_train, latent_val=np.array(latent_test), np.array(latent_train), np.array(latent_val)
    print(f'{latent_test.shape=}')

    print(f'{y_testb=}{y_testb.shape}')
    plot_pca(latent_test[0,:,:], latent_label=np.array(y_testb), nlabel=10,n_components=2, tag_file=self.vae_model+'_test', tag_title=self.vae_model+' Test', plot_dir=self.plot_dir)
    plot_pca(latent_train[0,:,:],latent_label=np.array(y_evalb_train), nlabel=10, n_components=2, tag_file=self.vae_model+'_train', tag_title=self.vae_model+' Train', plot_dir=self.plot_dir)

    # reconstruct output
#    print(f'{latent_test.shape=}')
    latent_test_recon = vae.get_layer('decoder').predict(latent_test[2,:,:])
    print(f'{latent_test_recon.shape=}')
    # select 15 samples since there are 10000
    nsample=10
    latent_test_recon= latent_test_recon[:nsample]
    # reshape
    latent_test_recon=latent_test_recon.reshape(latent_test_recon.shape[0], 28, -1) # should be (x, 28, 28)
    cprint(f'reshape 1 {latent_test_recon.shape=}')
    latent_test_recon=latent_test_recon.reshape(latent_test_recon.shape[0], 28,28, 1) # should be (x, 28, 28,1)
    cprint(f'reshape 2 {latent_test_recon.shape=}')
    fig = plt.figure(figsize=(15, 10))
 
    for i in range(nsample):
      ax = fig.add_subplot(5, 5, i+1)
      ax.axis('off')
#      ax.text(0.5, -0.15, str(label_dict[y_test[i]]), fontsize=10, ha='center', transform=ax.transAxes)
     
      ax.imshow(latent_test_recon[i, :,:,0]*255, cmap = 'gray') 
    #ax.imshow(latent_test_recon[i, :,:,0]*255, cmap = 'gray') 
    fig.suptitle('Reconstructed Image')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'output.png')
#    plt.show()
    plt.clf()



    latent_test_sigma, latent_train_sigma = self.transform_sigma(latent_test[1,:,:]), self.transform_sigma(latent_train[1, :,:])

#    for k in range(len(latent_test)):
    plot_1D_phi(latent_test[0,:,:],latent_train[0,:,:] , labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_mu', tag_title=self.vae_model +r" $\mu$", ylog=True)
    plot_1D_phi(latent_test_sigma, latent_train_sigma, labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_sigma', tag_title=self.vae_model  +r" $\sigma$", ylog=True)

    plot_1D_phi(latent_test[0,:,:], latent_train[0, :,:], labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_mu_custom', tag_title=self.vae_model +r" $\mu$", bins=np.linspace(-0.0001,0.0001, num=50))
    plot_1D_phi(latent_test_sigma,latent_train_sigma, labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_sigma_custom', tag_title=self.vae_model +r" $\sigma$", bins=np.linspace(0.9998,1.0002,num=50))

    plot_1D_phi(latent_test[2,:,:], latent_train[2, :,:], labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_sampling', tag_title=self.vae_model +" Sampling",bool_norm=True)

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch 
    try:plot_loss(h2, loss='loss', tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
    except: print('loading vae_model so cannot draw regular loss plot -> no h2')
    #plot_loss(h2, vae_model, "kl_loss")
    #plot_loss(h2, vae_model, "reco_loss")
    
    #2. Get loss
#    """
    pred_x_test = vae.predict(x_testb)['reconstruction']
    bkg_loss_mse = keras.losses.mse(x_testb, pred_x_test)

#    plot_score(bkg_loss_mse, np.array([]), False, True, tag_file=self.vae_model+'_single_loss', tag_title=self.vae_model+' (MSE)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
#    """
    #start = time.time()
    step_size=self.batchsize_vae
    bkg_loss,  bkg_kl_loss,  bkg_reco_loss  = get_multi_loss_each(vae, x_testb, step_size=step_size)
    bkg_loss,  bkg_kl_loss,  bkg_reco_loss = np.array(bkg_loss),  np.array( bkg_kl_loss), np.array(bkg_reco_loss)
    #end = time.time()
#    print("Elapsed (with get_multi_loss) = %s" % (end - start))
#    try:cprint(f'{min(bkg_loss)}, {min(sig_loss)}, {max(bkg_loss)}, {max(sig_loss)}', 'yellow')
#    except: cprint(f'{np.min(bkg_loss)}, {np.min(sig_loss)},{np.max(bkg_loss)}, {np.max(sig_loss)}', 'blue')
    print('\n')

    # xlog=True plots
    sig_loss, sig_kl_loss, sig_reco_loss=np.array([]), np.array([]), np.array([]) 
    plot_score(bkg_loss[bkg_loss>0], sig_loss[sig_loss>0], False, True, tag_file=self.vae_model+'_pos', tag_title=self.vae_model + ' (score > 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], False, True, tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss>0], sig_reco_loss[sig_reco_loss>0], False, True, tag_file=self.vae_model+"_MSE_pos", tag_title=self.vae_model+" MSE (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog=False plots
    plot_score(bkg_loss, sig_loss, False, False, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog= False plots plot only points less than 0 
    plot_score(bkg_loss[bkg_loss<=0], sig_loss[sig_loss<=0], False, False, tag_file=self.vae_model+'_neg', tag_title=self.vae_model+' (score <= 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss<=0], sig_kl_loss[sig_kl_loss<=0], False, False, tag_file=self.vae_model+"_KLD_neg", tag_title=self.vae_model+" KLD (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss<=0], sig_reco_loss[sig_reco_loss<=0], False, False, tag_file=self.vae_model+"_MSE_neg", tag_title=self.vae_model+" MSE (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    # # 3. Signal Sensitivity Score

    """
    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    sic_vals=do_roc(bkg_loss, sig_loss, tag_file=self.vae_model, tag_title=self.vae_model+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss, sig_kl_loss, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_reco=do_roc(bkg_reco_loss, sig_reco_loss, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    auc={sic_vals['auc'], sic_vals_kl['auc'], sic_vals_reco['auc']}
    """
    auc= {np.nan, np.nan, np.nan}
    bkg_events_num,sig_events_num=np.nan, np.nan 
    
  
    return self.all_dir, auc, bkg_events_num,sig_events_num

 
  def evaluate_vae(self):
    graph, scaler = self.load_pfn()
    phi_bkg,phi_testb, phi_evalb_train, phi_evalb_val, phi_sig=  self.prepare_pfn(graph,scaler)
    print('prepare_pfn')
      
    try: vae = self.load_vae()
    except: 
      print('loading vae not successful so will start the training process')
      vae,h2 = self.train_vae( phi_evalb_train, phi_evalb_val)
      print('training successful')

    #complex ae
    #with open(arch_dir+vae_model+"8.1_predstory.json", "w") as f:
    #    json.dump(h2.predstory, f)
    latent_bkg_test=vae.get_layer('encoder').predict(x_testb)
    latent_bkg_train=vae.get_layer('encoder').predict(x_evalb_train)
    latent_bkg_val=vae.get_layer('encoder').predict(x_evalb_val)

    #latent_bkg_test is a list but latent_bkg_test[0] is a numpy array
    latent_bkg_test, latent_bkg_train, latent_bkg_val, latent_sig=np.array(latent_bkg_test), np.array(latent_bkg_train), np.array(latent_bkg_val), np.array(latent_sig)
    print(f'{latent_bkg_test.shape=}')
    latent_bkg_test_sigma, latent_sig_sigma = self.transform_sigma(latent_bkg_test[1,:,:]), self.transform_sigma(latent_sig[1, :,:])


#    for k in range(len(latent_bkg_test)):
    plot_1D_phi(latent_bkg_test[0,:,:], latent_sig[0, :,:], labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_mu', tag_title=self.vae_model +r" $\mu$", ylog=True)
    plot_1D_phi(latent_bkg_test_sigma, latent_sig_sigma, labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_sigma', tag_title=self.vae_model  +r" $\sigma$", ylog=True)

    plot_1D_phi(latent_bkg_test[0,:,:], latent_sig[0, :,:], labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_mu_custom', tag_title=self.vae_model +r" $\mu$", bins=np.linspace(-0.0001,0.0001, num=50))
    plot_1D_phi(latent_bkg_test_sigma,latent_sig_sigma, labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_sigma_custom', tag_title=self.vae_model +r" $\sigma$", bins=np.linspace(0.9998,1.0002,num=50))

    plot_1D_phi(latent_bkg_test[2,:,:], latent_sig[2, :,:], labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_sampling', tag_title=self.vae_model +" Sampling",bool_norm=True)

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch 
    try:plot_loss(h2, loss='loss', tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
    except: print('loading vae_model so cannot draw regular loss plot -> no h2')
    #plot_loss(h2, vae_model, "kl_loss")
    #plot_loss(h2, vae_model, "reco_loss")
    
    #2. Get loss
    #bkg_loss, sig_loss = get_single_loss(ae, phi_testb, phi_sig)
#    """
    pred_phi_bkg = vae.predict(phi_testb)['reconstruction']
    pred_phi_sig = vae.predict(phi_sig)['reconstruction']
    bkg_loss_mse = keras.losses.mse(phi_testb, pred_phi_bkg)
    sig_loss_mse = keras.losses.mse(phi_sig, pred_phi_sig)

    plot_score(bkg_loss_mse, sig_loss_mse, False, True, tag_file=self.vae_model+'_single_loss', tag_title=self.vae_model+' (MSE)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
#    """
    #start = time.time()
    step_size=self.batchsize_vae
#    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig)
    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig, step_size=step_size)
    #bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig, step_size=100)
    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss= np.array(bkg_loss), np.array(sig_loss), np.array( bkg_kl_loss), np.array(sig_kl_loss), np.array(bkg_reco_loss), np.array(sig_reco_loss)
    #end = time.time()
#    print("Elapsed (with get_multi_loss) = %s" % (end - start))
    try:cprint(f'{min(bkg_loss)}, {min(sig_loss)}, {max(bkg_loss)}, {max(sig_loss)}', 'yellow')
    except: cprint(f'{np.min(bkg_loss)}, {np.min(sig_loss)},{np.max(bkg_loss)}, {np.max(sig_loss)}', 'blue')
    print('\n')

    # xlog=True plots 
    plot_score(bkg_loss[bkg_loss>0], sig_loss[sig_loss>0], False, True, tag_file=self.vae_model+'_pos', tag_title=self.vae_model + ' (score > 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], False, True, tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss>0], sig_reco_loss[sig_reco_loss>0], False, True, tag_file=self.vae_model+"_MSE_pos", tag_title=self.vae_model+" MSE (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog=False plots
    plot_score(bkg_loss, sig_loss, False, False, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog= False plots plot only points less than 0 
    plot_score(bkg_loss[bkg_loss<=0], sig_loss[sig_loss<=0], False, False, tag_file=self.vae_model+'_neg', tag_title=self.vae_model+' (score <= 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss<=0], sig_kl_loss[sig_kl_loss<=0], False, False, tag_file=self.vae_model+"_KLD_neg", tag_title=self.vae_model+" KLD (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss<=0], sig_reco_loss[sig_reco_loss<=0], False, False, tag_file=self.vae_model+"_MSE_neg", tag_title=self.vae_model+" MSE (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    # # 3. Signal Sensitivity Score

    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    sic_vals=do_roc(bkg_loss, sig_loss, tag_file=self.vae_model, tag_title=self.vae_model+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss, sig_kl_loss, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_reco=do_roc(bkg_reco_loss, sig_reco_loss, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    auc={sic_vals['auc'], sic_vals_kl['auc'], sic_vals_reco['auc']}
    bkg_events_num,sig_events_num=len(phi_bkg), len(phi_sig) 
    
    # LOG score
    print("Taking log of score...")
  
    bkg_loss_mse = np.log(bkg_loss_mse)
    sig_loss_mse = np.log(sig_loss_mse)
    score = getSignalSensitivityScore(bkg_loss_mse, sig_loss_mse)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    do_roc(bkg_loss_mse, sig_loss_mse, tag_file=self.vae_model+'_log_MSE', tag_title=self.vae_model+'log (MSE)',make_transformed_plot= True, plot_dir=self.plot_dir,  bool_pfn=False)
    return self.all_dir, auc, bkg_events_num,sig_events_num

## AE events
#sig_events = 20000
#bkg_events = 200000
if __name__=="__main__":
  pfn_model='PFNv6'
  """
  sig_events=1151555
  bkg_events=3234186
  ls_sig=[100000]
  ls_bkg=[500000]
  """
  ls_sig=[20000]
  ls_bkg=[200000]
  for  kl_loss_scalar in [1]:
    param1=Param_ANTELOPE(pfn_model=pfn_model,  h5_dir='h5dir/antelope/aug17_jetpt/', arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim=140)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim= 784, encoding_dim=196, latent_dim=49)# if flattening
#    stdoutOrigin=param1.open_print()
    all_dir, auc,bkg_events_num,sig_events_num=param1.evaluate_test_ECG()
    #all_dir, auc,bkg_events_num,sig_events_num=param1.evaluate_vae(bool_pfn=False)
    setattr(param1, 'auc',auc )
    setattr(param1, 'sig_events_num',sig_events_num )
    setattr(param1, 'bkg_events_num',bkg_events_num )
#    print(param1.close_print(stdoutOrigin))
#    print(param1.save_info())
"""
  ls_sig=[20000]
  ls_bkg=[200000]
  for  kl_loss_scalar in [1]:
    param1=Param_ANTELOPE(pfn_model=pfn_model,  h5_dir='h5dir/antelope/aug17_jetpt/', arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, batchsize_vae=64, nepochs=20)
    stdoutOrigin=param1.open_print()
    all_dir, auc,bkg_events_num,sig_events_num=param1.evaluate_vae(bool_pfn=True)
    #all_dir, auc,bkg_events_num,sig_events_num=param1.evaluate_vae(bool_pfn=False)
    setattr(param1, 'auc',auc )
    setattr(param1, 'sig_events_num',sig_events_num )
    setattr(param1, 'bkg_events_num',bkg_events_num )
    print(param1.close_print(stdoutOrigin))
    print(param1.save_info())
  for sig_events, bkg_events in zip(ls_sig, ls_bkg):
    param1=Param_ANTELOPE(pfn_model=pfn_model,bkg_events=bkg_events, sig_events=sig_events, h5_dir='h5dir/antelope/aug17_jetpt/', arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
  #    learning_rate=0.00001, extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar=100)
      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], arch_dir_vae='/data/users/kpark/svj-vae/results/antelope/08_21_23_12_04/architectures_saved/')
    stdoutOrigin=param1.open_print()
    all_dir, auc,bkg_events_num,sig_events_num=param1.evaluate_vae()
    setattr(param1, 'auc',auc )
    setattr(param1, 'sig_events_num',sig_events_num )
    setattr(param1, 'bkg_events_num',bkg_events_num )
    print(param1.close_print(stdoutOrigin))
    print(param1.save_info())
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
