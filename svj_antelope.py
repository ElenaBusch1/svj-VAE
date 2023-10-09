import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
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
params = {'legend.fontsize': 'x-large', 
           'figure.figsize': (10, 8), 
         'axes.labelsize': 'large', 
          'axes.titlesize':'x-large', 
         'xtick.labelsize':'large', 
         'ytick.labelsize':'large', 
 
         } 
""" 
 
         'axes.labelsize': 'x-large', 
          'axes.titlesize':'xx-large', 
         'xtick.labelsize':'x-large', 
         'ytick.labelsize':'x-large', 
 
""" 
plt.rcParams.update(params) 
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
      step_size=1,
      metric_dict={},
      bool_shift=False,
      bool_no_scaling=False,
      bool_nonzero=True,
      scalar_ecg=1,
      decoder_activation='relu',
      bool_float64=False,
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="user.ebusch.dataALL.root",  bool_weight=True, extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'],seed=0 ):
      #sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim0.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'],seed=0 ):
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
    self.step_size=step_size
    self.metric_dict=metric_dict
    self.bool_shift=bool_shift
    self.bool_no_scaling=bool_no_scaling # this overrides bool_shift
    self.bool_nonzero=bool_nonzero
    self.scalar_ecg=scalar_ecg
    self.decoder_activation=decoder_activation
    self.bool_float64=bool_float64
  # check for non scaled data -> bool_no_scaling=True, bool_decoder activation = 'relu', bool_float64 -> False

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
    vae = VAE(encoder,decoder,kl_loss_scalar=self.kl_loss_scalar)
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


  def plot_compare_ECG(self, x, label, title1, title2, x_latent_recon=np.array([])):
    # title1 = 'input' or 'output' 
    #title2 = 'normal' or 'anomalous' 
    # select 9 samples since there are 10000 
    nsample=9
    x= x[:nsample]
    if x_latent_recon.size !=0:    x_latent_recon= x_latent_recon[:nsample]
    fig = plt.figure()

    for i in range(nsample):
      ax = fig.add_subplot(3, 3, i+1)
      ax.set(xlabel='Time', ylabel='Voltage')
      if x_latent_recon.size ==0:
        ax.plot(np.arange(140),x[i], label=f'{label.capitalize()}')
      else:
        plt.plot(x[i], 'b', label=f'{label.capitalize()}')
        plt.plot(x_latent_recon[i], 'r', label='Reconstruction')
        ax.fill_between(np.arange(140),x_latent_recon[i], x[i], color='lightcoral', label='Error')

      ax.grid()
      handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'{title1.capitalize()} {title2.capitalize()} ECG [Normalized]')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'{title1}_{title2}.png')
#    plt.show() 
    plt.clf()


  def evaluate_test_ECG(self):
    x_testb, x_evalb, y_testb, y_evalb= self.prepare_test_ECG( bool_float64=self.bool_float64)

    #x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val= self.prepare_test_ECG()
    print('prepare_test_ECG')
#    try: vae = self.load_vae()
#    except: 
    print('loading vae not successful so will start the training process')
    vae,h2 = self.train_vae( x_evalb, y_evalb)
      #vae,h2 = self.train_vae( x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val)
    print('training successful')

    x_test_labels = y_testb.astype(bool)
    normal_x_test = x_testb[x_test_labels]
    anomalous_x_test = x_testb[~x_test_labels]

    var='train'
    plot_1D_phi(x_evalb[:,:7],np.array([]),labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.pfn_model+f'_{var}', tag_title=self.pfn_model+f' {var.capitalize()}')
    plot_phi(x_evalb,tag_file=f"ECG_normal_x_{var}", tag_title=f"Normal {var.capitalize()}", plot_dir=self.plot_dir)

    var='test'
    plot_1D_phi(normal_x_test[:,:7],anomalous_x_test[:,:7],labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.pfn_model+f'_{var}', tag_title=self.pfn_model+f' {var.capitalize()}')
    plot_phi(normal_x_test,tag_file=f"ECG_normal_x_{var}", tag_title=f"Normal {var.capitalize()}", plot_dir=self.plot_dir)
    plot_phi(anomalous_x_test,tag_file=f"ECG_anomalous_x_{var}", tag_title=f"Anomalous {var.capitalize()}", plot_dir=self.plot_dir)

    # if want to use validation_split instead of manual splitting
    x_evalb_train, y_evalb_train = x_evalb, y_evalb

    latent_test=vae.get_layer('encoder').predict(x_testb)
    latent_train=vae.get_layer('encoder').predict(x_evalb_train)
#    latent_val=vae.get_layer('encoder').predict(x_evalb_val)

    #latent_test is a list but latent_test[0] is a numpy array
    latent_test, latent_train =np.array(latent_test), np.array(latent_train)
    #latent_test, latent_train, latent_val=np.array(latent_test), np.array(latent_train), np.array(latent_val)
    print(f'{latent_test.shape=}')

    try:plot_pca(latent_test[0,:,:], latent_label=np.array(y_testb), nlabel=10,n_components=2, tag_file=self.vae_model+'_test', tag_title=self.vae_model+' Test', plot_dir=self.plot_dir)
    except:plot_pca(latent_test[:,:], latent_label=np.array(y_testb), nlabel=10,n_components=2, tag_file=self.vae_model+'_test', tag_title=self.vae_model+' Test', plot_dir=self.plot_dir)
    try:plot_pca(latent_train[0,:,:],latent_label=np.array(y_evalb_train), nlabel=10, n_components=2, tag_file=self.vae_model+'_train', tag_title=self.vae_model+' Train', plot_dir=self.plot_dir)
    except:plot_pca(latent_train[:,:],latent_label=np.array(y_evalb_train), nlabel=10, n_components=2, tag_file=self.vae_model+'_train', tag_title=self.vae_model+' Train', plot_dir=self.plot_dir)

    # reconstruct output
    test_labels = y_testb.astype(bool)
    bkg_test = x_testb[test_labels]
    y_bkg_test = y_testb[test_labels]
    y_sig_test = y_testb[~test_labels]
    sig_test = x_testb[~test_labels]
    try:bkg_latent_test=latent_test[:,test_labels,:]
    except:bkg_latent_test=latent_test[test_labels,:]
    try:sig_latent_test=latent_test[:,~test_labels,:]
    except:sig_latent_test=latent_test[~test_labels,:]

    train_labels = y_evalb_train.astype(bool)
    bkg_train = x_evalb_train[train_labels]
    sig_train = x_evalb_train[~train_labels]
    y_bkg_train = y_evalb_train[train_labels]
    y_sig_train = y_evalb_train[~train_labels]
    print(f'{bkg_latent_test.shape=}')
    print(f'{sig_latent_test.shape=}')

    try:bkg_latent_test_recon = vae.get_layer('decoder').predict(bkg_latent_test[2,:,:])
    except:bkg_latent_test_recon = vae.get_layer('decoder').predict(bkg_latent_test)
    try:sig_latent_test_recon = vae.get_layer('decoder').predict(sig_latent_test[2,:,:])
    except: sig_latent_test_recon = vae.get_layer('decoder').predict(sig_latent_test)
    print(f'{bkg_latent_test_recon.shape=}')
    print(f'{sig_latent_test_recon.shape=}')

    #REMOVE
    bkg_latent_test_recon*=self.scalar_ecg
    sig_latent_test_recon*=self.scalar_ecg

    self.plot_compare_ECG(x =bkg_test,label='input', title1 = 'output' ,title2='normal', x_latent_recon=bkg_latent_test_recon)
    self.plot_compare_ECG(x =sig_test,label='input', title1 = 'output',title2='anomalous', x_latent_recon=sig_latent_test_recon)

    try:bkg_latent_test_sigma, sig_latent_test_sigma = self.transform_sigma(bkg_latent_test[1,:,:]), self.transform_sigma(sig_latent_test[1, :,:])
    except: bkg_latent_test_sigma, sig_latent_test_sigma = self.transform_sigma(bkg_latent_test), self.transform_sigma(sig_latent_test)
    labels=['normal', 'anomalous']

    """
    plot_1D_phi(bkg_latent_test[0,:,:],sig_latent_test[0,:,:] , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_mu', tag_title=self.vae_model +r" $\mu$", ylog=True)
    plot_1D_phi(bkg_latent_test_sigma,sig_latent_test_sigma , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sigma', tag_title=self.vae_model +r" $\sigma$", ylog=True)

    plot_1D_phi(bkg_latent_test[0,:,:],sig_latent_test[0,:,:] , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_mu_custom', tag_title=self.vae_model +r" $\mu$",  bins=np.linspace(-0.0001,0.0001, num=50))
    plot_1D_phi(bkg_latent_test_sigma,sig_latent_test_sigma , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sigma_custom', tag_title=self.vae_model +r" $\sigma$", bins=np.linspace(0.9998,1.0002,num=50))

    plot_1D_phi(bkg_latent_test[2,:,:],sig_latent_test[2,:,:] , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sampling', tag_title=self.vae_model +" Sampling", bool_norm=True)
    """

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch 
    for loss in ['loss', 'reco_loss', 'kl_loss']:
      try:plot_loss(h2, loss=loss, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
      except: print(f'loading vae_model so cannot draw regular {loss} plot -> no h2')
    loss_dict_all={}
    print(bkg_train.shape)
    loss_dict_train, methods,y = self.calculate_loss(model = vae, bkg = bkg_train, sig = sig_train, y_bkg = y_bkg_train, y_sig = y_sig_train, bool_vae=True)
    loss_dict_all['train'] = loss_dict_train # cannot calculate a metric b/c sig_train is empty
    loss_dict_test, methods,y  = self.calculate_loss(model = vae, bkg = bkg_test, sig = sig_test, y_bkg = y_bkg_test, y_sig = y_sig_test, bool_vae=True)
    loss_dict_all['test'] = self.calculate_metric(loss_dict_test, methods=methods,y=y)
    example=np.array([1.1, 2.2, 2.2, 2.2, 3.3, 3.3])
    print(f'{example=}')
    self.freq_table(example, text='example')
    sic_vals_dict = self.plot_loss_dict(loss_dict_all)
    #sic_vals_dict= {np.nan, np.nan, np.nan}
    bkg_events_num,sig_events_num=np.nan, np.nan

    return self.all_dir, sic_vals_dict, bkg_events_num,sig_events_num

  def prepare_test_ECG(self, bool_float64):
#  def prepare_test_ECG(self):
    #https://www.tensorflow.org/tutorials/generative/autoencoder
    # classify an ECG as anomalous if the total reconstruction error is greater than one standard deviation 
    dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
    raw_data = dataframe.values
    dataframe.head()
    y=raw_data[:,-1] # label
    x=raw_data[:, 0:-1]

    x_labels = y.astype(bool)

    # Raw data    
    # only select the first 8 since there are too many (140.. since the dimension is (# samples, 140) )
    normal_x = x[x_labels]
    anomalous_x = x[~x_labels]
    var='raw'
    plot_1D_phi(normal_x[:,:7],anomalous_x[:,:7],labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.pfn_model+f'_{var}', tag_title=self.pfn_model+f' {var.capitalize()}')
    plot_phi(normal_x,tag_file="ECG_normal_x"+f'_{var}', tag_title="Normal"+f' {var.capitalize()}', plot_dir=self.plot_dir)
    plot_phi(anomalous_x,tag_file="ECG_anomalous_x"+f'_{var}', tag_title="Anomalous"+f' {var.capitalize()}', plot_dir=self.plot_dir)
    x = x * self.scalar_ecg
   # scale
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    print('before',type(x), x.shape)  
    x = (x - min_val) / (max_val - min_val)
    # change the datatype for precision
    if bool_float64:
      x = tf.cast(x, tf.float64)
    x= x.numpy()

    # Scale data    
    normal_x = x[x_labels]
    anomalous_x = x[~x_labels]
    var='normalized'
    plot_1D_phi(normal_x[:,:7],anomalous_x[:,:7],labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.pfn_model+f'_{var}', tag_title=self.pfn_model+f' {var.capitalize()}')
    plot_phi(normal_x,tag_file="ECG_normal_x"+f'_{var}', tag_title="Normal"+f' {var.capitalize()}', plot_dir=self.plot_dir)
    plot_phi(anomalous_x,tag_file="ECG_anomalous_x"+f'_{var}', tag_title="Anomalous"+f' {var.capitalize()}', plot_dir=self.plot_dir)


    # Multiplied by scalar_ecg data    
    normal_x = x[x_labels]
    anomalous_x = x[~x_labels]
    var='scaled'
    plot_1D_phi(normal_x[:,:7],anomalous_x[:,:7],labels=['normal', 'anomalous'], plot_dir=self.plot_dir, tag_file=self.pfn_model+'_input', tag_title=self.pfn_model+f' Input (x{self.scalar_ecg})')
    plot_phi(normal_x,tag_file="ECG_normal_x_scaled", tag_title=f"Normal (x{self.scalar_ecg})", plot_dir=self.plot_dir)
    plot_phi(anomalous_x,tag_file="ECG_anomalous_x_scaled", tag_title=f"Anomalous (x{self.scalar_ecg})", plot_dir=self.plot_dir)
   # scale
    print('after converting',type(x),f'{x.dtype=}', x.shape)

#    x = x.astype('float64')

    # split 
    x_evalb,  x_testb, y_evalb, y_testb = train_test_split(x, y, test_size=0.2)
    print(f'{x_evalb.shape=}, {x_testb.shape=}, {y_evalb.shape=}, {y_testb.shape=}, {np.unique(np.array(y_evalb))=}')
    # shuffle before flattening!
    x_evalb, y_evalb=self.shuffle_two(x_evalb, y_evalb)
    x_testb, y_testb=self.shuffle_two(x_testb, y_testb)
    print(f'before{x_evalb.shape=}, before{x_evalb.dtype=}, before{y_evalb.dtype=}')

#    x_evalb, x_testb= x_evalb.astype('float32'), x_testb.astype('float32') 
#    y_evalb, y_testb= y_evalb.astype('float32'), y_testb.astype('float32') 

    # separate the normal rhythms from abnormal rhythms
    test_labels = y_testb.astype(bool)
    normal_test = x_testb[test_labels]
    anomalous_test = x_testb[~test_labels]
    y_normal_test = y_testb[test_labels]
    y_anomalous_test = y_testb[~test_labels]

    train_labels = y_evalb.astype(bool)
    normal_train = x_evalb[train_labels]
    anomalous_train = x_evalb[~train_labels]
    y_normal_train = y_evalb[train_labels]
    y_anomalous_train = y_evalb[~train_labels]

    # Only train with normal train data -> remove all the
    x_evalb = normal_train
    y_evalb = y_normal_train
    print(f'{normal_x.shape=},{anomalous_x.shape=}')
    print(f'{normal_train.shape=},{anomalous_train.shape=}')
    print(f'{normal_test.shape=},{anomalous_test.shape=}')
    print(f'{x_evalb.shape=}')
    print(f'{x_testb.shape=}')
    print(f'{x_evalb.shape[0]=}={normal_train.shape[0]=}')
    print(f'{anomalous_train.shape[0]=}=({anomalous_x.shape[0]=}-{anomalous_test.shape[0]=})')
    print(anomalous_train.shape[0], anomalous_x.shape[0]-anomalous_test.shape[0])
    print(f'({x_evalb.shape[0]=}+{x_testb.shape[0]=}-{normal_test.shape[0]=}-{normal_train.shape[0]=}-{anomalous_test.shape[0]=})=0')
    print(x_evalb.shape[0]+x_testb.shape[0]-normal_test.shape[0]-normal_train.shape[0]-anomalous_test.shape[0])

    # plot test input
    self.plot_compare_ECG(x =normal_test,label='input',title1 = 'input' ,title2='normal')
    self.plot_compare_ECG(x =anomalous_test,label='input',title1 = 'input' ,title2='anomalous')

    print(f'{np.max(x_evalb)=}')
    """
    # validation data set manually
    # Prepare the training dataset
    idx = np.random.choice( np.arange(len(x_evalb)), size= round(.2 *len(x_evalb)) , replace=False) # IMPT that replace=False so that event is picked only once
    idx = np.sort(idx) 
    # Prepare the validation dataset
    x_evalb_val = x_evalb[idx, :] 
    x_evalb_train = np.delete(x_evalb, idx) # doesn't modify input array 
    print(f'{x_evalb_val.shape=}, {x_evalb_train.shape=}')     
    
    x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val = train_test_split(x_evalb, y_evalb, test_size=round(.2 *len(x_evalb)))
    #phi_evalb_train, phi_evalb_val, _, _ = train_testb_split(phi_evalb, phi_evalb, testb_size=round(.2 *len(phi_evalb)))
    #return  x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val
    """
    return  x_testb, x_evalb, y_testb, y_evalb


  def prepare_pfn(self, graph,scaler,  bool_float64):
    print('AE not VAE')
    # phi_bkg -> (phi_evalb and phi_testb) ; phi_sig 
    # scale values
    track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e","jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
    track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e","jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
    jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"] # order is important in apply_JetScalingRotation

    bool_weight_sig=False
    #sig, mT_sig, sig_sel, jet_sig, sig_in0, sig_in1 = getTwoJetSystem(nevents=self.sig_events,input_file=self.sig_file,
    sig, mT_sig, _, _, _, _ = getTwoJetSystem(nevents=self.sig_events,input_file=self.sig_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight_sig,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed,max_track=self.max_track, bool_pt=self.bool_pt,h5_dir=self.h5_dir,       read_dir='/data/users/ebusch/SVJ/autoencoder/v8.1/')    
    #bkg, mT_bkg, bkg_sel, jet_bkg,bkg_in0, bkg_in1 = getTwoJetSystem(nevents=self.bkg_events,input_file=self.bkg_file,
    bkg, mT_bkg, _, _, _, _ = getTwoJetSystem(nevents=self.bkg_events,input_file=self.bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=self.bool_weight,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed, max_track=self.max_track,bool_pt=self.bool_pt,h5_dir=self.h5_dir,
      read_dir='/data/users/kpark/SVJ/MicroNTuples/v9.2/') 
      #read_dir='/data/users/ebusch/SVJ/autoencoder/v9.1/') 
    print('hola') 
    bkg2,_ = apply_StandardScaling(bkg,scaler,False) # change
    sig2,_ = apply_StandardScaling(sig,scaler,False) # change
    plot_vectors(bkg2,sig2,tag_file="ANTELOPE", tag_title=" (ANTELOPE)", plot_dir=self.plot_dir)# change
    
    phi_bkg = graph.predict(bkg2)
    phi_sig = graph.predict(sig2)

#    """
    if bool_float64: 
      phi_bkg = phi_bkg.astype('float64')
      phi_sig = phi_sig.astype('float64')
      print('using float64 for phi in prepare_pfn')
    else:
      print('NOT using float64 for phi in prepare_pfn') 
 #   """ 
    print(f'{phi_bkg.dtype=}, {phi_sig.dtype=}')
    """
    plot_1D_phi(phi_bkg, phi_sig,labels=['QCD', 'sig'], plot_dir=self.plot_dir, tag_file=self.pfn_model, tag_title=self.pfn_model)
    """
    
    #phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0], random_state=42) # no info on labels e.g. y_train or y_test
    phi_evalb_idx, phi_testb_idx, _, _ = train_test_split(np.arange(len(phi_bkg)), phi_bkg, test_size=sig2.shape[0])
    phi_evalb, phi_testb, mT_evalb, mT_testb= phi_bkg[phi_evalb_idx, :], phi_bkg[phi_testb_idx, :], mT_bkg[phi_evalb_idx,:], mT_bkg[phi_testb_idx, :]
    print('idx',phi_evalb_idx, phi_testb_idx)
    print('after',phi_evalb.shape, phi_testb.shape,  mT_evalb.shape, mT_testb.shape)

    plot_single_variable([mT_evalb[:,2], mT_testb[:,2]],h_names= ['training and validation', 'test'],weights_ls=[mT_evalb[:,1], mT_testb[:,1]], tag_title= 'leading jet pT (QCD)', plot_dir=self.plot_dir,logy=True, tag_file='jet1_pt', bool_weight=self.bool_weight)
    plot_single_variable([mT_evalb[:,3], mT_testb[:,3]],h_names= ['training and validation', 'test'],weights_ls=[mT_evalb[:,1], mT_testb[:,1]], tag_title= 'subleading jet pT (QCD)', plot_dir=self.plot_dir, logy=True, tag_file='jet2_pt',bool_weight=self.bool_weight)

    #plot_phi(phi_evalb,tag_file="PFN_phi_train_raw",tag_title="Train",plot_dir=self.plot_dir) # change
    #plot_phi(phi_testb,tag_file="PFN_phi_test_raw",tag_title="Test", plot_dir=self.plot_dir)
    plot_phi(phi_sig,tag_file="PFN_phi_sig_raw", tag_title="Signal", plot_dir=self.plot_dir)
    plot_phi(phi_bkg,tag_file="PFN_phi_bkg_raw",tag_title="QCD",plot_dir=self.plot_dir) # change
     
    if self.bool_no_scaling:
      plot_1D_phi(phi_bkg, phi_sig,labels=['QCD', 'sig'], plot_dir=self.plot_dir, tag_file=self.pfn_model+'_input', tag_title=self.pfn_model+' Input')
      plot_phi(phi_sig,tag_file="PFN_phi_sig_input", tag_title="Signal Input", plot_dir=self.plot_dir)
      plot_phi(phi_bkg,tag_file="PFN_phi_bkg_input", tag_title="QCD Input", plot_dir=self.plot_dir)
      print('---no scaling or shifting for phis---')
      return  phi_bkg, phi_testb, phi_evalb, phi_sig

    eval_max = np.amax(phi_evalb)
    eval_min = np.amin(phi_evalb)
    sig_max = np.amax(phi_sig)
    print("Min: ", eval_min)
    print("Max: ", eval_max)
    if (sig_max > eval_max): eval_max = sig_max
    print("Final Max: ", eval_max)

    phi_evalb_pre = (phi_evalb - eval_min)/(eval_max-eval_min)
    phi_testb_pre = (phi_testb - eval_min)/(eval_max-eval_min)
    phi_sig_pre = (phi_sig - eval_min)/(eval_max-eval_min)
    phi_bkg_pre = (phi_bkg - eval_min)/(eval_max-eval_min)

    phi_evalb = scale_phi(phi_evalb, max_phi=eval_max, bool_nonzero=self.bool_nonzero)    
    phi_testb = scale_phi(phi_testb, max_phi=eval_max, bool_nonzero=self.bool_nonzero)    
    phi_sig = scale_phi(phi_sig, max_phi=eval_max, bool_nonzero=self.bool_nonzero)    
    phi_bkg = scale_phi(phi_bkg, max_phi=eval_max, bool_nonzero=self.bool_nonzero)   

 
    """
    plot_1D_phi(phi_bkg_pre, phi_sig_pre,labels=['QCD', 'sig'], plot_dir=self.plot_dir, tag_file=self.pfn_model+'_scaling', tag_title=self.pfn_model + ' (after scaling)')
    plot_1D_phi(phi_bkg, phi_sig,labels=['QCD', 'sig'], plot_dir=self.plot_dir, tag_file=self.pfn_model+'_shifting', tag_title=self.pfn_model + f'(after shifting)')
    # set representation plot
    plot_phi(phi_sig,tag_file="PFN_phi_sig_shifted", tag_title="Signal Shifted", plot_dir=self.plot_dir)
    plot_phi(phi_bkg,tag_file="PFN_phi_bkg_shifted",tag_title="QCD Shifted",plot_dir=self.plot_dir) # change

    
    plot_phi(phi_sig_pre,tag_file="PFN_phi_sig_scaled", tag_title="Signal Scaled", plot_dir=self.plot_dir)
    plot_phi(phi_bkg_pre,tag_file="PFN_phi_bkg_scaled", tag_title="QCD Scaled", plot_dir=self.plot_dir)
    """
    # validation data set manually
    # Prepare the training dataset
    """
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
    """

    if not(self.bool_shift): phi_bkg, phi_testb, phi_evalb, phi_sig = phi_bkg_pre, phi_testb_pre, phi_evalb_pre, phi_sig_pre
    plot_phi(phi_sig,tag_file="PFN_phi_sig_input", tag_title="Signal Input", plot_dir=self.plot_dir)
    plot_phi(phi_bkg,tag_file="PFN_phi_bkg_input", tag_title="QCD Input", plot_dir=self.plot_dir)
    plot_1D_phi(phi_bkg, phi_sig,labels=['QCD', 'sig'], plot_dir=self.plot_dir, tag_file=self.pfn_model+'_input', tag_title=self.pfn_model+' Input')
    return  phi_bkg, phi_testb, phi_evalb, phi_sig
 

  def train_vae(self, phi_evalb, y_phi_evalb=[]):
  #def train_vae(self, phi_evalb_train, phi_evalb_val, y_phi_evalb_train=[], y_phi_evalb_val=[]):

    #vae = get_ECG_ae(self.phi_dim,self.encoding_dim,self.latent_dim)
    vae = get_vae(input_dim=self.phi_dim,encoding_dim=self.encoding_dim,latent_dim=self.latent_dim, learning_rate=self.learning_rate,kl_loss_scalar= self.kl_loss_scalar, bool_test=False, scalar_ecg=self.scalar_ecg, decoder_activation=self.decoder_activation,  bool_float64=self.bool_float64)
      
#    h2 = vae.fit(phi_evalb_train, 
    h2 = vae.fit(phi_evalb, 
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

  def calculate_loss(self,model,bkg,sig, y_bkg, y_sig, bool_vae=True, bool_transformed=True ): # move
    if y_sig.size>0:    assert len(sig)==len(y_sig)
    if y_bkg.size>0:    assert len(bkg)==len(y_bkg)
      
    pred_bkg = model.predict(bkg)['reconstruction']
    loss_dict={}
    methods=['mse', 'multi_mse', 'multi_kl', 'multi_reco'] # if this is changed, the code below of defining also should be changed
    #methods=['mse', 'mae', 'multi_mse', 'multi_kl', 'multi_reco']
    # methods=['mse', 'mae'] # for autoencoder 
    # using 'non' reduction type -> default is 'sum_over_batch_size'
    loss_dict['mse']={}
    mse = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss_dict['mse']['bkg'] = mse(bkg, pred_bkg).numpy()
    """
    loss_dict['mae']={}
    mae = keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    loss_dict['mae']['bkg'] = mae(bkg, pred_bkg).numpy()
    """
    if bool_vae: 
      loss_dict['multi_mse'], loss_dict['multi_kl'], loss_dict['multi_reco']= {}, {}, {}
      loss_dict['multi_mse']['bkg'], loss_dict['multi_kl']['bkg'], loss_dict['multi_reco']['bkg']= get_multi_loss_each(model, bkg, step_size=self.step_size)
    if len(sig)!=0:
      if len(y_bkg)==0: y=np.array([]) # if y is not known, then make an empty array
      else: y= np.concatenate((y_bkg,y_sig), axis=0) 
      pred_sig = model.predict(sig)['reconstruction']
      loss_dict['mse']['sig'] = mse(sig, pred_sig).numpy()
#      loss_dict['mae']['sig'] = mae(sig, pred_sig).numpy()
      if bool_vae:
        loss_dict['multi_mse']['sig'], loss_dict['multi_kl']['sig'], loss_dict['multi_reco']['sig']= get_multi_loss_each(model, sig, step_size=self.step_size)
    else: y=y_bkg

   
    if bool_transformed:
      old_methods=methods.copy() # essential that new_methods and methods are separate b/c otherwise, will loop through methods that are already transformed
      for method in old_methods:
        new_method=f'{method}_transformed'
        print(f'{method=}, {new_method=}')
        loss_dict[new_method+'_log10_sig']={}
        loss_dict[new_method+'_log10']={}
        loss_dict[new_method+'_negx']={}
        loss_bkg=np.log10(loss_dict[method]['bkg'])
      
        if len(sig)!=0:
          loss_sig=np.log10(loss_dict[method]['sig'])

          """
          loss_both= np.concatenate((loss_bkg, loss_sig))
        else:loss_both=loss_bkg
        max_loss=np.max(loss_both)
        min_loss=np.min(loss_both)
        print(f'{max_loss=}, {min_loss=}, {loss_bkg[:5]}') 
        loss_transformed_bkg = (loss_bkg - min_loss)/(max_loss -min_loss) 

          """
        loss_dict[new_method+'_log10']['bkg'] =loss_bkg 
        loss_transformed_bkg = 1/(1 + np.exp(-loss_bkg)) 
        loss_dict[new_method+'_log10_sig']['bkg'] =loss_transformed_bkg 
        loss_transformed_bkg = 1/(1 + np.exp(loss_bkg)) 
        loss_dict[new_method+'_negx']['bkg'] =loss_transformed_bkg 
        if len(sig)!=0:
          """
          loss_transformed_sig = (loss_sig - min_loss)/(max_loss -min_loss) 
          """
          loss_dict[new_method+'_log10']['sig'] =loss_sig 
          loss_transformed_sig = 1/(1 + np.exp(-loss_sig)) 
          loss_dict[new_method+'_log10_sig']['sig'] =loss_transformed_sig 
          loss_transformed_sig = 1/(1 + np.exp(loss_sig)) 
          loss_dict[new_method+'_negx']['sig'] =loss_transformed_sig 
  
        methods.append(new_method+'_log10_sig')
        methods.append(new_method+'_log10')
        methods.append(new_method+'_negx')

    return loss_dict, methods, y

  def freq_table(self,array, text=''):
      """
      (values, counts) = np.unique(array, return_counts=True)
      freq_table = np.asarray((values, counts)).T
      
      print('-'*15+'frequency table' +'-'*15)
      print(f'{array.dtype=}')
      print(f'{np.array(["values", "counts"])=}')
      print(freq_table)
      print('-'*30)
      """
      print(text)
      print(f'{array.dtype=}')
      data=pd.Series(array)
      freq_table=data.value_counts(sort=True)
      print(freq_table)
      return freq_table

  def calculate_metric(self, loss_dict, methods, y):
    y = y.astype(bool)
    for method in methods:
      loss_dict[method]['all']= np.concatenate((loss_dict[method]['sig'],loss_dict[method]['bkg']), axis=0)
      if method == 'mae': 
        loss_dict[method]['threshold'] = np.mean(loss_dict[method]['all']) + np.std(loss_dict[method]['all'])
      else: 
        loss_dict[method]['threshold'] = np.mean(loss_dict[method]['all']) + np.std(loss_dict[method]['all'])
      print(f'{method}, {y.shape}, {loss_dict[method]["all"].shape}, {loss_dict[method]["sig"].shape}, {loss_dict[method]["bkg"].shape}')
      loss_dict[method]['pred']=np.less(loss_dict[method]['all'], loss_dict[method]['threshold']) # returns True if smaller than threshold -> what we want /smaller than 1 std dev
      print(f'{y[:5]=}')
      print(f'{loss_dict[method]["all"][:5]=}')
      print(f'{loss_dict[method]["threshold"]=}')
      print(f'{loss_dict[method]["pred"][:5]=}')
      if len(y)==0:
     
        accuracy_score_ex, precision_score_ex, recall_score_ex = np.nan, np.nan, np.nan
      else: 
        accuracy_score_ex,precision_score_ex, recall_score_ex =accuracy_score(y, loss_dict[method]['pred']), precision_score(y, loss_dict[method]['pred']), recall_score(y, loss_dict[method]['pred'])
      loss_dict[method]['accuracy']=accuracy_score_ex
      loss_dict[method]['precision']=precision_score_ex
      loss_dict[method]['recall']=recall_score_ex
    
      # Create frequency table
      self.freq_table(loss_dict[method]['all'], text=method)
     
    return loss_dict # loss_dict[method]={'bkg', 'sig', 'all', 'threshold', 'accuracy', 'precision'} 

  def plot_loss_dict(self, loss_dict_all):
    sic_vals_dict={} 
    for key in loss_dict_all:    # train or test 
      for method in loss_dict_all[key]: # mse, mae, etc
        print(f'{method=} hereee')
        bkg_loss=loss_dict_all[key][method]['bkg'] 
        if key=='test':sig_loss=loss_dict_all[key][method]['sig']
        else: sig_loss=np.array([]) 
        
        if key=='test':
          self.metric_dict[method]={}
          """
          for metric in {'accuracy', 'precision', 'recall'}:
            self.metric_dict[method][metric]=loss_dict_all[key][method][metric]
            print(f'{method}{metric},{loss_dict_all[key][method][metric]=}')
          """
    # # 3. Signal Sensitivity Score
    # Choose a threshold value that is one standard deviations above the mean.
        if key=='test':
          score = getSignalSensitivityScore(bkg_loss, sig_loss)
          print("95 percentile score = ",score)
          print(f'plotting roc of', key,method)
          # # 4. ROCs/AUCs using sklearn functions imported above 
          sic_vals=do_roc(bkg_loss, sig_loss, tag_file=self.vae_model+f'_{method}', tag_title=self.vae_model+ f' (step size={self.step_size} {method})',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
          try:sic_vals=do_roc(bkg_loss[bkg_loss>0], sig_loss[sig_loss>0], tag_file=self.vae_model+f'_{method}_pos', tag_title=self.vae_model+ f' (step size={self.step_size} {method}, score>0)',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
          except: print('sic_vals=do_roc(bkg_loss[bkg_loss>0] didnot work -> bkg_loss and sig_loss might be all negative')
          sic_vals_dict[method]=sic_vals
          auc=sic_vals['auc']
  
        else:auc=np.nan 

#        plot_score(bkg_loss, sig_loss, False, xlog=True, tag_file=self.vae_model+f'_{method}_{key}', tag_title=self.vae_model+f' {method} {key}', plot_dir=self.plot_dir, bool_pfn=False, auc=auc) # anomaly score
      # xlog=True plots
        if '_trans' not in method:
          plot_score(bkg_loss[bkg_loss>0], sig_loss[sig_loss>0], False, xlog=True, tag_file=self.vae_model+'_pos'+f'_{method}_{key}', tag_title=self.vae_model + ' (score > 0)'+f' {method} {key}', plot_dir=self.plot_dir, bool_pfn=False, auc=auc) # anomaly score
  
      # xlog=False plots
        plot_score(bkg_loss, sig_loss, False, xlog=False, tag_file=self.vae_model+f'_{method}_{key}', tag_title=self.vae_model+f' {method} {key}', plot_dir=self.plot_dir, bool_pfn=False, auc=auc) # anomaly score
      # xlog= False plots plot only points less than 0 
        plot_score(bkg_loss[bkg_loss<=0], sig_loss[sig_loss<=0], False, xlog=False, tag_file=self.vae_model+f'_neg_{method}_{key}', tag_title=self.vae_model+f' (score <= 0) {method} {key}', plot_dir=self.plot_dir, bool_pfn=False, auc=auc) # anomaly score

    return sic_vals_dict


  def evaluate_vae(self):
    graph, scaler = self.load_pfn()
    x_bkg,x_testb, x_evalb, x_sig=  self.prepare_pfn(graph,scaler,  bool_float64=self.bool_float64)
    print('prepare_pfn')
    try: vae = self.load_vae()
    except: 
      print('loading vae not successful so will start the training process')
      vae,h2 = self.train_vae( x_evalb)
      #vae,h2 = self.train_vae( x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val)
      print('training successful')

    # if want to use validation_split instead of manual splitting

    x_evalb_train = x_evalb

    bkg_latent_test=vae.get_layer('encoder').predict(x_testb)
    bkg_latent_train=vae.get_layer('encoder').predict(x_evalb_train)
#    bkg_latent_val=vae.get_layer('encoder').predict(x_evalb_val)

     # specific to PFN input
    sig_latent=vae.get_layer('encoder').predict(x_sig)

    # specific to PFN input
    #latent_test is a list but latent_test[0] is a numpy array
    bkg_latent_test, bkg_latent_train, sig_latent=np.array(bkg_latent_test), np.array(bkg_latent_train), np.array(sig_latent)
    sig_latent_test=sig_latent # all signals are test

    # bkg as zeros and signal as ones
    latent_test = np.concatenate((bkg_latent_test, sig_latent_test), axis=1)
    y_testb = np.concatenate((np.zeros(bkg_latent_test.shape[1]), np.ones(sig_latent_test.shape[1])), axis = 0)
    y_evalb_train = np.zeros(bkg_latent_train.shape[1])

    print(f'{latent_test.shape=}')
    print(f'{bkg_latent_test.shape=}')
    print(f'{y_testb.shape=}')
    #""" 
    var='mu'     
    plot_pca(latent_test[0,:,:], latent_label=np.array(y_testb), nlabel=10,n_components=2, tag_file=self.vae_model+f'_test_{var}', tag_title=self.vae_model+f' Test {var.capitalize()}', plot_dir=self.plot_dir)
    plot_pca(bkg_latent_train[0,:,:],latent_label=np.array(y_evalb_train), nlabel=10, n_components=2, tag_file=self.vae_model+f'_train_{var}', tag_title=self.vae_model+f' Train {var.capitalize()}', plot_dir=self.plot_dir)

    var='sigma'     
    plot_pca(latent_test[1,:,:], latent_label=np.array(y_testb), nlabel=10,n_components=2, tag_file=self.vae_model+f'_test_{var}', tag_title=self.vae_model+f' Test {var.capitalize()}', plot_dir=self.plot_dir)
    plot_pca(bkg_latent_train[1,:,:],latent_label=np.array(y_evalb_train), nlabel=10, n_components=2, tag_file=self.vae_model+f'_train_{var}', tag_title=self.vae_model+f' Train {var.capitalize()}', plot_dir=self.plot_dir)

    var='sampling'     
    plot_pca(latent_test[2,:,:], latent_label=np.array(y_testb), nlabel=10,n_components=2, tag_file=self.vae_model+f'_test_{var}', tag_title=self.vae_model+f' Test {var.capitalize()}', plot_dir=self.plot_dir)
    plot_pca(bkg_latent_train[2,:,:],latent_label=np.array(y_evalb_train), nlabel=10, n_components=2, tag_file=self.vae_model+f'_train_{var}', tag_title=self.vae_model+f' Train {var.capitalize()}', plot_dir=self.plot_dir)

    #""" 
    y_bkg_train, y_bkg_test, y_sig_train, y_sig_test = np.array([]), np.array([]),  np.array([]), np.array([]) 

    bkg_latent_test_recon = vae.get_layer('decoder').predict(bkg_latent_test[0,:,:])
    sig_latent_test_recon = vae.get_layer('decoder').predict(sig_latent_test[0,:,:])
    print(f'{bkg_latent_test_recon.shape=}')
    print(f'{sig_latent_test_recon.shape=}')
   
    bkg_latent_test_sigma, sig_latent_test_sigma = self.transform_sigma(bkg_latent_test[1,:,:]), self.transform_sigma(sig_latent_test[1, :,:])
    labels=['test QCD', 'SIG']
    #"""
    plot_1D_phi(bkg_latent_test[0,:,:],sig_latent_test[0,:,:] , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_mu', tag_title=self.vae_model +r" $\mu$", ylog=True)
    plot_1D_phi(bkg_latent_test_sigma,sig_latent_test_sigma , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sigma', tag_title=self.vae_model +r" $\sigma$", ylog=True)

    plot_1D_phi(bkg_latent_test[0,:,:],sig_latent_test[0,:,:] , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_mu_custom', tag_title=self.vae_model +r" $\mu$",  bins=np.linspace(-0.0001,0.0001, num=50))
    plot_1D_phi(bkg_latent_test_sigma,sig_latent_test_sigma , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sigma_custom', tag_title=self.vae_model +r" $\sigma$", bins=np.linspace(0.9998,1.0002,num=50))

    plot_1D_phi(bkg_latent_test[2,:,:],sig_latent_test[2,:,:] , labels=labels, plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_sampling', tag_title=self.vae_model +" Sampling", bool_norm=True)
    #"""

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch
 
    for loss in ['loss', 'reco_loss', 'kl_loss']:
      try:plot_loss(h2, loss=loss, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
      except: print(f'loading vae_model so cannot draw regular {loss} plot -> no h2')
    #plot_loss(h2, vae_model, "kl_loss")
    #plot_loss(h2, vae_model, "reco_loss")
    
#    """
    loss_dict_all={}
    loss_dict_train, methods,y = self.calculate_loss(model = vae, bkg = x_evalb_train, sig =np.array([]) , y_bkg = y_bkg_train, y_sig = y_sig_train)
    loss_dict_all['train'] = loss_dict_train # cannot calculate a metric b/c sig_train is empty
    loss_dict_test, methods,y  = self.calculate_loss(model = vae, bkg = x_testb, sig = x_sig, y_bkg = y_bkg_test, y_sig = y_sig_test)
    loss_dict_all['test'] = self.calculate_metric(loss_dict_test, methods,y)
    sic_vals_dict = self.plot_loss_dict(loss_dict_all)
    bkg_events_num,sig_events_num=np.nan, np.nan 
    
    return self.all_dir, sic_vals_dict, bkg_events_num,sig_events_num

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
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, arch_dir_vae='/data/users/kpark/svj-vae/results/grid_sept26/09_26_23_10_38/architectures_saved/', step_size=1)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, arch_dir_vae='/data/users/kpark/svj-vae/results/grid_sept26/09_27_23_01_32/architectures_saved/', step_size=1, bool_no_scaling=True)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, step_size=1, bool_nonzero=False, bool_shift=True)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, step_size=1, bool_nonzero=True, bool_shift=True)
      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar=kl_loss_scalar, step_size=1, bool_no_scaling=True, decoder_activation='relu', bool_float64=False, bool_weight=False)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar=kl_loss_scalar, step_size=1) # MAKE SURE TO CHECK RELU VS SIGMOID 
      #extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'],  step_size=1, bool_no_scaling=True, kl_loss_scalar=1000)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim=140, encoding_dim=16, latent_dim=8)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim=140, encoding_dim=16, latent_dim=8, scalar_ecg=10)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim= 784, encoding_dim=196, latent_dim=49)# if flattening
    
    stdoutOrigin=param1.open_print()
#    all_dir, sic_vals_dict,bkg_events_num,sig_events_num=param1.evaluate_test_ECG()
    print('using relu as activation function in decoder instead of sigmoid')
    all_dir, sic_vals_dict,bkg_events_num,sig_events_num=param1.evaluate_vae()
    #all_dir, sic_vals_dict,bkg_events_num,sig_events_num=param1.evaluate_vae(bool_pfn=False)
    setattr(param1, 'sic_vals_dict',sic_vals_dict )
    setattr(param1, 'sig_events_num',sig_events_num )
    setattr(param1, 'bkg_events_num',bkg_events_num )
    print(param1.close_print(stdoutOrigin))
    print(param1.save_info())
"""
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
