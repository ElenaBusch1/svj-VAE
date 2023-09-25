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
from svj_antelope import Param_ANTELOPE
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

class Param_test(Param_ANTELOPE):
  def __init__(self, arch_dir_pfn, arch_dir_vae='',kl_loss_scalar=100,
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
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim0.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'],seed=0 ):

    super().__init__(arch_dir,print_dir,plot_dir,h5_dir,
      pfn_model, vae_model, bkg_events, sig_events,
      num_elements, element_size, encoding_dim, latent_dim, phi_dim, nepochs, n_neuron, learning_rate,
      nlayer_phi, nlayer_F,
      max_track,
      batchsize_pfn,
      batchsize_vae,
      bool_pt,
      sig_file, bkg_file,  bool_weight, extraVars,seed)

    """
#    super().__init__(
      arch_dir_pfn,
      arch_dir_vae,
      kl_loss_scalar,
      step_size,
      metric_dict,
      bool_shift,
      bool_no_scaling, # this overrides bool_shift
      bool_nonzero,
      scalar_ecg )
    """
    self.arch_dir_pfn=arch_dir_pfn
    self.arch_dir_vae=arch_dir_vae
    self.kl_loss_scalar=kl_loss_scalar
    self.step_size=step_size
    self.metric_dict=metric_dict
    self.bool_shift=bool_shift
    self.bool_no_scaling=bool_no_scaling # this overrides bool_shift
    self.bool_nonzero=bool_nonzero
    self.scalar_ecg=scalar_ecg
    #super(Param_ANTELOPE, self).__init__()

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
    x_testb, x_evalb, y_testb, y_evalb= self.prepare_test_ECG(scalar_ecg=self.scalar_ecg)

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
    try:plot_loss(h2, loss='loss', tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
    except: print('loading vae_model so cannot draw regular loss plot -> no h2')
    loss_dict_all={}
    print(bkg_train.shape)
    loss_dict_train, methods,y = self.calculate_loss(model = vae, bkg = bkg_train, sig = sig_train, y_bkg = y_bkg_train, y_sig = y_sig_train, bool_vae=True)
    loss_dict_all['train'] = loss_dict_train # cannot calculate a metric b/c sig_train is empty
    loss_dict_test, methods,y  = self.calculate_loss(model = vae, bkg = bkg_test, sig = sig_test, y_bkg = y_bkg_test, y_sig = y_sig_test, bool_vae=True)
    loss_dict_all['test'] = self.calculate_metric(loss_dict_test, methods=methods,y=y)
    example=np.array([1.1, 2.2, 2.2, 2.2, 3.3, 3.3])
    print(f'{example=}')
    self.freq_table(example, text='example')
    auc = self.plot_loss_dict(loss_dict_all)
    #auc= {np.nan, np.nan, np.nan}
    bkg_events_num,sig_events_num=np.nan, np.nan

    return self.all_dir, auc, bkg_events_num,sig_events_num



  def prepare_test_ECG(self, scalar_ecg):
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
    x = x * scalar_ecg
    """
   # scale
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    print('before',type(x), x.shape)  
    x = (x - min_val) / (max_val - min_val)
    """
    # change the datatype for precision
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


  def prepare_test(self):
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
    nsample=9
    x_testb_plt= x_testb[:nsample]
    # reshape
    if bool_flat:
      x_testb_plt=x_testb.reshape(x_testb.shape[0], 28,28, 1) # should be (x, 28, 28,1)
    cprint(f'reshape {x_testb.shape=}')
    fig = plt.figure()

    for i in range(nsample):
      ax = fig.add_subplot(3, 3, i+1)
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
    nsample=9
    latent_test_recon= latent_test_recon[:nsample]
    # reshape
    latent_test_recon=latent_test_recon.reshape(latent_test_recon.shape[0], 28, -1) # should be (x, 28, 28)
    cprint(f'reshape 1 {latent_test_recon.shape=}')
    latent_test_recon=latent_test_recon.reshape(latent_test_recon.shape[0], 28,28, 1) # should be (x, 28, 28,1)
    cprint(f'reshape 2 {latent_test_recon.shape=}')
    fig = plt.figure()

    for i in range(nsample):
      ax = fig.add_subplot(3, 3, i+1)
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
    #step_size=self.batchsize_vae
    bkg_loss,  bkg_kl_loss,  bkg_reco_loss  = get_multi_loss_each(vae, x_testb, step_size=self.step_size)
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
    #start = time.time()
#    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig)
    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig, step_size=self.step_size)
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

if __name__=="__main__":
  pfn_model='PFNv6'
  print('hi')
  """
  sig_events=1151555
  bkg_events=3234186
  ls_sig=[100000]
  ls_bkg=[500000]
  """
  ls_sig=[20000]
  ls_bkg=[200000]
  for  kl_loss_scalar in [1]:
    param1=Param_test(pfn_model=pfn_model,  h5_dir='h5dir/antelope/aug17_jetpt/', arch_dir_pfn='/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/',
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, arch_dir_vae='/data/users/kpark/svj-vae/results/test/09_11_23_11_38/architectures_saved/', step_size=100)
      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim=140, encoding_dim=16, latent_dim=8)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim=140, encoding_dim=16, latent_dim=8, scalar_ecg=10)
#      extraVars=['mT_jj', 'weight', 'jet1_pt', 'jet2_pt'], kl_loss_scalar= kl_loss_scalar, phi_dim= 784, encoding_dim=196, latent_dim=49)# if flattening

    stdoutOrigin=param1.open_print()
    #sys.exit()
    all_dir, auc,bkg_events_num,sig_events_num=param1.evaluate_test_ECG()
    setattr(param1, 'auc',auc )
    setattr(param1, 'sig_events_num',sig_events_num )
    setattr(param1, 'bkg_events_num',bkg_events_num )
    print(param1.close_print(stdoutOrigin))
    print(param1.save_info())
                                                                                                                                                              
