import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import *
from root_to_numpy import *
from plot_helper import *
from eval_helper import *

import sys
import time
import pandas as pd
#plot_dir='/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun29/' 
# Example usage
#added
"""import matplotlib.pyplot as plt
a=[1,1,3]
bin_min=0
bin_max=np.max(a)
print(bin_max)
bins=np.array(range(bin_min-1,bin_max+2))
print(bins)
x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
print(x_bins)

count=plt.hist(a, bins=x_bins)
plt.show()
print(count)
sys.exit()
"""
class Param:
  def __init__(self,  arch_dir="architectures_saved/",print_dir='',plot_dir='plots/', 
      pfn_model='PFN', ae_model='PFN', bkg_events=500000, sig_events=500000, 
      num_elements=100, element_size=7, encoding_dim=32, latent_dim=4, phi_dim=64, nepochs=100, n_neuron=75, learning_rate=0.001, nlayer=3, 
      batchsize_pfn=512,batchsize_ae=32, # batchsize_pfn=500 -> 512 or any power of 2
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim3.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=[],seed=0):
      #sig_file="user.ebusch.SIGskim.mc20e.root", bkg_file="user.ebusch.QCDskim.mc20e.root",  bool_weight=True, extraVars=[]):
     
    self.time=time.strftime("%m_%d_%y_%H_%M", time.localtime())
    self.time_dir=time.strftime("%m_%d/", time.localtime())
#    self.all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/stats/'+self.time+'/' # for statistics
#    self.all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/test/'+self.time+'/' # for statistics
    self.all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/'+self.time+'/'

    self.arch_dir=self.all_dir+arch_dir
    self.print_dir=self.all_dir+print_dir
    self.plot_dir=self.all_dir+plot_dir

    dir_ls =[self.all_dir, self.arch_dir, self.print_dir, self.plot_dir] 
    for d in dir_ls:
      if not os.path.exists(d):
        os.mkdir(d)
        print(f'made a directory: {self.all_dir}')

    self.pfn_model=pfn_model
    self.ae_model=ae_model
    self.bkg_events=bkg_events
    self.sig_events=sig_events

    self.num_elements=num_elements 
    self.element_size=element_size
    self.encoding_dim=encoding_dim
    self.latent_dim=latent_dim
    self.phi_dim=phi_dim
    self.nepochs=nepochs
    self.n_neuron=n_neuron
    self.learning_rate=learning_rate
    self.nlayer=nlayer

    self.batchsize_pfn=batchsize_pfn
    self.batchsize_ae=batchsize_ae

    self.sig_file=sig_file
    self.bkg_file=bkg_file

    self.bool_weight=bool_weight
    self.extraVars=extraVars
    self.seed=seed

    if self.bool_weight:self.weight_tag='ws'
    else:self.weight_tag='nws'
    self.tag= f'{self.pfn_model}_2jAvg_MM_{self.weight_tag}'

    self.auc=0


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

  def open_print(self):
    print('printing in\n', self.print_dir)
    stdoutOrigin=sys.stdout
    sys.stdout = open(self.print_dir+f'stdout.txt', 'w')
    return stdoutOrigin

  def close_print(self, stdoutOrigin):
    sys.stdout.close()
    sys.stdout =stdoutOrigin

  def scan(self, var):
    
    return self.var
   
  def train(self):
    
    track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e","jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
    track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e","jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
    jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"] # order is important in apply_JetScalingRotation
    #jet_array = ["jet1_eta", "jet2_eta", "jet1_phi", "jet2_phi"] # order is important in apply_JetScalingRotation

   ## Load leading two jets
    # Plot inputs before the jet rotation
#    bkg, sig, mT_bkg, mT_sig = getTwoJetSystem(self.x_events,self.y_events,tag_file=self.tag+"_NSNR", tag_title=self.weight_tag+"_NSNR", bool_weight=self.bool_weight, sig_file=self.sig_file,bkg_file=self.bkg_file, extraVars=self.extraVars, plot_dir=self.plot_dir)

    bool_weight_sig=False

    
    sig, mT_sig, sig_sel, jet_sig = getTwoJetSystem(nevents=self.sig_events,input_file=self.sig_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight_sig,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed)
    bkg, mT_bkg, bkg_sel, jet_bkg = getTwoJetSystem(nevents=self.bkg_events,input_file=self.bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=self.bool_weight,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed)

    print(jet_bkg)
#    plot_vectors_jet(jet_bkg,jet_sig,jet_array, tag_file=self.tag+"_NSNR", tag_title=self.weight_tag+"_NSNR", plot_dir=self.plot_dir)
    plot_vectors(bkg_sel,sig_sel,tag_file=self.tag+"_NSNR", tag_title=self.weight_tag+"_NSNR", plot_dir=self.plot_dir)

    # Plot inputs after the jet rotation
    plot_vectors(bkg,sig,tag_file=self.tag+"_NSYR", tag_title=self.weight_tag+"_NSYR", plot_dir=self.plot_dir)

    bkg_events_num=bkg.shape[0]
    sig_events_num=sig.shape[0]
    print(bkg.shape,bkg_events_num, sig.shape,sig_events_num)
    # Create truth target
    input_data = np.concatenate((bkg,sig),axis=0)

    truth_bkg = np.zeros(bkg.shape[0])
    truth_sig = np.ones(sig.shape[0])

    truth_1D = np.concatenate((truth_bkg,truth_sig))
    truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)

    ## MAKE 1D DATA - DNN only
    #input_data = input_data.reshape(input_data.shape[0],40*4)

    print("Training shape, truth shape")
    print(input_data.shape, truth.shape)

    # Load the model
    pfn,graph_orig = get_full_PFN([self.num_elements,self.element_size], self.phi_dim, self.n_neuron, self.learning_rate, self.nlayer)
    #pfn = get_dnn(160)

   # Split the data 
    x_train, x_test, y_train, y_test = train_test_split(input_data, truth, test_size=0.02, random_state=42)

    # Fit scaler to training data, apply to testing data
    x_train, scaler = apply_StandardScaling(x_train)
    dump(scaler, self.arch_dir+self.pfn_model+'_scaler.bin', compress=True) #save the scaler
    x_test,_ = apply_StandardScaling(x_test,scaler,False)

    # Check the scaling & test/train split
    bkg_train_scaled = x_train[y_train[:,0] == 1]
    sig_train_scaled = x_train[y_train[:,0] == 0]
    bkg_test_scaled = x_test[y_test[:,0] == 1]
    sig_test_scaled = x_test[y_test[:,0] == 0]

   # Plot inputs before the jet rotation 
    print('bkg_train_scaled, bkg_test_scaled',bkg_train_scaled.shape, bkg_test_scaled.shape)
    bkg_scaled=np.concatenate((bkg_train_scaled, bkg_test_scaled), axis=0)
    sig_scaled=np.concatenate((sig_train_scaled, sig_test_scaled), axis=0)
    plot_vectors(bkg_scaled,sig_scaled,tag_file=self.tag+"_YSYR", tag_title=self.weight_tag+"_YSYR", plot_dir=self.plot_dir)

    plot_vectors(bkg_train_scaled,sig_train_scaled,tag_file=self.tag+"_train", tag_title=self.weight_tag+"_train", plot_dir=self.plot_dir)
    plot_vectors(bkg_test_scaled,sig_test_scaled,tag_file=self.tag+"_test", tag_title=self.weight_tag+"_test",  plot_dir=self.plot_dir)


    # Train
    h = pfn.fit(x_train, y_train,
      epochs=self.nepochs,
      batch_size=self.batchsize_pfn,
      validation_split=0.2,
      verbose=1)

    # Save the model
    pfn.get_layer('graph').save_weights(self.arch_dir+self.pfn_model+'_graph_weights.h5')
    pfn.get_layer('classifier').save_weights(self.arch_dir+self.pfn_model+'_classifier_weights.h5')
    pfn.get_layer('graph').save(self.arch_dir+self.pfn_model+'_graph_self.arch')
    pfn.get_layer('classifier').save(self.arch_dir+self.pfn_model+'_classifier_self.arch')

    ## PFN training plots
    # 1. Loss vs. epoch 
    plot_loss(h, loss= 'loss', tag_file=self.tag, tag_title=self.tag,  plot_dir=self.plot_dir) # tag_title=self.tag instead of self.weight_tag b/c specific to the model 
    # 2. Score 
    preds = pfn.predict(x_test)
    bkg_score = preds[:,1][y_test[:,1] == 0]
    sig_score = preds[:,1][y_test[:,1] == 1]
    plot_score(bkg_score, sig_score, False, False,tag_file=self.tag, tag_title=self.tag,  plot_dir=self.plot_dir)
    n_test = min(len(sig_score),len(bkg_score))
    bkg_score = bkg_score[:n_test]
    sig_score = sig_score[:n_test]
    auc=do_roc(bkg_score, sig_score, tag_file=self.tag, tag_title=self.tag, make_transformed_plot=False,  plot_dir=self.plot_dir)
    print(self.all_dir)
    return self.all_dir, auc, bkg_events_num,sig_events_num

"""
sig_events=915000
bkg_events=665000
sig_events=5000
bkg_events=5000
"""
"""
for latent_dim in [4,2,8]:
#for latent_dim in [2,8,4]:
  param1=Param( latent_dim=latent_dim, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info()) 
"""
"""
for latent_dim in [2,8,4]:
  n_events=500000
  param1=Param( latent_dim=latent_dim, bkg_events=n_events, sig_events=n_events)
  print(param1.save_info()) 
  print(param1.train())
  
for n_neuron in [75,40, 150]:
  param1=Param( n_neuron=n_neuron, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info()) 

# tested n_neuron: 07_03_23_11_05 07_03_23_11_35 07_03_23_11_5for phi_dim in [32,64, 128]:
#for phi_dim in [32, 128]:
for phi_dim in [ 128]:
  param1=Param( phi_dim=phi_dim, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
for batchsize_pfn in [250, 1000]:
  param1=Param( batchsize_pfn=batchsize_pfn, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
for nepochs in [50, 200]:
  param1=Param( nepochs=nepochs, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
"""
# half the statistics
sig_events=1151555
bkg_events=3234186

# read all the files in the directory
# load all the dictionaries
# find seed and auc pair
# get rid of duplicates of seed
# 
# make a histogram
# write into hdf5 for an average with the format compatible with the grid_scan 
# grid_scan 
import json
def read_auc(filedir):
  dict_ls={}
  
  onlydirs = [f for f in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, f))]
  for subdir in onlydirs:
    filepath=filedir+'/'+subdir+'/info.csv'
    print(f'{filepath=}')
    df_each=pd.read_csv(filepath)
    print(f'{df_each=}')
    sys.exit()
      
  
    print("Data type after reconstruction : ", type(dict_ls[filename]))
  print(dict_ls.keys())
  return dict_ls

#read_auc('/nevis/katya01/data/users/kpark/svj-vae/results/test')
#sig_events=900
#sig_events=90000
#bkg_events=600
#bkg_events=66000
#sig_events=915000
#bkg_events=665000
#seeds=np.random.randint(0,300,100)
#seeds=np.arange(1,100, dtype=int)
seeds=np.arange(0,100, dtype=int)

for seed in seeds:
  param1=Param(  bkg_events=bkg_events, sig_events=sig_events,seed=seed)
#  sys.exit()
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

  sys.exit()

 
for learning_rate in [0.0005,0.002]:
  param1=Param( learning_rate=learning_rate, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
"""
for nlayer in [6, 12]:
  param1=Param( nlayer=nlayer, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir, auc,bkg_events_num,sig_events_num=param1.train()
  setattr(param1, 'auc',auc )
  setattr(param1, 'sig_events_num',sig_events_num )
  setattr(param1, 'bkg_events_num',bkg_events_num )
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
"""
#original
#element_size = 4 # change here

sys.exit()

