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
#plot_dir='/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun29/' 
# Example usage
#added
class Param:
  def __init__(self,  arch_dir="architectures_saved/",print_dir='',plot_dir='plots/', 
      pfn_model='PFN', ae_model='PFN', x_events=500000, y_events=500000, 
      num_elements=100, element_size=7, encoding_dim=32, latent_dim=4, phi_dim=64, nepochs=100, nlayer=75, learning_rate=0.001,  
      batchsize_pfn=500,batchsize_ae=32,
      sig_file="user.ebusch.SIGskim.mc20e.root", bkg_file="user.ebusch.QCDskim.mc20e.root",  bool_weight=True, extraVars=[]):
     
    self.time=time.strftime("%m_%d_%y_%H_%M", time.localtime())
    self.time_dir=time.strftime("%m_%d/", time.localtime())
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
    self.x_events=x_events
    self.y_events=y_events

    self.num_elements=num_elements 
    self.element_size=element_size
    self.encoding_dim=encoding_dim
    self.latent_dim=latent_dim
    self.phi_dim=phi_dim
    self.nepochs=nepochs
    self.nlayer=nlayer
    self.learning_rate=learning_rate

    self.batchsize_pfn=batchsize_pfn
    self.batchsize_ae=batchsize_ae

    self.sig_file=sig_file
    self.bkg_file=bkg_file

    self.bool_weight=bool_weight
    self.extraVars=extraVars

    if self.bool_weight:self.weight_tag='ws'
    else:self.weight_tag='nws'
    self.tag= f'{self.pfn_model}_2jAvg_MM_{self.weight_tag}'

  def save_info(self):
    text=f'{vars(param1)}' # print all attributes of the class as dictionary
    print(text)
    print('printing in', self.print_dir)
    with open(self.print_dir+f'info.txt', 'w') as f: 
      f.write(text)
     
    return f'saved info in {self.print_dir}\n {text}'
   
  def train(self):
    
    ## Load leading two jets
    # Plot inputs before the jet rotation
    bkg, sig, mT_bkg, mT_sig = getTwoJetSystem(self.x_events,self.y_events,tag_file=self.tag+"_NSNR", tag_title=self.weight_tag+"_NSNR", bool_weight=self.bool_weight, sig_file=self.sig_file,bkg_file=self.bkg_file, extraVars=self.extraVars, plot_dir=self.plot_dir)
    #mT_bkg, mT_sig not used until after prediction e.g. when applying the scores

    # Plot inputs after the jet rotation
    plot_vectors(bkg,sig,tag_file=self.tag+"_NSYR", tag_title=self.weight_tag+"_NSYR", plot_dir=self.plot_dir)

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
    pfn,graph_orig = get_full_PFN([self.num_elements,self.element_size], self.phi_dim, self.nlayer, self.learning_rate)
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
    do_roc(bkg_score, sig_score, tag_file=self.tag, tag_title=self.tag, make_transformed_plot=False,  plot_dir=self.plot_dir)
     
    return self.all_dir

for latent_dim in [4]:
  n_events=1000000
  param1=Param( latent_dim=latent_dim, x_events=n_events, y_events=n_events)
  print(param1.x_events, param1.phi_dim)  
  print(param1.save_info()) 
  print(param1.train())

  

#original
#element_size = 4 # change here

sys.exit()

