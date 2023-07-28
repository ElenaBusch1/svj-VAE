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
#from numba import jit
import sys
import time
import pandas as pd
import shutil
import bs4
from sklearn.model_selection import StratifiedKFold
#plot_dir='/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun29/' 
# Example usage
#added
class Param:
  def __init__(self,  arch_dir="architectures_saved/",print_dir='/',plot_dir='plots/',h5_dir='h5dir/jul18/', 
      pfn_model='PFN', ae_model='PFN', bkg_events=500000, sig_events=500000, 
      num_elements=100, element_size=7, encoding_dim=32, phi_dim=64, nepochs=100, n_neuron=75, learning_rate=0.001,
      nlayer_phi=3, nlayer_F=3,
      max_track=80, 
      batchsize_pfn=512,
      batchsize_ae=32, # batchsize_pfn=500 -> 512 or any power of 2
      bool_pt=False,
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim3.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=[],seed=0,
      nfolds=3):
      #sig_file="user.ebusch.SIGskim.mc20e.root", bkg_file="user.ebusch.QCDskim.mc20e.root",  bool_weight=True, extraVars=[]):
     
    self.time=time.strftime("%m_%d_%y_%H_%M", time.localtime())
    self.time_dir=time.strftime("%m_%d/", time.localtime())
    self.parent_dir='/nevis/katya01/data/users/kpark/svj-vae/'
    self.all_dir=self.parent_dir+'results/test/'+self.time+'/' # for statistics

    self.arch_dir=self.all_dir+arch_dir
    self.print_dir=self.all_dir+print_dir
    self.plot_dir=self.all_dir+plot_dir
    self.h5_dir=self.parent_dir+h5_dir
   
    dir_ls =[self.all_dir, self.arch_dir, self.print_dir, self.plot_dir] 
    for d in dir_ls:
      if not os.path.exists(d):
        os.mkdir(d)
        print(f'made a directory: {self.all_dir}')
 
    
    self.nfolds=nfolds
    self.pfn_model=pfn_model
    self.ae_model=ae_model
    self.bkg_events=bkg_events
    self.sig_events=sig_events

    self.num_elements=num_elements 
    self.element_size=element_size
    self.encoding_dim=encoding_dim
    self.phi_dim=phi_dim
    self.nepochs=nepochs
    self.n_neuron=n_neuron
    self.learning_rate=learning_rate

    self.nlayer_phi=nlayer_phi
    self.nlayer_F=nlayer_F
    self.max_track=max_track

    self.batchsize_pfn=batchsize_pfn
    self.batchsize_ae=batchsize_ae
    self.bool_pt=bool_pt

    self.sig_file=sig_file
    self.bkg_file=bkg_file

    self.bool_weight=bool_weight
    self.extraVars=extraVars
    self.seed=seed

    if self.bool_weight:self.weight_tag='ws'
    else:self.weight_tag='nws'
    self.tag= f'{self.pfn_model}_2jAvg_MM_{self.weight_tag}'

#    self.auc=0
    

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
    # redirect the print statement
    sys.stdout.close()
    sys.stdout =stdoutOrigin

  def make_html(self, custom_plot_dir='', custom_www_dir='', html_title='/plots'):
    # if using this function: change www_dir to your www_dir
    www_dir='/nevis/kolya/home/kpark/WWW/'+ self.plot_dir.split(self.parent_dir)[-1]+'/'
    print(self.plot_dir.split(self.parent_dir))
    html_file= www_dir + f'{html_title}.html'

    # copy the directory containing plots -> only copies if the ww_dir doesn't already exists
    shutil.copytree(self.plot_dir, www_dir)
 
    # copy the template to the www dir
    temp_path='/nevis/kolya/home/kpark/WWW/template.html'    
    shutil.copyfile(temp_path, html_file)
    # change the template depending  on the # of plots dir
    with open(temp_path) as f:
      txt= f.read()
      soup=bs4.BeautifulSoup(txt, features="html.parser")

    # list all plots
    for i,filename in enumerate(os.listdir(www_dir)):
      plot_path=filename
      if 'html' in  filename:
        continue      
      
      new_td=soup.new_tag('td')
      new_img=soup.new_tag('img', src=plot_path)
      if i%4==0: 
        new_row=soup.new_tag('tr')
        new_row.append(new_td)

      new_td.append(new_img)
      if i%4==0:
         soup.table.append(new_row)
      else: 
         soup.table.append(new_td)
    
    with open(html_file, 'w') as f:
      f.write(str(soup.prettify()))
    
    cprint(f'{html_file=}', 'green')
    return 

  def example_skf(self, a=[]):
    if a==[]:
      a=np.array([[[1,2], [3,4], [5,6]], [[7,8], [9,10], [11,12]],[[13,14], [15,16], [17,18]]]) 
    b=np.ones(np.shape(a)[0])
    nfold=3
    cprint(f'before dividing into {nfold}-fold','yellow') 
    print(f'{a=}, {a.shape=}') 
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    for index, (index_train, index_test) in enumerate(skf.split(a, b)):
      cprint(f'{index}th of {nfold}-fold', 'yellow')
      print(f'{a[index_train].shape=}, {a[index_train]=}')
      print(f'{a[index_test].shape=}, {a[index_test]=}')
    return
 
  def prepare(self, dsid=0):
       
    all_start = time.time()
    track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e","jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
    track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e","jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
    jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"] # order is important in apply_JetScalingRotation
    #jet_array = ["jet1_eta", "jet2_eta", "jet1_phi", "jet2_phi"] # order is important in apply_JetScalingRotation

   ## Load leading two jets
    # Plot inputs before the jet rotation
#    bkg, sig, mT_bkg, mT_sig = getTwoJetSystem(self.x_events,self.y_events,tag_file=self.tag+"_NSNR", tag_title=self.weight_tag+"_NSNR", bool_weight=self.bool_weight, sig_file=self.sig_file,bkg_file=self.bkg_file, extraVars=self.extraVars, plot_dir=self.plot_dir)

    bool_weight_sig=False # important that this is False for sig 

    start = time.time()

    sig, mT_sig, sig_sel, jet_sig, sig_in0, sig_in1 = getTwoJetSystem(nevents=self.sig_events,input_file=self.sig_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight_sig,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed,max_track=self.max_track, bool_pt=self.bool_pt,h5_dir=self.h5_dir)
    bkg, mT_bkg, bkg_sel, jet_bkg,bkg_in0, bkg_in1 = getTwoJetSystem(nevents=self.bkg_events,input_file=self.bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=self.bool_weight,  extraVars=self.extraVars, plot_dir=self.plot_dir, seed=self.seed, max_track=self.max_track,bool_pt=self.bool_pt,h5_dir=self.h5_dir)
    
    end = time.time()
    print("Elapsed (with getTwoJetSystem) = %s  seconds" % (end - start))
    # FIX THESE
    plot_ntrack([ np.concatenate((sig_in0, sig_in1),axis=1), np.concatenate((bkg_in0,bkg_in1), axis=1),sig, bkg],  tag_file='_jet12', tag_title=' leading & subleading jet', plot_dir=self.plot_dir, bin_max=self.max_track*2)
    plot_ntrack([ sig_in0,  bkg_in0, sig[:,:80,:], bkg[:,:80,:]],  tag_file='_jet1', tag_title='leading jet', plot_dir=self.plot_dir, bin_max=self.max_track)
    plot_ntrack([ sig_in1,  bkg_in1, sig[:,80:,:], bkg[:,80:,:]],  tag_file='_jet2', tag_title='subleading jet', plot_dir=self.plot_dir, bin_max=self.max_track) # 80 not 81

    """
    plot_ntrack([sig_in0, bkg_in0],  tag_file='_jet1', tag_title=' leading jet', plot_dir=self.plot_dir, bin_max=self.max_track)
    plot_ntrack([sig_in1, bkg_in1],  tag_file='_jet2', tag_title=' subleading jet', plot_dir=self.plot_dir, bin_max=self.max_track)
    plot_ntrack([sig_in0, bkg_in0],  tag_file='_jet1_exp', tag_title=' leading jet', plot_dir=self.plot_dir)
    plot_ntrack([sig_in1, bkg_in1],  tag_file='_jet2_exp', tag_title=' subleading jet', plot_dir=self.plot_dir)
    sys.exit()
    """
    return sig, bkg,all_start
    
  def train(self, sig,bkg, all_start,dsid=0):
    y_sig=np.ones(np.shape(sig)[0])
    y_bkg=np.zeros(np.shape(bkg)[0])
    skf = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=42)

    ls_dict={}
    auc_dict={}
    sicMax_dict={}
    sigEff_dict={}
    qcdEff_dict={}
    score_cut_dict={}
    sig_events_num_dict={}
    bkg_events_num_dict={}
    for i, (index_train, index_test) in enumerate(skf.split(sig, y_sig)):
      ls_dict[str(i)]={}
      cprint(i, 'yellow')
      print(sig[index_train].shape)
      print(sig[index_test].shape)
      ls_dict[str(i)]['index_train_sig']=index_train
      ls_dict[str(i)]['index_test_sig']=index_test
      
    for i, (index_train, index_test) in enumerate(skf.split(bkg, y_bkg)):
      cprint(i, 'yellow')
      print(bkg[index_train].shape)
      print(bkg[index_test].shape)
      ls_dict[str(i)]['index_train_bkg']=index_train
      ls_dict[str(i)]['index_test_bkg']=index_test
#skf for kfold
    for key,value in ls_dict.items():
      nfold=key
      bkg_test= bkg[value['index_test_bkg']]
      sig_test= sig[value['index_test_sig']]
      bkg_train= bkg[value['index_train_bkg']]
      sig_train= sig[value['index_train_sig']]
      y_sig_train=y_sig[value['index_train_sig']]
      y_bkg_train=y_bkg[value['index_train_bkg']]
      y_sig_test=y_sig[value['index_test_sig']]
      y_bkg_test=y_bkg[value['index_test_bkg']]

      x_train=np.concatenate((sig_train, bkg_train), axis=0)
      x_test=np.concatenate((sig_test, bkg_test), axis=0)
      y_train=tf.keras.utils.to_categorical(np.concatenate((y_sig_train, y_bkg_train), axis=0), num_classes=2)
      y_test=tf.keras.utils.to_categorical(np.concatenate((y_sig_test, y_bkg_test), axis=0), num_classes=2)
      """
      # check if categorizing is correct: we would expect all the signals to have look like [[1,0], [1,0]...]] and bkg to look like [[0,1], [0,1],...]]
      check_y_sig=tf.keras.utils.to_categorical(y_sig_train, num_classes=2)
      check_y_bkg=tf.keras.utils.to_categorical(y_bkg_train, num_classes=2)
      cprint(f'{check_y_sig.shape}, {check_y_sig}', 'red')
      cprint(f'{check_y_bkg.shape}, {check_y_bkg}', 'red')
      """
 
      bkg_skf=np.concatenate((bkg_test, bkg_train), axis=0)
      sig_skf=np.concatenate((sig_test, sig_train), axis=0)

      bkg_events_num=bkg.shape[0]
      sig_events_num=sig.shape[0]

      path=f'{self.all_dir}/{nfold}/'
      arch_dir=path+self.arch_dir.split('/')[-2]+'/'
      plot_dir=path+self.plot_dir.split('/')[-2]+'/'
      dir_ls=[arch_dir,  plot_dir] # don't include print_dir here
      if not os.path.exists(path):
        os.mkdir(path)
        for d in dir_ls:
          if not os.path.exists(d):
            os.mkdir(d)
            print(f'made a directory: {path}')
  
      setattr(self, 'arch_dir',arch_dir)
      setattr(self, 'plot_dir',plot_dir)
      #plot_vectors(bkg_sel,sig_sel,tag_file=self.tag+"_NSNR", tag_title=self.weight_tag+"_NSNR", plot_dir=self.plot_dir) # wrong but keep for reference
      
      plot_vectors(bkg_skf,sig_skf,tag_file=self.tag+f"_NSNR_{nfold}", tag_title=self.weight_tag+f"_NSNR ({nfold}th)", plot_dir=self.plot_dir)
  
      # Plot inputs after the jet rotation
      plot_vectors(bkg_skf,sig_skf,tag_file=self.tag+f"_NSYR_{nfold}", tag_title=self.weight_tag+f"_NSYR ({nfold}th)", plot_dir=self.plot_dir)
  
      # Create truth target
   #   input_data = np.concatenate((bkg,sig),axis=0)
      """
      truth_bkg = np.zeros(bkg.shape[0])
      truth_sig = np.ones(sig.shape[0])
  
      truth_1D = np.concatenate((truth_bkg,truth_sig))
      truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)
  
      """
         #   print("Training shape, truth shape")
  #    print(input_data.shape, truth.shape)
  
      # Load the model
      pfn,graph_orig = get_full_PFN([self.num_elements,self.element_size], self.phi_dim, self.n_neuron, self.learning_rate, self.nlayer_phi, self.nlayer_F)
      #pfn = get_dnn(160)
  
     # Split the data 
  #    x_train, x_test, y_train, y_test = train_test_split(input_data, truth, test_size=0.02, random_state=42)
  
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
      plot_vectors(bkg_scaled,sig_scaled,tag_file=self.tag+f"_YSYR_{nfold}", tag_title=self.weight_tag+f"_YSYR ({nfold}th)", plot_dir=self.plot_dir)
  
      plot_vectors(bkg_train_scaled,sig_train_scaled,tag_file=self.tag+f"_train_{nfold}", tag_title=self.weight_tag+f"_train ({nfold}th)", plot_dir=self.plot_dir)
      plot_vectors(bkg_test_scaled,sig_test_scaled,tag_file=self.tag+f"_test_{nfold}", tag_title=self.weight_tag+f"_test ({nfold}th)",  plot_dir=self.plot_dir)
  
  
      start = time.time()
      # Train
      cprint(f'{x_train.shape}, {y_train.shape}', 'red')
      h = pfn.fit(x_train, y_train,
        epochs=self.nepochs,
        batch_size=self.batchsize_pfn,
        validation_split=0.2,
        verbose=1)
  
      end = time.time()
      print("Elapsed (with fitting the model) = %s seconds" % (end - start))
      # Save the model
      pfn.get_layer('graph').save_weights(self.arch_dir+self.pfn_model+'_graph_weights.h5')
      pfn.get_layer('classifier').save_weights(self.arch_dir+self.pfn_model+'_classifier_weights.h5')
      pfn.get_layer('graph').save(self.arch_dir+self.pfn_model+'_graph_arch')
      pfn.get_layer('classifier').save(self.arch_dir+self.pfn_model+'_classifier_arch')
  
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
      sicMax, sigEff, qcdEff, score_cut, auc=do_roc(bkg_score, sig_score, tag_file=self.tag, tag_title=self.tag +f' ({nfold}th)', make_transformed_plot=False,  plot_dir=self.plot_dir).values()

      auc_dict[str(nfold)]=auc
      sicMax_dict[str(nfold)]=sicMax
      sigEff_dict[str(nfold)]=sigEff
      qcdEff_dict[str(nfold)]=qcdEff
      score_cut_dict[str(nfold)]=score_cut
      sig_events_num_dict[str(nfold)]=sig_events_num
      bkg_events_num_dict[str(nfold)]=bkg_events_num
      self.make_html()

      #write hdft of the indices

    setattr(self, 'auc',auc_dict )
    setattr(self, 'sicMax',sicMax_dict )
    setattr(self, 'sigEff',sigEff_dict )
    setattr(self, 'qcdEff',qcdEff_dict )
    setattr(self, 'score_cut',score_cut_dict )
    setattr(self, 'sig_events_num',sig_events_num_dict )
    setattr(self, 'bkg_events_num',bkg_events_num_dict )
  
    all_end = time.time()
    print("Elapsed (in total) = %s seconds" % (all_end - all_start))
    print(self.all_dir)
    return self.all_dir



"""
"""
import json
def read_info(filedir):
  df=pd.DataFrame()
  onlydirs = [f for f in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, f))] # all the directories in the parent directory
  for subdir in onlydirs:
    filepath=filedir+'/'+subdir+'/info.csv'
    print(f'{filepath=}')
    df_row=pd.read_csv(filepath)
    print(subdir, df_row)
    df=pd.concat([df, df_row]).reset_index(drop=True)
    if print(df.duplicated().any()):
      cprint(f'there is a duplicate so exiting {df.duplicated()}' , 'red')
      sys.exit() # check if there are any duplicates is noted as True
  print(df.columns)
  return df # {directory#1: df in the csv , ... directory#n: df in the corresponding csv}


def plot_info(df, param, plot_dir):
  df=df[param] # find one hyperparameter that's different

  if param =='auc':
    a=[(json.loads(element.replace("'", '"')))[param] for element in df.tolist()] # element is dictionary that looks like ['sicMax': 3.68, ... 'auc':...] # need to replace single quote with double quote for json to work
  else:a=df
  print(a)
  
  mean, std=np.mean(a), np.std(a)
  plt.hist(a,bins=10, alpha=.8, histtype='step', label=f'{len(df.index)} entries')
  plt.plot([], [], label=f'$\mu$={round(mean,4)}')
  plt.plot([], [], label=f'$\sigma$={round(std,4)}')
  plt.plot([], [], label=f'$min$={round(np.min(a),4)}, $max$={round(np.max(a),4)}')
  plt.title(f'Histogram of {param.upper()}')
  plt.xlabel(f'{param.upper()}')
  plt.ylabel('count')
  plt.legend(loc='upper left')
  plt.show()
#  plt.savefig(plot_dir+'/'+param+'_newbin.png')
  plt.savefig(plot_dir+'/'+param+'.png')
  #plt.show()
  plt.clf()     

"""
#make variance scan plot
dir_read='/nevis/katya01/data/users/kpark/svj-vae/results/stats'
df_info=read_info(dir_read)
plot_info(df_info, param='auc', plot_dir=dir_read)
sys.exit()

seeds=np.arange(0,100, dtype=int)
#seed=seeds[0]
sig_events=1151555
bkg_events=3234186
"""
#sig_events=502000 # change after no pt requirement
#bkg_events=502000
sig_events=5000
bkg_events=5000
#max_track=80 #160
max_track=15 #160
"""
for nlayer in [2,3,4]:
#for nlayer in [3]:
  param1=Param(  bkg_events=bkg_events, sig_events=sig_events, nlayer_phi=nlayer, nlayer_F=nlayer)
  stdoutOrigin=param1.open_print()
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
for nevents in [1151555, 251000]:
#for nlayer in [3]:
  param1=Param(  bkg_events=nevents, sig_events=nevents)
  stdoutOrigin=param1.open_print()
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
for max_t in [60, 100]:
  param1=Param(  bkg_events=bkg_events, sig_events=sig_events, max_track=max_t)
#  sys.exit()
  stdoutOrigin=param1.open_print()
  print(param1.save_info())
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
  sys.exit()
"""
  
for n_neuron in [40, 150]:
#for n_neuron in [75,40, 150]:
  param1=Param( bkg_events=bkg_events, sig_events=sig_events, n_neuron=n_neuron)
#  stdoutOrigin=param1.open_print()
  sig,bkg,all_start=param1.prepare()
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
#  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info()) 
  sys.exit()
for phi_dim in [32, 128]:
  param1=Param( bkg_events=bkg_events, sig_events=sig_events, phi_dim=phi_dim)
  stdoutOrigin=param1.open_print()
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

for learning_rate in [0.0005,0.002]:
  param1=Param(bkg_events=bkg_events, sig_events=sig_events,  learning_rate=learning_rate)
  stdoutOrigin=param1.open_print()
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

for nepochs in [50, 200]:
  param1=Param( nepochs=nepochs, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

for batchsize_pfn in [256, 1024]:
  param1=Param( batchsize_pfn=batchsize_pfn, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  all_dir=param1.train(sig=sig, bkg=bkg, all_start=all_start)
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())


#original
#element_size = 4 # change here

sys.exit()

