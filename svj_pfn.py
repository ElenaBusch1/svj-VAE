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
from antelope_h5eval import *
#plot_dir='/nevis/katya01/data/users/kpark/svj-vae/plots_result/jun29/' 
# Example usage
#added
track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e","jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e","jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"] # order is important in apply_JetScalingRotation
class Param:
  def __init__(self,  arch_dir="architectures_saved/",print_dir='/',plot_dir='plots/',h5_dir='h5dir/jul28/', 
      pfn_model='PFN', ae_model='PFN', bkg_events=500000, sig_events=500000, 
      num_elements=100, element_size=7, encoding_dim=32, phi_dim=64, nepochs=100, n_neuron=75, learning_rate=0.001,
      nlayer_phi=3, nlayer_F=3,
      max_track=80, 
      batchsize_pfn=512,
      batchsize_ae=32, # batchsize_pfn=500 -> 512 or any power of 2
      bool_pt=False,
      sig_file="skim3.user.ebusch.SIGskim.root", bkg_file="skim3.user.ebusch.QCDskim.root",  bool_weight=True, extraVars=[],seed=0,
      nfolds=3, bool_select_all=True, folder=''):
      #sig_file="user.ebusch.SIGskim.mc20e.root", bkg_file="user.ebusch.QCDskim.mc20e.root",  bool_weight=True, extraVars=[]):
    if folder=='': # name of directory e.g. 07_24_23_07_11
      self.time=time.strftime("%m_%d_%y_%H_%M", time.localtime())
    else:self.time=folder
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
        print(f'made a directory: {d}')
 
    
    self.nfolds=nfolds
    self.pfn_model=pfn_model
    self.ae_model=ae_model
    self.bkg_events=bkg_events
    self.sig_events=sig_events

    self.num_elements=num_elements 
    self.element_size=element_size # this should equal the number of track variables used -> 7 track variables used -> element_size=7
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
    self.bool_select_all=bool_select_all
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

  def open_print(self): # redirects the printed statement on the terminal to be saved in a stdout.txt file
    print('printing in\n', self.print_dir)
    stdoutOrigin=sys.stdout
    sys.stdout = open(self.print_dir+f'stdout.txt', 'w')
    return stdoutOrigin

  def close_print(self, stdoutOrigin): # close the stdout.txt file and the statement printed after this function will be printed on the terminal
    # redirect the print statement
    sys.stdout.close()
    sys.stdout =stdoutOrigin

  def make_html(self, nfold,plot_dir='', www_dir='', html_title='/plots'): # uses the template.html and makes an html file; if there's an issue -> check href of mystyle.css, nevis_logo.jpg, and image files; '../' might need to be added/removed depending on the relative location of the html file 
    # if using this function: change www_dir to your www_dir
    www_dir='/nevis/kolya/home/kpark/WWW/'+ self.plot_dir.split(self.parent_dir)[-1]+'/'
    print(self.plot_dir.split(self.parent_dir), nfold)
    html_file= www_dir + f'{html_title}.html'
    # copy the directory containing plots -> only copies if the ww_dir doesn't already exists
    shutil.copytree(plot_dir, www_dir)
 
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
    y=np.ones(np.shape(a)[0])
    nfold=3
    cprint(f'before dividing into {nfold}-fold','yellow') 
    print(f'{a=}, {a.shape=}') 
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    arr_skf={}
    for index, (index_train, index_test) in enumerate(skf.split(a, y)):
      cprint(f'{index}th of {nfold}-fold', 'yellow')
      print(f'{a[index_train].shape=}, {a[index_train]=}')
      print(f'{a[index_test].shape=}, {a[index_test]=}')
   
      train=a[index_train]
      y_train=y[index_train]
      arr_dict={'index_train': index_train,'index_test': index_test}
      arr_skf[nfold]=arr_dict

    # check if the index_train of one set and index_test of the other contain the same -> if not apply the model
    """
    for nfold in arr_skf:
      for nfold_other in arr_skf:
        if not(set(arr_skf[nfold]['index_train'])&set(arr_skf[nfold_other]['index_test']) ):
          # no common element
          print(f'{nfold},{nfold_other}')
    """
     
    return 

  def kfold(self, arr,bool_sig, dsid, all_dir,mT=np.array([]), extraVars=np.array([])): # read/write from/to a hdf5
    bool_rewrite=False
    if len(extraVars)==0:
      str_ls=['index_train', 'index_test','arch_dir', 'plot_dir', 'train', 'test', 'y_train', 'y_test', 'idx_dir']
    else: 
      newVars=[f'{x}_train' for x in extraVars]
      newVars+=[f'{x}_test' for x in extraVars]
      str_ls=['index_train', 'index_test','arch_dir', 'plot_dir', 'train', 'test', 'y_train', 'y_test', 'idx_dir',*newVars]
    arr_dict={}
    arr_skf={} 
    for nfold in range(self.nfolds):
      idxpath=f'{self.h5_dir}idx/'
      idx_dir=idxpath+f'{nfold}/'
      idx_file=idx_dir+f'idx_{dsid}_{nfold}.hdf5'
      # EVEN IF there is a file existing, it might not already have the extraVars info available, if so manually make a new idx_file containing all extraVars info  
      path=f'{all_dir}{nfold}/'
      arch_dir=path+self.arch_dir.split('/')[-2]+'/'
      plot_dir=path+self.plot_dir.split('/')[-2]+'/'
      dir_ls =[all_dir, arch_dir, plot_dir, idxpath,idx_dir] 
      if not os.path.exists(path):
        os.mkdir(path)
        print(f'made a directory: {path}')
        for d in dir_ls:
          if not os.path.exists(d):
            os.mkdir(d)
            print(f'made a directory: {d}')
      if os.path.exists(idx_file):
        with h5py.File(idx_file, 'r') as f:
          for i in range(len(str_ls)):
            arr_dict[str_ls[i]] = f["default"][str_ls[i]][()]

        arr_skf[nfold]=arr_dict 
      else: 
        bool_rewrite=True
        print('the file does not exist so will write to', idx_file) 
        break

    if not(bool_rewrite):
      if 'extraVars' in arr_skf.keys():      print(arr_skf['extraVars'])
      print('successfully read from idx files that have skf infos e.g. test, train,...')
      return arr_skf

    arr_skf={} # refresh b/c have to rewrite as not all files are existent
    arr_dict={}
    if bool_sig:
      y=np.ones(np.shape(arr)[0])
    else: y=np.zeros(np.shape(arr)[0])
    skf = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=42)
    for nfold, (index_train, index_test) in enumerate(skf.split(arr, y)):
      idxpath=f'{self.h5_dir}idx/'
      idx_dir=idxpath+f'{nfold}/'
      idx_file=idx_dir+f'idx_{dsid}_{nfold}.hdf5'
      path=f'{all_dir}{nfold}/'
      arch_dir=path+self.arch_dir.split('/')[-2]+'/'
      plot_dir=path+self.plot_dir.split('/')[-2]+'/'
      dir_ls =[all_dir, arch_dir, plot_dir, idxpath,idx_dir] 
      if not os.path.exists(path):
        os.mkdir(path)
        print(f'made a directory: {path}')
        for d in dir_ls:
          if not os.path.exists(d):
            os.mkdir(d)
            print(f'made a directory: {d}')
      train=arr[index_train] 
      test=arr[index_test] 
      y_train=y[index_train] 
      y_test=y[index_test]
      if len(extraVars)!=0:
        mT_train=mT[index_train] 
        mT_test=mT[index_test]
        data_ls=[index_train, index_test, arch_dir, plot_dir, train, test, y_train, y_test, idx_dir]
        print(mT_test.shape) 
    # save idx hdf5 here with a name idx_{nfold}.hdf5
      with h5py.File(idx_file,"w") as f:
        dset = f.create_group('default')
        for i in range(len(data_ls)):  # not str_ls which also contains 'extraVars'
          dset.create_dataset(str_ls[i],data=data_ls[i])
        for i in range(len(extraVars)):
          dset.create_dataset(f'{extraVars[i]}_train',data=mT_train[:,i])
          dset.create_dataset(f'{extraVars[i]}_test',data=mT_test[:,i])
          #data= dset.create_dataset(str_ls[i],data=data_ls[i])
      
      if len(extraVars)==0:
        arr_dict={'train': train,'test':test, 'y_train': y_train, 'y_test': y_test, 'arch_dir': arch_dir, 'plot_dir': plot_dir, 'idx_dir': idx_dir}
      else:
        arr_dict={'train': train,'test':test, 'y_train': y_train, 'y_test': y_test,'mT_train': mT_train, 'mT_test': mT_test, 'extraVars': extraVars,'arch_dir': arch_dir, 'plot_dir': plot_dir, 'idx_dir': idx_dir}
      arr_skf[nfold]=arr_dict
     
    return arr_skf

  def prepare(self,extraVars=["mT_jj", "weight"]):
    all_start = time.time()
    setattr(self, 'all_start',all_start)

   ## Load leading two jets
    bool_weight_sig=False # important that this is False for sig 
    start = time.time()

    # this creates a hdf5 file of all the information (output of getTwoJetSystem function) titled {input_file}_s={seed}_ne={nevents}_mt={max_track}.hdf5'
    dsids=list(range(515487,515527))
    corrupt_files=[515508, 515511,515493]
    dsids=[x for x in dsids if x not in corrupt_files ]
    file_ls=[]
    sig_dict={}
    for dsid in dsids:
      file_ls.append("skim3.user.ebusch."+str(dsid)+".root")
    
#    file_ls=[self.sig_file]
    for fl in file_ls:
      dsid=fl.split('.')[-2]

      sig, mT_sig, sig_sel, jet_sig, sig_in0, sig_in1 = getTwoJetSystem(nevents=self.sig_events,input_file=fl,
        track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
        bool_weight=bool_weight_sig,  extraVars=extraVars, plot_dir=self.plot_dir, seed=self.seed,max_track=self.max_track, bool_pt=self.bool_pt,h5_dir=self.h5_dir, bool_select_all=self.bool_select_all)
     # input_file for sig is fl not self.sig_file
      sig_skf=self.kfold(arr=sig, bool_sig=True,dsid=dsid, all_dir=self.all_dir,mT=mT_sig, extraVars=extraVars)
      for nfold, val in sig_skf.items():
        if fl ==file_ls[0]:
          sig_dict[nfold]={}
          for key in ['y_train', 'y_test','train', 'test']: 
            sig_dict[nfold][key]=val[key]
        # for each trial, add the file 
        else: 
          for key in ['y_train', 'y_test','train', 'test']: 
            sig_dict[nfold][key]=np.concatenate((sig_dict[nfold][key],val[key]), axis=0)
   
    setattr(self, 'sig_file',file_ls)
    bkg, mT_bkg, bkg_sel, jet_bkg,bkg_in0, bkg_in1 = getTwoJetSystem(nevents=self.bkg_events,input_file=self.bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=self.bool_weight,  extraVars=extraVars, plot_dir=self.plot_dir, seed=self.seed, max_track=self.max_track,bool_pt=self.bool_pt,h5_dir=self.h5_dir, bool_select_all=False)
    
    bkg_dict=self.kfold(arr=bkg, bool_sig=False,dsid=self.bkg_file.split('.')[-2], all_dir=self.all_dir,mT=mT_bkg, extraVars=extraVars) # important that bool_sig=False for bkg

    end = time.time()
    print("Elapsed (with getTwoJetSystem) = %s  seconds" % (end - start))
    # FIX THESE
    for nfold in sig_dict:
      cprint(f"for {nfold}th, {sig_dict[nfold]['test'].shape=},{bkg_dict[nfold]['test'].shape=}",'yellow')
    plot_ntrack([ np.concatenate((sig_in0, sig_in1),axis=1), np.concatenate((bkg_in0,bkg_in1), axis=1),sig, bkg],  tag_file='_jet12', tag_title=' leading & subleading jet', plot_dir=self.plot_dir, bin_max=self.max_track*2)
#    plot_ntrack([ sig_in0,  bkg_in0, sig[:,:80,:], bkg[:,:80,:]],  tag_file='_jet1', tag_title='leading jet', plot_dir=self.plot_dir, bin_max=self.max_track)

    self.train(sig_dict=sig_dict, bkg_dict=bkg_dict)

    return sig, bkg
 
  def do_apply(self, all_dir,extraVars=["mT_jj", "weight"],bool_select_all=True):
    all_start = time.time()
    setattr(self, 'all_start',all_start)
    bool_weight_sig=False # important that this is False for sig 
    start = time.time()

    # this creates a hdf5 file of all the information (output of getTwoJetSystem function) titled {input_file}_s={seed}_ne={nevents}_mt={max_track}.hdf5'
    dsids=list(range(515487,515527))
    corrupt_files=[515508, 515511,515493]
    dsids=[x for x in dsids if x not in corrupt_files ]
    file_ls=[]
    for dsid in dsids:
      file_ls.append("skim3.user.ebusch."+str(dsid)+".root")
    #file_ls=["skim3.user.ebusch.SIGskim.root"]
    file_ls.append("skim3.user.ebusch.QCDskim.root")
    for fl in file_ls:
      cprint(f'{fl}', 'red')
      sig_dict={}
      dsid=fl.split('.')[-2]
      # use extraVars instead of self.extraVars 
      sig, mT_sig, sig_sel, jet_sig, sig_in0, sig_in1 = getTwoJetSystem(nevents=self.sig_events,input_file=fl,
        track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
        bool_weight=bool_weight_sig,  extraVars=extraVars, plot_dir=self.plot_dir, seed=self.seed,max_track=self.max_track, bool_pt=self.bool_pt,h5_dir=self.h5_dir, bool_select_all=bool_select_all)
     # input_file for sig is fl not self.sig_file
      print('e', extraVars)
      cprint(f'{mT_sig}', 'red')
#      all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/test/07_28_23_14_24/'
      #all_dir='/nevis/katya01/data/users/kpark/svj-vae/results/test/07_28_23_13_39/'
      if ('QCD' in fl) or ('Znunu' in fl):
        sig_skf=self.kfold(arr=sig, bool_sig=False,dsid=dsid, all_dir=self.all_dir,mT=mT_sig, extraVars=extraVars) # important that bool_sig=False for bkg
      else:sig_skf=self.kfold(arr=sig, bool_sig=True,dsid=dsid, all_dir=self.all_dir,mT=mT_sig, extraVars=extraVars)
      for nfold, val in sig_skf.items():
        arr_dict, newVars,h5path=self.apply(val, dsid, extraVars, all_dir)
        cprint(f'{nfold}among {self.nfolds}, {newVars}', 'yellow')
        if nfold == 0 or nfold==str(0):
          for var in newVars: # not extraVars
            sig_dict[var]=arr_dict[f'{var}_test']
        # for each trial, add the file 
        else: 
          for var in newVars:
            sig_dict[var]=np.concatenate((sig_dict[var], arr_dict[f'{var}_test']), axis=0)

      
      self.save(sig_dict, newVars, h5path)               
      

    end = time.time()
    print("Elapsed (with getTwoJetSystem and do_apply) = %s  seconds" % (end - start))
    # FIX THESE

    #self.train(sig_dict=sig_dict, bkg_dict=bkg_dict)

    return all_dir
   
  def apply(self, arr_dict, dsid, extraVars, all_dir):

    arch_dir=arr_dict['arch_dir']
    #  mT=arr_dict['mT_test']
    applydir=all_dir+f'/applydir/'
    if not(os.path.exists(applydir)):
      os.mkdir(applydir)
    h5path=applydir+"v8p1_"+str(dsid)+".hdf5" 
    pfn_model = 'PFN'
    cprint(arch_dir+pfn_model+'_graph_arch','yellow')
    graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
    graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
    graph.compile()

    ## Load classifier model
    classifier = keras.models.load_model(arch_dir+pfn_model+'_classifier_arch')
    classifier.load_weights(arch_dir+pfn_model+'_classifier_weights.h5')
    classifier.compile()

    test2=arr_dict['test']
    scaler = load(arch_dir+pfn_model+'_scaler.bin')
    test2,_ = apply_StandardScaling(test2,scaler,False)

    phi_test = graph.predict(test2)

    # each event has a pfn score 
    pred_phi_test = classifier.predict(phi_test)
    arr_dict['score_test'] = pred_phi_test[:,1]
    newVars=['score'] # not score_test
    newVars+=extraVars
    return arr_dict, newVars,h5path

  def save(self, arr_dict, newVars, h5path):
    save_test=arr_dict['score'][:,None]
    for var in [x for x in newVars if x!='score']: 
      cprint(save_test.shape,'green')
      print(np.array([arr_dict[var]]).T)
      
      save_test = np.concatenate((save_test, np.array([arr_dict[var]]).T),axis=1)
    #save_test = np.concatenate((test_loss[:,None], mT),axis=1)
    ds_dt = np.dtype({'names':newVars,'formats':[(float)]*len(newVars)})
    rec_bkg = np.rec.array(save_test, dtype=ds_dt)

    with h5py.File(h5path,"w") as f:
      dset = f.create_dataset("data",data=rec_bkg)
    print('saved ', h5path)
 
  def train(self, sig_dict,bkg_dict):
    auc_dict={}
    sicMax_dict={}
    sigEff_dict={}
    qcdEff_dict={}
    score_cut_dict={}
    sig_events_num_dict={}
    bkg_events_num_dict={}
     

#skf for kfold
    for nfold,value in bkg_dict.items():
      setattr(self, 'arch_dir',bkg_dict[nfold]['arch_dir'])
      setattr(self, 'plot_dir',bkg_dict[nfold]['plot_dir'])
      cprint(f'{nfold}{self.arch_dir},{self.plot_dir}', 'yellow')
#      cprint(f"{sig_dict[nfold]['train']},{bkg_dict[nfold]['train']}", 'yellow') 
#      cprint(f"{sig_dict[nfold]['y_train']},{bkg_dict[nfold]['y_train']}", 'yellow')
#      sys.exit() 
      x_train=np.concatenate((sig_dict[nfold]['train'], bkg_dict[nfold]['train']), axis=0)
      x_test=np.concatenate((sig_dict[nfold]['test'], bkg_dict[nfold]['test']), axis=0)
      y_train=tf.keras.utils.to_categorical(np.concatenate((sig_dict[nfold]['y_train'],bkg_dict[nfold]['y_train']), axis=0), num_classes=2)
      y_test=tf.keras.utils.to_categorical(np.concatenate((sig_dict[nfold]['y_test'],bkg_dict[nfold]['y_test']), axis=0), num_classes=2)
      """
      # check if categorizing is correct: we would expect all the signals to have look like [[1,0], [1,0]...]] and bkg to look like [[0,1], [0,1],...]]
      check_y_sig=tf.keras.utils.to_categorical(y_sig_train, num_classes=2)
      check_y_bkg=tf.keras.utils.to_categorical(y_bkg_train, num_classes=2)
      cprint(f'{check_y_sig.shape}, {check_y_sig}', 'red')
      cprint(f'{check_y_bkg.shape}, {check_y_bkg}', 'red')
      """
 
      bkg_skf=np.concatenate((bkg_dict[nfold]['train'], bkg_dict[nfold]['test']), axis=0)
      sig_skf=np.concatenate((sig_dict[nfold]['train'], sig_dict[nfold]['test']), axis=0)

      bkg_events_num=bkg_skf.shape[0]
      sig_events_num=sig_skf.shape[0]

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
      # choose the smaller of the two : size of signal and background for calculating the auc score
      n_test = min(len(sig_score),len(bkg_score))
      bkg_score = bkg_score[:n_test]
      sig_score = sig_score[:n_test]
      sicMax, sigEff, qcdEff, score_cut, auc=do_roc(bkg_score, sig_score, tag_file=self.tag, tag_title=self.tag +f' ({nfold}th)', make_transformed_plot=False,  plot_dir=self.plot_dir).values()

      auc_dict[nfold]=auc
      sicMax_dict[nfold]=sicMax
      sigEff_dict[nfold]=sigEff
      qcdEff_dict[nfold]=qcdEff
      score_cut_dict[nfold]=score_cut
      sig_events_num_dict[nfold]=sig_events_num
      bkg_events_num_dict[nfold]=bkg_events_num
      
      plot_dir=self.plot_dir
      plot_dir=plot_dir.replace('//','/')
      print(plot_dir)
      self.make_html(plot_dir=plot_dir,nfold=nfold)

      #write hdft of the indices
      # apply the model 
      # average the auc

    
    setattr(self, 'auc',auc_dict )
    setattr(self, 'sicMax',sicMax_dict )
    setattr(self, 'sigEff',sigEff_dict )
    setattr(self, 'qcdEff',qcdEff_dict )
    setattr(self, 'score_cut',score_cut_dict )
    setattr(self, 'sig_events_num',sig_events_num_dict )
    setattr(self, 'bkg_events_num',bkg_events_num_dict )
  
    all_end = time.time()
    print("Elapsed (in total) = %s seconds" % (all_end - self.all_start))
    print(self.all_dir)
    return self.all_dir


"""
Below are functions to do variance test
"""
import json
def read_info(filedir): # read information from an info.csv file and return a pandas dataframe 
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


def plot_info(df, param, plot_dir): # plot a variance of a variable e.g. auc score in a histogram
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
execute below to make variance scan plot
"""
"""
dir_read='/nevis/katya01/data/users/kpark/svj-vae/results/stats'
df_info=read_info(dir_read)
plot_info(df_info, param='auc', plot_dir=dir_read)
sys.exit()

"""
#seeds=np.arange(0,100, dtype=int)
#seed=seeds[0]

"""
Here are some parameters to change to make PFN models

"""

#sig_events=1151555
#bkg_events=3234186
#sig_events=502000 # change after no pt requirement
#bkg_events=502000
#sig_events=2000
sig_events=500
#sig_events=50000
#sig_events=100000000
#sig_events=1000000
#bkg_events=1151555
bkg_events=5000
#bkg_events=1151000
#max_track=80 #160
max_track=15 #160
"""
for nlayer in [2,3,4]:
#for nlayer in [3]:
  param1=Param(  bkg_events=bkg_events, sig_events=sig_events, nlayer_phi=nlayer, nlayer_F=nlayer)
  stdoutOrigin=param1.open_print()
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
for nevents in [1151555, 251000]:
#for nlayer in [3]:
  param1=Param(  bkg_events=nevents, sig_events=nevents)
  stdoutOrigin=param1.open_print()
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
for max_t in [60, 100]:
  param1=Param(  bkg_events=bkg_events, sig_events=sig_events, max_track=max_t)
#  sys.exit()
  stdoutOrigin=param1.open_print()
  print(param1.save_info())
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())
  sys.exit()
"""
# there are multiple hdf5files
"""
1)
* /nevis/katya01/data/users/kpark/svj-vae/h5dir/jul28/twojet
* made in getTwoJetSystem in eval_helper.py
* helpful as jet rotation and scaling takes a long time
2) 
* /nevis/katya01/data/users/kpark/svj-vae/h5dir/jul28/idx
* made in kfold in svj_pfn.py
* helpful b/c it saves numerous infos from kfold e.g. arrays in [events, track, var] form of train set, train indices, direcotories
 
* Troubleshooting: FileNotFoundError: [Errno 2] No such file or directory: '/nevis/katya01/data/users/kpark/svj-vae/results/test/07_30_23_08_14/2/plots/inputs_PFN_2jAvg_MM_ws_NSNR_01.png'
-> This is because the original directory has been removed and the hdf5 has an plot_dir location saved of that directoy so it can't do anything about it 
-> sol: delete the content in idx directory and rerun the code  
"""
#for n_neuron in [40, 150]:
for n_neuron in [75,40, 150]:
  param1=Param( bkg_events=bkg_events, sig_events=sig_events, n_neuron=n_neuron)
#  stdoutOrigin=param1.open_print()
#  param1.example_skf()
  param1.prepare()
  all_dir=param1.all_dir
  
#  all_dir=param1.do_apply(all_dir="/nevis/katya01/data/users/kpark/svj-vae/results/test/07_30_23_09_19/")
  all_dir=param1.do_apply(all_dir=all_dir)
  title=f'track={param1.max_track}'
  grid_scan(title,all_dir=all_dir)
#  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info()) 
  sys.exit()
"""
type in param1.prepare() below!
for phi_dim in [32, 128]:
  param1=Param( bkg_events=bkg_events, sig_events=sig_events, phi_dim=phi_dim)
  stdoutOrigin=param1.open_print()
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

for learning_rate in [0.0005,0.002]:
  param1=Param(bkg_events=bkg_events, sig_events=sig_events,  learning_rate=learning_rate)
  stdoutOrigin=param1.open_print()
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

for nepochs in [50, 200]:
  param1=Param( nepochs=nepochs, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

for batchsize_pfn in [256, 1024]:
  param1=Param( batchsize_pfn=batchsize_pfn, bkg_events=bkg_events, sig_events=sig_events)
  stdoutOrigin=param1.open_print()
  print(param1.close_print(stdoutOrigin)) 
  print(param1.save_info())

"""

#original
#element_size = 4 # change here

sys.exit()

