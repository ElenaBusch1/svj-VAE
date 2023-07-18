import numpy as np
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import MaxAbsScaler
from plot_helper import *
from models import *
from termcolor import cprint
import json
import h5py 
#def getTwoJetSystem(x_events,y_events, tag_file, tag_title, bool_weight, sig_file,bkg_file="user.ebusch.QCDskim.mc20e.root",extraVars=[], plot_dir=''):
def getTwoJetSystem(nevents,input_file, track_array0, track_array1, jet_array,seed,max_track, plot_dir,extraVars=[], bool_weight=True, bool_pt=True, h5_dir=''):
    
    getExtraVars = len(extraVars) > 0
    h5path=f'{h5_dir}/{input_file}_s={seed}_ne={nevents}_mt={max_track}.hdf5'
    str_ls=['bkg', 'vars_bkg', 'bkg_sel', 'jet_bkg', 'bkg_in0', 'bkg_in1']
    
    bkg, vars_bkg, bkg_sel, jet_bkg, bkg_in0, bkg_in1=np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    data_ls=[bkg,vars_bkg, bkg_sel, jet_bkg, bkg_in0,bkg_in1]



    if  os.path.exists(h5path):
      with h5py.File(h5path, 'r') as f:
        for i in range(len(str_ls)):  
          data_ls[i] = f["default"][str_ls[i]][()]
          
      bkg, vars_bkg, bkg_sel, jet_bkg, bkg_in0, bkg_in1=[*data_ls]
      print('file already exists')
    else:

      cprint(f'reading {input_file}','red')
      read_dir='/nevis/katya01/data/users/ebusch/SVJ/autoencoder/v8.1/'
      input_path=read_dir+input_file
      cprint(nevents, 'red')
      print(input_path)
      print('bkg_in0') 
      bkg_in0 = read_vectors(input_path, nevents, track_array0,seed=seed, max_track=max_track,bool_weight=bool_weight)
      print('bkg_in1') 
      bkg_in1 = read_vectors(input_path, nevents, track_array1, seed=seed,max_track=max_track,bool_weight=bool_weight)
      print('jet_bkg') 
  
      memoryissue=False
      if memoryissue: return [],[],[],[], bkg_in0, bkg_in1
  
  
  
      jet_bkg = read_flat_vars(input_path, nevents, jet_array,seed=seed, bool_weight=bool_weight)  # select with weight??, no need for max_track info
      #jet_bkg = read_hlvs(input_path, nevents, jet_array, bool_weight=bool_weight)  # select with weight??
  
      if getExtraVars: 
        vars_bkg = read_flat_vars(input_path, nevents, extraVars,seed=seed, bool_weight=bool_weight) # no need for max_track info
  
      #bkg_in0, x0, bkg_in1, x1
      #plot_vectors_jet(jet_bkg,jet_sig,jet_array, tag_file=tag_file, tag_title=tag_title, plot_dir=plot_dir)
      print('for leading jet') 
      tag0="jet1"
      tag1="jet2"
      tag3="jet12"
      tag2=input_file.split('.')[-2]
         
      cprint(jet_bkg, 'yellow') 
      cprint(f'{jet_bkg.shape=}', 'yellow') 
      cprint(f'{bkg_in0.shape=}', 'yellow') 
      cprint(f'{bkg_in1.shape=}', 'yellow') 
       
      """
      
      x_0_0, _, bkg_nz0_0,x_pt_0_0 = apply_TrackSelection(bkg_in0, jet_bkg, ntrack=0, bool_pt=bool_pt)
      x_1_0, _, bkg_nz1_0,x_pt_1_0 = apply_TrackSelection(bkg_in1, jet_bkg, ntrack=0, bool_pt=bool_pt)
      bkg_nz_0 = bkg_nz0_0 & bkg_nz1_0 
      
      
      x_0_1, _, bkg_nz0_1,x_pt_0_1  = apply_TrackSelection(bkg_in0, jet_bkg, ntrack=1, bool_pt=bool_pt)
      x_1_1, _, bkg_nz1_1,x_pt_1_1 = apply_TrackSelection(bkg_in1, jet_bkg, ntrack=1, bool_pt=bool_pt)
      bkg_nz_1 = bkg_nz0_1 & bkg_nz1_1 
  
      x_0_2, _, bkg_nz0_2,x_pt_0_2 = apply_TrackSelection(bkg_in0, jet_bkg, ntrack=2, bool_pt=bool_pt)
      x_1_2, _, bkg_nz1_2,x_pt_1_2  = apply_TrackSelection(bkg_in1, jet_bkg, ntrack=2, bool_pt=bool_pt)
      bkg_nz_2 = bkg_nz0_2 & bkg_nz1_2 
      """
  
      # select events which have both valid leading and subleading jet tracks
      #with pt requirement and track selection
      x_0, _, bkg_nz0,x_pt_0 = apply_TrackSelection(bkg_in0, jet_bkg, bool_pt=bool_pt) #x_0 is leading jet 1 indices applied, x_pt_0 only pt selection applied -> so should use x_pt_0 for the compatibility of dimension with bkg_nz
      x_1, _, bkg_nz1, x_pt_1 = apply_TrackSelection(bkg_in1, jet_bkg, bool_pt=bool_pt)
      bkg_nz = bkg_nz0 & bkg_nz1
      cprint(f'{x_0.shape=}, {x_1.shape=}, {x_pt_0.shape=}, {x_pt_1.shape=},{bkg_nz.shape=}, {bkg_nz0.shape=}, {bkg_nz1.shape=}')
      bkg_pt0 = x_pt_0[bkg_nz]  # with pt and track requirement
      #bkg_pt0 = bkg_in0[bkg_nz]  # with pt and track requirement
  
     
      # WHAT ABOUT JETS? PT SELECTION? 
      """
      hist0=[x_pt_0_0, x_pt_0_0[bkg_nz0_0], x_pt_0_0[bkg_nz0_1],x_pt_0_0[bkg_nz0_2], x_pt_0_0[bkg_nz0]] # leading jet ntrack comparison
      hist1=[x_pt_1_0, x_pt_1_0[bkg_nz1_0], x_pt_1_0[bkg_nz1_1],x_pt_1_0[bkg_nz1_2], x_pt_1_0[bkg_nz1]] # leading jet ntrack comparison
      hist2=[x_pt_0_0, x_pt_0_0[bkg_nz_0], x_pt_0_0[bkg_nz_1],x_pt_0_0[bkg_nz_2], bkg_pt0] # leading jet ntrack comparison
  
      plot_ntrack(hist1,  tag_file=tag1+tag2, tag_title=tag1+tag2, plot_dir=plot_dir)
      plot_ntrack(hist0,  tag_file=tag0+tag2, tag_title=tag0+tag2, plot_dir=plot_dir)
      plot_ntrack(hist2,  tag_file=tag3+tag2, tag_title=tag3+tag2, plot_dir=plot_dir)
      """
  
  
      #bkg_pt1 = bkg_in1[bkg_nz]
      bkg_pt1 = x_pt_1[bkg_nz]
  
      bjet_sel = jet_bkg[bkg_nz]
      cprint('for both leading and subleading jet')
      cprint(f"{bkg_pt0.shape=}", 'red')
      cprint(f"{bkg_pt1.shape=}", 'blue')
      if getExtraVars:
        vars_bkg = vars_bkg[bkg_nz]   # vars_bkg -> be careful if there's pt for track selection 
  #    sys.exit()
      bkg_sel = np.concatenate((bkg_pt0,bkg_pt1),axis=1)
      cprint(f"{bkg_sel.shape=}", 'blue')
      
  #    print('bkg_sel0, bkg_sel1',bkg_sel.shape, bkg_sel1.shape)
  # ADDED 5/22/23 
  
  #    plot_vectors(bkg_sel,sig_sel,tag_file=tag_file, tag_title=tag_title, plot_dir=plot_dir)
      bkg = apply_JetScalingRotation(bkg_sel, bjet_sel,0)
  
  
      data_ls=[bkg,vars_bkg, bkg_sel, jet_bkg, bkg_in0,bkg_in1] # reset here
      with h5py.File(h5path,"w") as f:
        dset = f.create_group('default')
        for i in range(len(str_ls)):  
          data= dset.create_dataset(str_ls[i],data=data_ls[i])
  
     
    #write a metadata


    
    cprint(f'{bkg=}', 'red')

    if getExtraVars: return bkg, vars_bkg, bkg_sel, jet_bkg, bkg_in0, bkg_in1
    #if getExtraVars: return bkg, vars_bkg, bkg_sel, jet_bkg
    else: return bkg, np.array([]), bkg_sel, jet_bkg, bkg_in0, bkg_in1

def get_dPhi(x1,x2):
    dPhi = x1 - x2
    if(dPhi > 3.14):
        dPhi -= 2*3.14
    elif(dPhi < -3.14):
        dPhi += 2*3.14
    return dPhi

def remove_zero_padding(x):
    #x has shape (nEvents, nSteps, nFeatures)
    #x_out has shape (nEvents, nFeatures)
    x_nz = np.any(x,axis=2) #find zero padded steps
    x_out = x[x_nz]

    return x_out

def reshape_3D(x, nTracks, nFeatures):
    print(x[4])
    x_out = x.reshape(x.shape[0],nTracks,nFeatures)
    print(x_out[4])
    return x_out

def pt_sort(x):
    for i in range(x.shape[0]):
        ev = x[i]
        x[i] = ev[ev[:,0].argsort()]
    return x

def apply_TrackSelection(x_raw, jets, bool_pt=True, bool_track=True, pt=10, ntrack=3):
    x = np.copy(x_raw)
 
    if bool_pt:
      x[x[:,:,0] < pt] = 0
      # should this be enforced in jets as well? and return the new jets
    
    
    x_pt=x.copy() # KEEP THIS -> IMPT
    print("important that the order of variables is such that pt is the first for this pt selection to work")
    print("Input track shape: ", x.shape)
    # require at least 3 tracks
    x_test=x.copy()
    if bool_track:
#      for ntrack in ntracks:

      x_nz = np.array([len(jet.any(axis=1)[jet.any(axis=1)==True]) >= ntrack for jet in x])
      x = x[x_nz]
    else: 
      x_nz = np.array([len(jet.any(axis=1)[jet.any(axis=1)==True]) >= 0 for jet in x])
    print(f'{x_nz.shape=}')
    print(f'{jets.shape=}')
    """
    first_var=x[:,:,0]
    nevent=first_var.shape[0]
    count=np.count_nonzero(first_var, axis = 1)

    first_var1=x_test[:,:,0]
    nevent1=first_var1.shape[0]
    count1=np.count_nonzero(first_var1, axis = 1)

    bin_min=0
    bin_max=np.max(count)
    print(f'{bin_max=}')
    bins=np.array(range(bin_min-1,bin_max+2))
    x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])

    label=[f'NJ={int(np.sum(count))}, NE={int(nevent)}', f'NJ={int(np.sum(count1))}, NE={int(nevent1)}']
    plt.hist([count,count1], bins=x_bins, label=label, histtype='step')
    plt.title(f'ntrack>={ntrack}')
    plt.legend()
     
    plt.show()
    """
   
    
    """

    x_nz_not=np.invert(x_nz) 
    x_not = x_orig[x_nz_not]
    cprint(f'passing track selections {x[:,:,0].shape=}', 'red')
    cprint(f'indices {x_nz=}', 'red')
    cprint(f' {x[:,:,0]=}', 'red')
    cprint(f' the first event {x[0,:,0]=}', 'red')
    for i in range(4):
      cprint(f'  the ith event{x[i,:,0]=}', 'red')
    print('*'*50)
    cprint(f'NOT passing track selections {x_not[:,:,0].shape=}', 'blue')
    cprint(f'indices {x_nz_not=}', 'blue')
    cprint(f' {x_not[:,:,0]=}', 'blue')
    cprint(f'  the first event{x_not[0,:,0]=}', 'blue')
    for i in range(4):
      cprint(f'  the ith event{x_not[i,:,0]=}', 'blue')
    """
    
    jets = jets[x_nz]
    print("Track selections")
    print("Selected track shape: ", x.shape)
    print("Selected jet shape: ", jets.shape)
    
    
    return x, jets, x_nz,x_pt

def apply_StandardScaling(x_raw, scaler=MinMaxScaler(), doFit=True):
    x= np.zeros(x_raw.shape)
    
    x_nz = np.any(x_raw,axis=len(x_raw.shape)-1) #find zero padded events 
    x_scale = x_raw[x_nz] #scale only non-zero jets
    #scaler = StandardScaler()
    if (doFit): scaler.fit(x_scale) 
    x_fit = scaler.transform(x_scale) #do the scaling
    
    x[x_nz]= x_fit #insert scaled values back into zero padded matrix
    
    return x, scaler

def apply_EventScaling(x_raw):
    
    x = np.copy(x_raw) #copy

    x_totals = x_raw.sum(axis=1) #get sum total pt, eta, phi, E for each event
    x[:,:,0] = (x_raw[:,:,0].T/x_totals[:,0]).T  #divide each pT entry by event pT total
    x[:,:,3] = (x_raw[:,:,3].T/x_totals[:,3]).T  #divide each E entry by event E total

    return x

def apply_JetScalingRotation(x_raw, jet, jet_idx):
   
    if (x_raw.shape[0] != jet.shape[0]):
        print("Track shape", x_raw.shape, "is incompatible with jet shape", jet.shape)
        print("Exiting...")
        return
    
    
    x = np.copy(x_raw) #copy
    x_totals = x_raw.sum(axis=1) #get sum total pt, eta, phi, E for each event
    """
    cprint(f'{x_raw}=','yellow')
    cprint(f'{x_totals[:,0]}=', 'blue')
    """
    print(f'{x_raw.shape}=')


    x[:,:,0] = (x_raw[:,:,0].T/x_totals[:,0]).T  #divide each pT entry by event pT total
    x[:,:,3] = (x_raw[:,:,3].T/x_totals[:,3]).T  #divide each E entry by event E total
    
    cprint(f"{x.shape=}", 'red')
    #jet_phi_avs = np.zeros(x.shape[0])
    print(f'{jet.shape=}')
#    print('x_raw', x_raw.shape, x_raw)
#    print('jet_idx',  jet_idx)
    for e in range(x.shape[0]):
        """
        jet_eta_av = (jet[e,0,0] + jet[e,1,0])/2.0 
        jet_phi_av = (jet[e,0,1] + jet[e,1,1])/2.0 
        cprint(f"{jet_eta_av=}", 'blue')
        cprint(f"{jet_phi_av=}", 'red')
        """
    
#        print('jet_eta_av',jet[e,0,0],jet[e,1,0], jet_eta_av)
# change
        jet_eta_av = (jet[e,0] + jet[e,2])/2.0 
        #jet_eta_av = (jet[e,0] + jet[e,1])/2.0 
        #jet_phi_av = (jet[e,2] + jet[e,3])/2.0 
        jet_phi_av = (jet[e,1] + jet[e,3])/2.0 

        #jet_phi_avs[e] = jet_phi_av
        for t in range(x.shape[1]):
            if not x[e,t,:].any():
                #cprint(f'{x[e,t,:]=}', 'magenta')
                continue
            #if not jet[e,jet_idx,:].any():
            #    x[e,t,:] = 0
            else:
                x[e,t,1] = x_raw[e,t,1] - jet_eta_av # subtrack subleading jet eta from each track
                x[e,t,2] = get_dPhi(x_raw[e,t,2],jet_phi_av) # subtrack subleading jet phi from each track
                """
                cprint(f'{x_raw[e,t,1]=}', 'yellow')
                cprint(f'{jet_eta_av=}', 'blue')
                cprint(f'{x[e,t,1]=}', 'red')
                cprint(f'{x_raw[e,t,2]=}', 'yellow')
                cprint(f'{jet_phi_av=}', 'blue')
                cprint(f'{get_dPhi(x_raw[e,t,2],jet_phi_av)=}', 'magenta')
                cprint(f'{x[e,t,2]=}', 'red')
                """
                #x[e,t,1] = x_raw[e,t,1] - jet[e,jet_idx,0] # subtrack subleading jet eta from each track
                #x[e,t,2] = get_dPhi(x_raw[e,t,2],jet[e,jet_idx,1]) # subtrack subleading jet phi from each track
    #plt.hist(jet_phi_avs)
    #plt.show()
    return x


def get_multi_loss(model_svj, x_test, y_test):
    bkg_total_loss = []
    sig_total_loss = []
    bkg_kld_loss = []
    sig_kld_loss = []
    bkg_reco_loss = []
    sig_reco_loss = []
    nevents = min(len(y_test),len(x_test))
    step_size = 4
    for i in range(0,nevents, step_size):
        xt = x_test[i:i+step_size]
        yt = y_test[i:i+step_size]
      
        # NOTE - unclear why they are printed in this order, but it seems to be the case
        x_loss,x_reco,x_kld = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
        y_loss,y_reco,y_kld = model_svj.evaluate(yt, batch_size = step_size, verbose=0)
      
        bkg_total_loss.append(x_loss)
        sig_total_loss.append(y_loss)
        bkg_kld_loss.append(x_kld)
        sig_kld_loss.append(y_kld)
        bkg_reco_loss.append(x_reco)
        sig_reco_loss.append(y_reco)
        if i%100 == 0: print("Processed", i, "events")

    return bkg_total_loss, sig_total_loss, bkg_kld_loss, sig_kld_loss, bkg_reco_loss, sig_reco_loss

def get_single_loss(model_svj, x_test, y_test):
    bkg_loss = []
    sig_loss = []
    nevents = min(len(y_test),len(x_test))
    step_size = 4
    for i in range(0,nevents, step_size):
        xt = x_test[i:i+step_size]
        yt = y_test[i:i+step_size]
    
        x_loss = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
        y_loss = model_svj.evaluate(yt, batch_size = step_size, verbose=0)
        
        bkg_loss.append(x_loss)
        sig_loss.append(y_loss)
        if i%100 == 0: print("Processed", i, "events")

    return bkg_loss, sig_loss
def equal_length(bkg_loss, sig_loss):
#  np.random.seed(7)
#  bkg_loss=np.random.shuffle(bkg_loss)
#  sig_loss=np.random.shuffle(sig_loss)
  if (len(bkg_loss) > len(sig_loss)): # necessary when computing AUC score
    bkg_loss = bkg_loss[:len(sig_loss)]
#    mT_bkg=mT_bkg[:len(sig_loss)] # added
  else:
    sig_loss = sig_loss[:len(bkg_loss)]
#    mT_sig=mT_sig[:len(sig_loss)] # added
  return bkg_loss,sig_loss


def transform_loss(bkg_loss, sig_loss, make_plot=False, tag_file="", tag_title="", plot_dir=''):
    bkg_loss,sig_loss=equal_length(bkg_loss,sig_loss)
    nevents = len(sig_loss) 
    
    truth_sig = np.ones(nevents)
    truth_bkg = np.zeros(nevents)
    truth_labels = np.concatenate((truth_bkg, truth_sig))
    eval_vals = np.concatenate((bkg_loss,sig_loss))
    eval_min = min(eval_vals)
    eval_max = max(eval_vals)-eval_min
    eval_transformed = [(x - eval_min)/eval_max for x in eval_vals]
    bkg_transformed = [(x - eval_min)/eval_max for x in bkg_loss]
    sig_transformed = [(x - eval_min)/eval_max for x in sig_loss]
    if make_plot:
        plot_score(bkg_transformed, sig_transformed, False, False, tag_file=tag_file+'_Transformed', tag_title=tag_title+'_Transformed', plot_dir=plot_dir)
    return truth_labels, eval_vals 

def getSignalSensitivityScore(bkg_loss, sig_loss, percentile=95):
    nSigAboveThreshold = np.sum(sig_loss > np.percentile(bkg_loss, percentile))
    return nSigAboveThreshold / len(sig_loss)

def applyScoreCut(loss,test_array,cut_val):
    return test_array[loss>cut_val] 

def do_roc(bkg_loss, sig_loss, tag_file, tag_title, make_transformed_plot=False, plot_dir=''):
    truth_labels, eval_vals = transform_loss(bkg_loss, sig_loss, make_plot=make_transformed_plot, tag_file=tag_file, tag_title=tag_title, plot_dir=plot_dir) 
    fpr, tpr, trh = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
    auc = roc_auc_score(truth_labels, eval_vals)
    print("AUC - "+tag_file+": ", auc)
    make_roc(fpr,tpr,auc,tag_file=tag_file, tag_title=tag_title, plot_dir=plot_dir)
    make_sic(fpr,tpr,auc,tag_file=tag_file, tag_title=tag_title, plot_dir=plot_dir)
    return auc

def do_grid_plots(sic_vals, title, plot_dir=''):
    with open("dsids_grid_locations.json", "r") as f:
      dsid_coords = json.load(f)
    dsids = list(sic_vals.keys())
    print(f'{dsids=}')
    vals = list(sic_vals[dsids[0]].keys())
    print(f'{sic_vals=}')
    for val in vals:
        values = np.zeros([4,10])
        for dsid in dsids:
            loc = tuple(dsid_coords[str(dsid)])
            values[loc] = sic_vals[dsid][val]
        make_grid_plot(values, val, title, plot_dir)

