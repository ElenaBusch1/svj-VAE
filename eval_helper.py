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

def getTwoJetSystem(x_events,y_events):
    track_array0 = ["jet_GhostTrack_pt_0", "jet_GhostTrack_eta_0", "jet_GhostTrack_phi_0", "jet_GhostTrack_e_0"]
    track_array1 = ["jet_GhostTrack_pt_1", "jet_GhostTrack_eta_1", "jet_GhostTrack_phi_1", "jet_GhostTrack_e_1"]
    jet_array = ["jet_eta", "jet_phi"]
    bkg_in0 = read_vectors("../v8/v8SmallPartialQCDmc20e.root", x_events, track_array0)
    sig_in0 = read_vectors("../v8/v8SmallSIGmc20e.root", y_events, track_array0)
    bkg_in1 = read_vectors("../v8/v8SmallPartialQCDmc20e.root", x_events, track_array1)
    sig_in1 = read_vectors("../v8/v8SmallSIGmc20e.root", y_events, track_array1)
    jet_bkg = read_vectors("../v8/v8SmallPartialQCDmc20e.root", x_events, jet_array)
    jet_sig = read_vectors("../v8/v8SmallSIGmc20e.root", y_events, jet_array)

    _, _, bkg_nz0 = apply_TrackSelection(bkg_in0, jet_bkg, True)
    _, _, sig_nz0 = apply_TrackSelection(sig_in0, jet_sig, False)
    _, _, bkg_nz1 = apply_TrackSelection(bkg_in1, jet_bkg, True)
    _, _, sig_nz1 = apply_TrackSelection(sig_in1, jet_sig, False)
    
    bkg_nz = bkg_nz0 & bkg_nz1
    sig_nz = sig_nz0 & sig_nz1

    # select events which have both valid leading and subleading jet tracks
    bkg_pt0 = bkg_in0[bkg_nz]
    bkg_pt1 = bkg_in1[bkg_nz]
    sig_pt0 = sig_in0[sig_nz]
    sig_pt1 = sig_in1[sig_nz]
    bjet_sel = jet_bkg[bkg_nz]
    sjet_sel = jet_sig[sig_nz]

    bkg_sel0 = pt_sort(bkg_pt0, 0)
    bkg_sel1 = pt_sort(bkg_pt1, 1)
    sig_sel0 = pt_sort(sig_pt0, 0)
    sig_sel1 = pt_sort(sig_pt1, 1)

    bkg_sel = np.concatenate((bkg_sel0,bkg_sel1),axis=1)
    sig_sel = np.concatenate((sig_sel0,sig_sel1),axis=1)

    bkg = apply_JetScalingRotation(bkg_sel, bjet_sel,0)
    sig = apply_JetScalingRotation(sig_sel, sjet_sel,0)

    #plot
    #x_sel_nz = remove_zero_padding(bkg_sel)
    #sig_sel_nz = remove_zero_padding(sig_sel)
    #plot_vectors(bkg_sel,sig_sel,"PFNraw")
    #x_nz = remove_zero_padding(bkg)
    #sig_nz = remove_zero_padding(sig)
    #plot_vectors(x_nz,sig_nz,"AErotated_avg")

    #x_2D,scaler = apply_StandardScaling(bkg)
    #sig_2D,_ = apply_StandardScaling(sig, scaler, False)
    #sig_2D,_ = apply_StandardScaling(sig)
    #x = bkg
    #sig = sig

    print(bkg.shape)
    print(sig.shape)
    #x = x_2D.reshape(bkg.shape[0],x_2D.shape[1]*4)
    #sig = sig_2D.reshape(sig_2D.shape[0],x_2D.shape[1]*4)
    #plot_vectors(remove_zero_padding(x_2D),remove_zero_padding(sig_2D),"AEscaled")
    #plot_vectors(x,sig,"AEWithZeroRotated")
    return bkg, sig

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

def pt_sort(x, jet_idx):
    for i in range(x.shape[0]):
        ev = x[i]
        x[i] = ev[ev[:,0].argsort()]
    if (jet_idx == 0):
        y = x[:,-80:,:]
    elif (jet_idx == 1):
        y = x[:,-80:,:]
    else:
        y = x[:,-3:,:]
    return y

def apply_TrackSelection(x_raw, jets, highTrackMult):
    x = np.copy(x_raw)
    x[x[:,:,0] < 10] = 0 # apply pT requirement
    print("Input track shape: ", x.shape)
    # require at least 3 tracks
    x_nz = np.array([len(jet.any(axis=1)[jet.any(axis=1)==True]) >= 3 for jet in x])
    x = x[x_nz]
    jets = jets[x_nz]
    print("Track selections")
    print("Selected track shape: ", x.shape)
    print("Selected jet shape: ", jets.shape)
    return x, jets, x_nz

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
    x[:,:,0] = (x_raw[:,:,0].T/x_totals[:,0]).T  #divide each pT entry by event pT total
    x[:,:,3] = (x_raw[:,:,3].T/x_totals[:,3]).T  #divide each E entry by event E total
    
    #jet_phi_avs = np.zeros(x.shape[0])
    for e in range(x.shape[0]):
        jet_eta_av = (jet[e,0,0] + jet[e,1,0])/2.0 
        jet_phi_av = (jet[e,0,1] + jet[e,1,1])/2.0 
        #jet_phi_avs[e] = jet_phi_av
        for t in range(x.shape[1]):
            if not x[e,t,:].any():
                #print(x[e,t,:])
                continue
            #if not jet[e,jet_idx,:].any():
            #    x[e,t,:] = 0
            else:
                x[e,t,1] = x_raw[e,t,1] - jet_eta_av # subtrack subleading jet eta from each track
                x[e,t,2] = get_dPhi(x_raw[e,t,2],jet_phi_av) # subtrack subleading jet phi from each track
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

def transform_loss(bkg_loss, sig_loss, make_plot=False, plot_tag=''):
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
        plot_score(bkg_transformed, sig_transformed, False, False, plot_tag+'_Transformed')
    return truth_labels, eval_vals 

def getSignalSensitivityScore(bkg_loss, sig_loss, percentile=95):
    nSigAboveThreshold = np.sum(sig_loss > np.percentile(bkg_loss, percentile))
    return nSigAboveThreshold / len(sig_loss)

def applyScoreCut(loss,test_array,cut_val):
    return test_array[loss>cut_val] 

def do_roc(bkg_loss, sig_loss, plot_tag, make_transformed_plot=False):
    truth_labels, eval_vals = transform_loss(bkg_loss, sig_loss, make_plot=make_transformed_plot, plot_tag=plot_tag) 
    fpr, tpr, trh = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
    auc = roc_auc_score(truth_labels, eval_vals)
    print("AUC - "+plot_tag+": ", auc)
    make_roc(fpr,tpr,auc,plot_tag)
    make_sic(fpr,tpr,auc,plot_tag)
    return auc

