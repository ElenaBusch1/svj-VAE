import numpy as np
from plot_helper import *
from models import *
from sklearn.preprocessing import MinMaxScaler

def check_weights(x_events):
    bkg_nw = read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", x_events, ["jet1_pt"], use_weight=False)
    bkg_w = read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", x_events, ["jet1_pt"], use_weight=True)
    sig_nw = read_flat_vars("../v8.1/user.ebusch.SIGskim.mc20e.root", x_events, ["jet1_pt"], use_weight=False)
    plot_single_variable([bkg_nw,bkg_w, sig_nw], ["QCD No Weights", "QCD Weights", "SIG No Weights"], "QCD Weight Check", logy=True) 

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
    #y = x[:,-60:,:]
    return x

def apply_TrackSelection(x_raw, jets):
    x = np.copy(x_raw)
    x[x[:,:,0] < 10] = 0 # apply pT requirement
    print("Input track shape: ", x.shape)
    # require at least 3 tracks
    x_nz = np.array([len(jet.any(axis=1)[jet.any(axis=1)==True]) >= 3 for jet in x])
    x = x[x_nz]
    jets = jets[x_nz]
    print("Selected track shape: ", x.shape)
    print("Selected jet shape: ", jets.shape)
    print()
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
        jet_eta_av = (jet[e,0] + jet[e,2])/2.0 
        jet_phi_av = (jet[e,1] + jet[e,3])/2.0 
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

