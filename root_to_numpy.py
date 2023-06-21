import uproot
import numpy as np
import awkward as ak
import ipdb
from helper import apply_TrackSelection, apply_JetScalingRotation

sample_dir = '/data/users/ebusch/SVJ/autoencoder/v8.1/'
npy_dir = "npy_inputs/"

def get_spaced_elements(arr_len,nElements):
    return np.round(np.linspace(0,arr_len-1, nElements)).astype(int)

def get_weighted_elements(tree, nEvents):
    weight_array=["weight"]
    my_weight_array = tree.arrays(weight_array, library = "np")
    my_weight_array = my_weight_array[weight_array[0]]
    np.random.seed(0)
    idx = np.random.choice( my_weight_array.size,size= nEvents, p=my_weight_array/float(my_weight_array.sum()),replace=False) # IMPT that replace=False so that event is picked only once
    return idx

def read_flat_vars(infile, nEvents, variable_array, use_weight=True):
    print('reading flat vars...')
    file = uproot.open(infile)
    tree = file["PostSel"]
    
    # Read flat branches from nTuple
    my_array = tree.arrays(variable_array, library="np")
    if (use_weight):
        idx = get_weighted_elements(tree, nEvents)
    else:
        idx = get_spaced_elements(len(my_array[variable_array[0]]),nEvents)

    #print('Flat variable index:', idx.shape, idx)
    selected_array = np.array([val[idx] for _,val in my_array.items()]).T

    return selected_array

def read_vectors(infile, nEvents, jet_array, use_weight=True):
    print('reading vectors...')
    file = uproot.open(infile)
    max_jets = 80

    if(infile.find("Small") != -1): myTree = "outTree"
    else: myTree = "PostSel"
    tree = file[myTree]

    # Read vector branches from nTuple
    my_jet_array = tree.arrays(jet_array, library = "np")
    if (use_weight):
        idx = get_weighted_elements(tree, nEvents)
    else:
        idx = get_spaced_elements(len(my_jet_array[jet_array[0]]),nEvents)

    #print('Vector variable index:', idx.shape, idx)
    selected_jet_array = np.array([val[idx] for _,val in my_jet_array.items()]).T

    # create jet matrix
    padded_jet_array = np.zeros((len(selected_jet_array),max_jets,len(jet_array)))
    for jets,zeros in zip(selected_jet_array,padded_jet_array):
        jet_ar = np.stack(jets, axis=1)[:max_jets,:]
        zeros[:jet_ar.shape[0], :jet_ar.shape[1]] = jet_ar

    return padded_jet_array

def getTwoJetSystem(bkg_events,sig_events, extraVars=[]):
    getExtraVars = len(extraVars) > 0

    track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e", "jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
    track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e", "jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
    jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"]
 
    bkg_in0 = read_vectors(sample_dir+"user.ebusch.QCDskim.mc20e.root", bkg_events, track_array0, use_weight=True)
    sig_in0 = read_vectors(sample_dir+"user.ebusch.SIGskim.mc20e.root", sig_events, track_array0, use_weight=False)
    bkg_in1 = read_vectors(sample_dir+"user.ebusch.QCDskim.mc20e.root", bkg_events, track_array1, use_weight=True)
    sig_in1 = read_vectors(sample_dir+"user.ebusch.SIGskim.mc20e.root", sig_events, track_array1, use_weight=False)
    jet_bkg = read_flat_vars(sample_dir+"user.ebusch.QCDskim.mc20e.root", bkg_events, jet_array, use_weight=True)
    jet_sig = read_flat_vars(sample_dir+"user.ebusch.SIGskim.mc20e.root", sig_events, jet_array, use_weight=False)
    if getExtraVars:
        vars_bkg = read_flat_vars(sample_dir+"user.ebusch.QCDskim.mc20e.root", bkg_events, extraVars, use_weight=True)
        vars_sig = read_flat_vars(sample_dir+"user.ebusch.SIGskim.mc20e.root", sig_events, extraVars, use_weight=False)

    _, _, bkg_nz0 = apply_TrackSelection(bkg_in0, jet_bkg)
    _, _, sig_nz0 = apply_TrackSelection(sig_in0, jet_sig)
    _, _, bkg_nz1 = apply_TrackSelection(bkg_in1, jet_bkg)
    _, _, sig_nz1 = apply_TrackSelection(sig_in1, jet_sig)

    bkg_nz = bkg_nz0 & bkg_nz1
    sig_nz = sig_nz0 & sig_nz1

    # select events which have both valid leading and subleading jet tracks
    bkg_pt0 = bkg_in0[bkg_nz]
    bkg_pt1 = bkg_in1[bkg_nz]
    sig_pt0 = sig_in0[sig_nz]
    sig_pt1 = sig_in1[sig_nz]
    bjet_sel = jet_bkg[bkg_nz]
    sjet_sel = jet_sig[sig_nz]
    if getExtraVars:
        vars_bkg = vars_bkg[bkg_nz]
        vars_sig = vars_sig[sig_nz]

    bkg_sel = np.concatenate((bkg_pt0,bkg_pt1),axis=1)
    sig_sel = np.concatenate((sig_pt0,sig_pt1),axis=1)

    bkg = apply_JetScalingRotation(bkg_sel, bjet_sel,0)
    sig = apply_JetScalingRotation(sig_sel, sjet_sel,0)

    print(bkg.shape)
    print(sig.shape)
    if getExtraVars: return bkg, sig, vars_bkg, vars_sig
    else: return bkg, sig

def main():
    #read_flat_vars(file_name, 100, ['mT_jj', 'met_met'])
    #read_flat_vars(file_name, 200, ['mT_jj', 'met_met'])
    #read_vectors(file_name, 100, ['jet0_GhostTrack_pt'])

    bkg_events = 200000
    sig_events = 20000

    bkg, sig = getTwoJetSystem(bkg_events, sig_events)
    np.save(npy_dir+"bkg_n"+str(bkg_events)+".npy", bkg)
    np.save(npy_dir+"sig_n"+str(sig_events)+".npy", bkg)

if __name__ == '__main__':
    main()
