import uproot
import numpy as np
import awkward as ak

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
    file = uproot.open(infile)
    
    max_jets = 100

    tree = file["PostSel"]

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

def main():
    read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 100, ['mT_jj', 'met_met'])
    read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 200, ['mT_jj', 'met_met'])
    read_vectors("../v8.1/user.ebusch.QCDskim.mc20e.root", 100, ['jet0_GhostTrack_pt'])

if __name__ == '__main__':
    main()
