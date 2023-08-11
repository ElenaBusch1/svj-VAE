import uproot
import numpy as np
import awkward as ak

import os
from termcolor import cprint
import sys

def get_spaced_elements(arr_len,nElements):
    return np.round(np.linspace(0,arr_len-1, nElements)).astype(int)

def get_weighted_elements(tree, nEvents, seed=0):
    weight_array=["weight"]
    my_weight_array = tree.arrays(weight_array, library = "np")
    my_weight_array = my_weight_array[weight_array[0]]
    np.random.seed(seed)
    idx = np.random.choice( my_weight_array.size,size= nEvents, p=my_weight_array/float(my_weight_array.sum()),replace=False) # IMPT that replace=False so that event is picked only once
    idx = np.sort(idx)
    return idx

def read_flat_vars(infile, nEvents, variable_array,seed, bool_weight=True, bool_select_all=False):
    file = uproot.open(infile)
    
    tree = file["PostSel"]
    
    # Read flat branches from nTuple
    my_array = tree.arrays(variable_array, library="np")
    if not(bool_select_all):
      if (bool_weight):
          idx = get_weighted_elements(tree, nEvents,seed)
      else:
          idx = get_spaced_elements(len(my_array[variable_array[0]]),nEvents)

    else:
      print('selecting all events')
      length=len(my_array[variable_array[0]])
      idx=np.array(list(range(length)))
    selected_array = np.array([val[idx] for _,val in my_array.items()]).T
 
    return selected_array


def read_vectors(infile, nEvents,jet_array,seed,max_track,bool_weight=True,bool_select_all=False):
    file = uproot.open(infile)
    try:tree = file["PostSel"]
    except:   tree = file["outTree"]
    """
    if(infile.find("Small") != -1): myTree = "outTree"
    else: myTree = "PostSel"
    """
    tree = file[myTree]
    my_jet_array = tree.arrays(jet_array, library = "np")
    if not(bool_select_all):
      if bool_weight:
        idx = get_weighted_elements(tree, nEvents,seed)
      else: 
    # select evenly spaced events from input distribution
        idx = get_spaced_elements(len(my_jet_array[jet_array[0]]),nEvents)
    else:
      print('selecting all events')
      length=len(my_jet_array[jet_array[0]])
      idx=np.array(list(range(length)))
      
#      idx=list(range(len(my_jet_array))) # this is wrong and gives an error gives (7,15, 7) instead of something like (631735, 15, 7) 
    
    selected_jet_array = np.array([val[idx] for _,val in my_jet_array.items()]).T
    j=idx[0] # REMOVE this line
    k=j+1
    a=idx[0]
    """
    try:  
      cprint(f'{my_jet_array["jet0_GhostTrack_pt"][a]=}', 'magenta') # my_jet_array[var][nth event]=[ 0th track, 1st track, ...] 
      cprint(f'{selected_jet_array[a,0]=}', 'magenta') # selected_jet_array[nth event, nth var]=[0th track, 1st track, ...] 
    except:
      cprint(f'{my_jet_array["jet1_GhostTrack_pt"][a]=}', 'magenta') # my_jet_array[var][nth event]=[ 0th track, 1st track, ...] 
      cprint(f'{selected_jet_array[a,0]=}', 'magenta') # selected_jet_array[nth event, nth var]=[0th track, 1st track, ...] 
    """ 
    # create jet matrix
    padded_jet_array = np.zeros((len(selected_jet_array),max_track,len(jet_array)))
    for jets,zeros in zip(selected_jet_array,padded_jet_array):
        jet_ar = np.stack(jets, axis=1)[:max_track,:]
        zeros[:jet_ar.shape[0], :jet_ar.shape[1]] = jet_ar

#    print('-'*50)
    return padded_jet_array

def main():
    read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 100, ['mT_jj', 'met_met'])
    read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 200, ['mT_jj', 'met_met'])
    read_vectors("../v8.1/user.ebusch.QCDskim.mc20e.root", 100, ['jet0_GhostTrack_pt'])

if __name__ == '__main__':
    main()
