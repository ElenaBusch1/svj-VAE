import uproot
import numpy as np
import awkward as ak

import os
from termcolor import cprint
import sys
variable_array = ["jet1_pt", "met_met", "dphi_min", "pt_balance_12", "mT_jj", "rT", "dR_12", "deltaY_12", "deta_12", "hT", "maxphi_minphi", "n_r04_jets"]
#jet_array = ["all_jets_pt", "all_jets_eta", "all_jets_phi", "all_jets_E"]
## Track array
#jet_array = ["jet_GhostTrack_pt_1", "jet_GhostTrack_eta_1", "jet_GhostTrack_phi_1", "jet_GhostTrack_e_1"] #"jet_GhostTrack_d0_0", "jet_GhostTrack_z0_0", "jet_GhostTrack_qOverP_0"]
#jet_array = ["all_jets_pt", "all_jets_eta", "all_jets_phi", "all_jets_E"]

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


def read_test_variables(infile, nEvents, variables):
    file = uproot.open(infile)

    tree = file["PostSel"]

    # Select nEvent for each requested variable
    my_dict = tree.arrays(variables, library="np") 
    idx = get_spaced_elements(len(my_dict[variables[0]]),nEvents)
    for key in my_dict.keys():
        my_dict[key] = my_dict[key][idx]
    return my_dict

def read_hlvs(infile, nEvents, variable_array, seed, bool_weight=False): # different from variable_array on the top of this file
    file = uproot.open(infile)
	
	
    tree = file["PostSel"]

	# A random 6 variables	
    my_array = tree.arrays(variable_array, library="np")
#    if nEvents > len(my_array):
    
          
    try:
      if bool_weight:
        idx=get_weighted_elements(tree, nEvents, seed)
#      print('weighted sampling idx', idx.shape, idx)

      else: 
    # select evenly spaced events from input distribution
        idx = get_spaced_elements(len(my_array[variable_array[0]]),nEvents)
#      print(f'{len(my_jet_array[jet_array[0]])/nEvents}th idx', idx.shape, idx)
#      print(my_jet_array[jet_array[0]])
#      print('evenly spaced sampling idx', idx.shape, idx)

    #padded_array=selected_jet_array
    except:
      print('Given nEvents larger than events present so selecting all events')
#      print(my_array)
#      print(my_array)
#      print(my_array.size)
      length=len(my_array[variable_array[0]])
      idx=list(range(length))

    selected_array = np.array([val[idx] for _,val in my_array.items()]).T
#    """ 
    cprint(f'{bool_weight=}', 'magenta')
    cprint(f'{len(idx)=}{idx=}', 'magenta') 
    cprint(f'{idx[0]=}', 'magenta') 
    a=idx[0]
    cprint(f'{my_array["jet1_eta"]=}', 'blue') 
    cprint(f'{selected_array=}', 'blue')
    cprint(f'{my_array["jet1_eta"][a]=}', 'magenta') # my_array[var][nth event]= val # a single value instead of an array of [0th track, 1st track, ...] 
    cprint(f'{selected_array[a,0]=}', 'magenta') # selected_array[nth event, nth var]= val # a single value instead of an array of [0th track, 1st track, ...] 
    cprint(f'{my_array["jet1_phi"][a]=}', 'blue') 
    cprint(f'{selected_array[a,1]=}', 'blue')
    #cprint(f'{selected_array[a,2]=}', 'blue')
    
    cprint(f'{my_array["jet1_eta"].shape=}', 'yellow') 
    cprint(f'{selected_array.shape=}', 'yellow') 
#    """ 
    cprint(f'given events{nEvents}, selecting events {len(idx)}','green')
	#print("My array:")
	#print(selected_array)
	#print(type(selected_array))
    
    #print('SEL',selected_array.shape,selected_array)   

#    print('before-'*50)
#    print(f'{infile}\n{selected_array.shape}\n{selected_array}')
    max_jets=15 
    unique_variable_array=list(set([x.split("_")[-1] for x in variable_array])) # unique variable name without 'jetx_' 
 #   print('&'*50)
 #   print( f'{infile}\n{selected_array.shape}\n{selected_array}')
    """
    
    ### Two different ways to reshape the 2D to 3D array of shape (5000, max_jets, 2) where 2 is for 2 hlvs
    # 1st method: loop -> less efficient, but more intuitive
    padded_array= np.zeros((len(selected_array),max_jets,len(unique_variable_array)))
    for e in range(selected_array.shape[0]):
      padded_array[e, 0,0]=selected_array[e,0]
      padded_array[e, 1,0]=selected_array[e,1]
      padded_array[e, 0,1]=selected_array[e,2]
      padded_array[e, 1,1]=selected_array[e,3]
    print('-'*50)
    print(f'{infile}\n{padded_array.shape}\n{padded_array}')
    #"""
 
    # 2nd method: use reshape, transpose and pad functions -> more efficient, but less intuitive 
    size=2  # b/c jet1_pt and jet2_pt
    length=2 # b/c eta and phi
    padded_array=selected_array.reshape(selected_array.shape[0],size, length).transpose(0,2,1)
    pad_width =((0,0),(0, max_jets-size), (0,0))
    padded_array=np.pad(padded_array,pad_width=pad_width, constant_values= 0) # (nth event, nth jet, nth var)
#    print('!'*50)
#    """ 
    cprint(f'{padded_array[a,:,0]=}', 'blue') # should all be the same 
    cprint(f'{padded_array[a,:,1]=}', 'blue') # should all be the same
    #cprint(f'{padded_array[a,:,1]=}', 'blue') # should all be the same
    print(f'{padded_array.shape=}\n{padded_array=}')
 #   """ 
    #sys.exit() 
    #return selected_array

    
    ##############

    return padded_array
 
def read_flat_vars(infile, nEvents, variable_array,seed, bool_weight=True):
    file = uproot.open(infile)
    
    tree = file["PostSel"]
    
    # Read flat branches from nTuple
    my_array = tree.arrays(variable_array, library="np")
    try:
      if (bool_weight):
          idx = get_weighted_elements(tree, nEvents,seed)
      else:
          idx = get_spaced_elements(len(my_array[variable_array[0]]),nEvents)

    except:
      idx=list(range(len(my_array)))
    #print('Flat variable index:', idx.shape, idx)
    selected_array = np.array([val[idx] for _,val in my_array.items()]).T
 
    return selected_array


#def read_vectors(infile, nEvents, jet_array, bool_weight=True):
#def read_vectors(infile, nEvents, bool_weight=True, flatten=True):
def read_vectors(infile, nEvents,jet_array,seed,bool_weight=True,bool_select_all=False):
    file = uproot.open(infile)
    
    #print("File keys: ", file.keys())
    max_jets = 15

    try:tree = file["PostSel"]
    except:   tree = file["outTree"]
    #print("Tree Variables: ", tree.keys())
    """
    branch = "outTree"
    h5dir="/nevis/katya01/data/users/kpark/SVJ/h5files/pfn/"
    filename=infile.split('.')[-2]      
    if not os.path.exists(h5dir+filename+'.h5'):
      branches=tree.arrays(library="pd")
      try:branches[0].to_hdf(f'{h5dir}{filename}.h5', branch)
      except: 
        print('failed to use branches[0] so using something different')
        branches.to_hdf(f'{h5dir}{filename}.h5', branch)
    else: branches=pd.read_hdf(h5dir+filename+'.h5')
    tree_dict={}
    tree_dict[filename]=branches
    print(tree_dict)
     
    sys.exit()
    """
    my_jet_array = tree.arrays(jet_array, library = "np")
    try:
      if bool_weight:

        idx = get_weighted_elements(tree, nEvents,seed)

      else: 
#      print('*'*30)
    # select evenly spaced events from input distribution
        idx = get_spaced_elements(len(my_jet_array[jet_array[0]]),nEvents)
    except:
      length=len(my_jet_array[jet_array[0]])
      idx=list(range(length))
     
#      idx=list(range(len(my_jet_array))) # this is wrong and gives an error gives (7,15, 7) instead of something like (631735, 15, 7) 

    
    selected_jet_array = np.array([val[idx] for _,val in my_jet_array.items()]).T
    j=idx[0] # REMOVE this line
    k=j+1
    cprint(f'{bool_weight=}', 'magenta')
    cprint(f'{len(idx)=}{idx=}', 'magenta') 
    a=idx[0]
    """
    try:  
      cprint(f'{my_jet_array["jet0_GhostTrack_pt"][a]=}', 'magenta') # my_jet_array[var][nth event]=[ 0th track, 1st track, ...] 
      cprint(f'{selected_jet_array[a,0]=}', 'magenta') # selected_jet_array[nth event, nth var]=[0th track, 1st track, ...] 
      cprint(f'{my_jet_array["jet0_GhostTrack_eta"][a]=}', 'blue') 
      cprint(f'{selected_jet_array[a,1]=}', 'blue') 
      cprint(f'{my_jet_array["jet0_GhostTrack_pt"].shape=}', 'yellow') 
    except:
      cprint(f'{my_jet_array["jet1_GhostTrack_pt"][a]=}', 'magenta') # my_jet_array[var][nth event]=[ 0th track, 1st track, ...] 
      cprint(f'{selected_jet_array[a,0]=}', 'magenta') # selected_jet_array[nth event, nth var]=[0th track, 1st track, ...] 
      cprint(f'{my_jet_array["jet1_GhostTrack_eta"][a]=}', 'blue') 
      cprint(f'{selected_jet_array[a,1]=}', 'blue')
      cprint(f'{my_jet_array["jet1_GhostTrack_pt"].shape=}', 'yellow') 
    cprint(f'{selected_jet_array.shape=}', 'yellow')
    """ 
    """
    print(f'my_jet_array[jet_array[0]], {my_jet_array[jet_array[0]]}') #dict  {'jet_GhostTrack_pt_0': array([array([ 1.0182312,...
    print(j)
    print(my_jet_array[jet_array[0]][j],my_jet_array[jet_array[1]][j], my_jet_array[jet_array[2]][j], my_jet_array[jet_array[3]][j])
    print('/'*15)
    print(k)
    print(my_jet_array[jet_array[0]][k],my_jet_array[jet_array[1]][k], my_jet_array[jet_array[2]][k], my_jet_array[jet_array[3]][k])
    print('selected_jet_array')
    print(selected_jet_array)
    sys.exit()
    """
    # create jet matrix
    padded_jet_array = np.zeros((len(selected_jet_array),max_jets,len(jet_array)))
    for jets,zeros in zip(selected_jet_array,padded_jet_array):
        jet_ar = np.stack(jets, axis=1)[:max_jets,:]
        zeros[:jet_ar.shape[0], :jet_ar.shape[1]] = jet_ar

#    print('-'*50)
    """ 
    cprint(f'{padded_jet_array[a,:,0]=}', 'blue')
    cprint(f'{padded_jet_array[a,:,1]=}', 'blue')
    print(f'{padded_jet_array.shape=}\n{padded_jet_array=}')
    """ 
    return padded_jet_array

def read_vectors_MET(infile, nEvents, flatten=True):
    file = uproot.open(infile)
    
    #print("File keys: ", file.keys())
    max_jets = 15    

    tree = file["PostSel"]
    #print("Tree Variables: ", tree.keys())

    # select evenly spaced events from input distribution
    my_jet_array = tree.arrays(jet_array, library = "np")
    my_eventNumber_array = tree.arrays(["eventNumber"], library = "np")
    idx = get_spaced_elements(len(my_jet_array[jet_array[0]]),nEvents)
    my_met_array = tree.arrays(["met_met", "met_phi"], library = "np")
    selected_met_array = np.array([val[idx] for _,val in my_met_array.items()]).T
    selected_jet_array = np.array([val[idx] for _,val in my_jet_array.items()]).T

    # create jet matrix
    padded_jet_array = np.zeros((len(selected_jet_array),max_jets+1,4)) # (nth event, nth track, nth var)
    for jets,zeros,met in zip(selected_jet_array,padded_jet_array,selected_met_array):
        jet_ar = np.stack(jets, axis=1)[:max_jets,:]
        zeros[0,0] = met[0] #pt energy
        zeros[0,2] = met[1] #phi
        zeros[0,3] = met[0] #total energy = pt
        zeros[1:jet_ar.shape[0]+1, :jet_ar.shape[1]] = jet_ar

    if (flatten):
        padded_jet_array = padded_jet_array.reshape(len(selected_jet_array),(max_jets+1)*4)
    #print(padded_jet_array)

    return padded_jet_array


def main():
    read_test_variables("../v6.4/v6p4smallQCD.root", 100, ['mT_jj', 'met_met'])

if __name__ == '__main__':
    main()
