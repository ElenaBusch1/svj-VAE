import uproot
import numpy as np
import awkward as ak

variable_array = ["jet1_pt", "met_met", "dphi_min", "pt_balance_12", "mT_jj", "rT", "dR_12", "deltaY_12", "deta_12", "hT", "maxphi_minphi", "n_r04_jets"]
#jet_array = ["all_jets_pt", "all_jets_eta", "all_jets_phi", "all_jets_E"]
## Track array
#jet_array = ["jet_GhostTrack_pt_1", "jet_GhostTrack_eta_1", "jet_GhostTrack_phi_1", "jet_GhostTrack_e_1"] #"jet_GhostTrack_d0_0", "jet_GhostTrack_z0_0", "jet_GhostTrack_qOverP_0"]

def get_spaced_elements(arr_len,nElements):
    return np.round(np.linspace(0,arr_len-1, nElements)).astype(int)

def read_test_variables(infile, nEvents, variables):
    file = uproot.open(infile)

    tree = file["PostSel"]

    # Select nEvent for each requested variable
    my_dict = tree.arrays(variables, library="np") 
    idx = get_spaced_elements(len(my_dict[variables[0]]),nEvents)
    for key in my_dict.keys():
        my_dict[key] = my_dict[key][idx]
    return my_dict

def read_vectors(infile, nEvents, jet_array):
    file = uproot.open(infile)
    
    #print("File keys: ", file.keys())
    max_jets = 40

    tree = file["outTree"]
    #print("Tree Variables: ", tree.keys())

    # select evenly spaced events from input distribution
    my_jet_array = tree.arrays(jet_array, library = "np")
    idx = get_spaced_elements(len(my_jet_array[jet_array[0]]),nEvents)
    selected_jet_array = np.array([val[idx] for _,val in my_jet_array.items()]).T

    # create jet matrix
    padded_jet_array = np.zeros((len(selected_jet_array),max_jets,len(jet_array)))
    for jets,zeros in zip(selected_jet_array,padded_jet_array):
        jet_ar = np.stack(jets, axis=1)[:max_jets,:]
        zeros[:jet_ar.shape[0], :jet_ar.shape[1]] = jet_ar

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
    padded_jet_array = np.zeros((len(selected_jet_array),max_jets+1,4))
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
