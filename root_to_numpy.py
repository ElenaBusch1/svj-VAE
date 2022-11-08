import uproot
import numpy as np
import awkward as ak

variable_array = ["jet1_pt", "met_met", "dphi_min", "pt_balance_12", "mT_jj", "rT", "dR_12", "deltaY_12", "deta_12", "hT", "maxphi_minphi", "n_r04_jets"]
jet_array = ["all_jets_pt", "all_jets_eta", "all_jets_phi", "all_jets_E"]

def read_hlvs(infile, nEvents):
	file = uproot.open(infile)
	
	#print("File keys: ", file.keys())
	
	tree = file["PostSel"]
	#print("Tree Variables: ", tree.keys())

	# A random 6 variables	
	my_array = tree.arrays(variable_array, library="np")
	selected_array = np.array([val[:nEvents] for _,val in my_array.items()]).T
	#print("My array:")
	print(selected_array)
	#print(type(selected_array))
	#print(selected_array.shape)

	return selected_array

def read_vectors(infile, nEvents):
	file = uproot.open(infile)
	
	#print("File keys: ", file.keys())
	max_jets = 15	

	tree = file["PostSel"]
	#print("Tree Variables: ", tree.keys())

	my_jet_array = tree.arrays(jet_array, library = "np")
	selected_jet_array = np.array([val[:nEvents] for _,val in my_jet_array.items()]).T
	padded_jet_array = np.zeros((len(selected_jet_array),max_jets+1,4))
	my_met_array = tree.arrays(["met_met", "met_phi"], library = "np")
	selected_met_array = np.array([val[:nEvents] for _,val in my_met_array.items()]).T
	for jets,zeros,met in zip(selected_jet_array,padded_jet_array,selected_met_array):
		jet_ar = np.stack(jets, axis=1)[:max_jets,:]
		zeros[0,0] = met[0]
		zeros[0,2] = met[1]
		zeros[1:jet_ar.shape[0]+1, :jet_ar.shape[1]] = jet_ar
	padded_jet_array = padded_jet_array.reshape(len(selected_jet_array),(max_jets+1)*4)
	print(padded_jet_array)

	return padded_jet_array


def main():
	read_vectors("../smallBackground.root", 50)

if __name__ == '__main__':
	main()
