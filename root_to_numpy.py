import uproot
import numpy as np

def read_files(infile, nEvents):
	file = uproot.open(infile)
	
	#print("File keys: ", file.keys())
	
	tree = file["PostSel"]
	#print("Tree Variables: ", tree.keys())

	# A random 6 variables	
	my_array = tree.arrays(["jet1_pt", "met_met", "dphi_min", "pt_balance_12", "mT_jj", "rT", "dR_12", "deltaY_12", "deta_12", "hT", "maxphi_minphi", "n_r04_jets"], library="np")
	selected_array = np.array([val[:nEvents] for _,val in my_array.items()]).T
	#print("My array:")
	#print(selected_array)
	#print(type(selected_array))
	#print(selected_array.shape)

	return selected_array

def main():
	read_files("../smallBackground.root", 5)

if __name__ == '__main__':
	main()
