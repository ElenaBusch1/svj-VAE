import uproot
import numpy as np

def read_files(infile):
	file = uproot.open(infile)
	
	print(file.keys())
	
	tree = file["PostSel"]
	print(tree.keys())

	# A random 6 variables	
	my_array = tree.arrays(["jet1_pt", "met_met", "dphi_min", "pt_balance_12", "mT_jj", "rT", "dR_12", "deltaY_12", "deta_12", "hT", "maxphi_minphi", "n_r04_jets"], library="np")
	print(my_array)

	return my_array

read_files("smallBackground.root")
