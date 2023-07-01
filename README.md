# SVJ s-channel PFN and ANTELOPE

Codes for training & evaluating the SVJ PFN and associated architecutres

## Files

Input files can be found in the shared eos area: /eos/atlas/atlascerngroupdisk/phys-exotics/jdm/svjets-schannel, or check /data/users/ebusch/SVJ/autoencoder/ on katya.

## Conda environment

Create a conda enviroment from the svj\_env.yml file. Key packages are:
`h5py tensorflow keras numpy matplotlib pandas scikit-learn`

**Note**: uproot is not included in the conda environment. The Columbia group uses a local installation of uprooti (v4.3.5), which may be more efficient


## Running & Evaluating

Training is done with the file `svj_pfn.py`, in svj conda environment.
```
python svj_pfn.py
```

You can save your trained network, and evaluate using the `pfn_evaluate.py` script.
```
python pfn_evaluate.py
```

**Note**: plot path is hardcoded in `plot_heler.py`

## File descriptions
`antelope_evaluate.py`: evaluate trained ANTELOPE model on more data and save HDF5s  
`antelope_h5eval.py`: evaluate HDF5s (AUC, grid plots, sensitivity, etc)  
`eval_helper.py`: functions used in training and evaluation, including to read in and apply track selection to jets  
`evaluate.py`: evaluate simple AE (defunct)  
`models.py`: model architectures  
`models_archive.py`: old models that we might work on again (defunct)  
`pfn_evaluate`: evaluate trained PFN model on more data and save HDF5s  
`plot_helper.py`: plotting scripts  
`root_to_numpy.py`: functions to load data from nTuples  
`svj_antelope.py`: train ANTELOPE (requires trained PFN)  
`svj_pfn.py`: train PFN  
`svj_vae.py`: train simple AE (defunct)  
