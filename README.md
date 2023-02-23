# SVJ s-channel AutoEncoder

Codes for training & evaluating the SVJ autoencoder

## Files

Input files can be found in the shares eos area: /eos/atlas/atlascerngroupdisk/phys-exotics/jdm/svjets-schannel

## Conda environment

Create a conda enviroment from the svj\_env.yml file. Key packages are:
`h5py tensorflow keras numpy matplotlib pandas scikit-learn`

**Note**: uproot is not included in the conda environment. The Columbia group uses a local installation of uprooti (v4.3.5), which may be more efficient


## Running & Evaluating

Training is done with the file `svj_vae.py`, in svj conda environment.
```
python svj_vae.py
```

You can optionally save your trained network, and evaluate using the evaluate.py script.
```
python evaluate.py
```

**Note**: currently there is too much copy pasting between training and eval scripts, this should be fixed\\
**Note**: plot path is hardcoded in `plot_heler.py`

## File descriptions
`svj_vae.py`: load training/valdiation/testing background & signal, train, save, evaluate  
`evaluate.py`: load testing background & signal, load saved model, evaluate  
`plot_helper.py`: plotting scripts  
`eval_helper.py`: functions used in evaluation
`models.py`: models classes and functions to construct and compile models 
`root_to_numpy.py`: loads data from nTuples
`models_archive.py`: a place to save old code that might be used in small tests
`svj_gvae.py`: specialized training script for probing PFN-AE latent spaces
