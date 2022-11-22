# SVJ s-channel AutoEncoder

Codes for training & evaluating the SVJ autoencoder

## Files

Input files can be found in the shares eos area (/eos/atlas/atlascerngroupdisk/phys-exotics/jdm/svjets-schannel)

## Conda environment

Create a conda enviroment from the svj\_env.yml file. Key packages are:
`h5py tensorflow keras numpy matplotlib pandas scikit-learn`

## Running & Evaluating

Training is done with the file `svj_vae.py`, in conda environment.
```
python svy_vae.py
```

You can optionally save your trained network, and evaluate using the evaluate.py script.
```
python evaluate.py
```

## File descriptions
`svj_vae.py`: load training/valdiation/testing background & signal, train, save, evaluate
`evaluate.py`: load testing background & signal, load saved model, evaluate
`plot_helper.py`: plotting scripts
`models.py`: AE models
`root_to_numpy.py`: loads data from nTuples
