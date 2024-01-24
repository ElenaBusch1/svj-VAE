# SVJ s-channel PFN and ANTELOPE
Codes for training & evaluating the SVJ PFN and associated architecutre
s
## Index of the documentation
1) Organization of the directory
2) Organization of the files
3) Getting started (Installation and Prerequisites)
4) Usage


## Organization of the directory
1) main files to run
`svj_antelope.py`: train ANTELOPE (requires trained PFN)  
`svj_pfn.py`: train PFN  
`pfn_evaluate`: evaluate trained PFN and/or ANTELOPE model on data and save HDF5s  

2) helper files for main files listed above
`antelope_h5eval.py`: evaluate HDF5s (AUC, grid plots, sensitivity, etc) 
`eval_helper.py`: functions used in training and evaluation, including to read in and apply track selection to jets  
`models.py`: model architectures  
`models_archive.py`: old models that we might work on again (defunct)  
`plot_helper.py`: plotting scripts  
`root_to_numpy.py`: functions to load data from nTuples  

3) files that can potentially be deleted in future (scripts that are no longer in use)
`svj_vae.py`: train simple AE  
`evaluate.py`: evaluate simple AE (defunct)  

## Organization of the files
Input files can be found in the shared eos area: /eos/atlas/atlascerngroupdisk/phys-exotics/jdm/svjets-schannel, or check /data/users/ebusch/SVJ/autoencoder/ on katya.
There are 3 different types of input files relevant here:
0) original root file
- this is in /nevis/katya01/data/users/ebusch/SVJ/autoencoder/$NTUPLE\_VERSION e.g. /nevis/katya01/data/users/ebusch/SVJ/autoencoder/v9.2
- this can be changed when calling getTwoJetSystem() with parameter read\_dir 
- reading these root files and extracting and processing information (to be explained more in 4) usage) takes a long time so getTwoJetSystem() reads this root file and makes the file below
1) type 1 HDF5 file
- this file has a name with a format of f'{h5\_dir}/twojet/{input\_file}\_s={seed}\_ne={nevents}\_mt={max\_track}{h5tag}.hdf5' e.g. /nevis/katya01/data/users/kpark/svj-vae/h5dir/antelope/v9p2/twojet/skim0.user.ebusch.bkgAll.root\_s=0\_ne=-1\_mt=80.hdf5 where h5\_dir is '/h5dir/antelope/v9p2' 
- the variables within this name can be set with input parameters of getTwoJetSystem() along with read\_dir
- this file has been processed but it still doesn't have information of PFN and ANTELOPE scores 
2) type 2 HDF5 file
- this file has been created from running pfn\_evaluate.py and has PFN and ANTELOPE scores
- the reason why PFN and ANTELOPE score HDF5 files are separate from type 1 files is to avoid data corruption eventhough there is a downside of taking up more storage space

Now that the main input files (0 and 1) and output file (2) are explained, let's look at some folders where some other output files will be stored. Within the directory, (once you have run svj\_pfn.py or pfn\_evaluate.py files following the description in 4) usage), the following directories should be created (if parent\_dir is set as /nevis/katya01/data/users/$USERNAME/svj-vae/) :
A) /nevis/katya01/data/users/$USERNAME/svj-vae/results/grid\_sept26/$FOLDERNAME: 
	this is where most files will be created from running the main files; the most important directory($FOLDERNAME) is 10_08_23_04_08 which contains the latest version of the ANTELOPE model that was trained on data 
  a)  /nevis/katya01/data/users/$USERNAME/svj-vae/results/grid\_sept26/$FOLDERNAME/architectures\_saved
    where architectures are saved (examples of files and subdirectories vANTELOPE_decoder_arch, vANTELOPE_encoder_arch, vANTELOPE_decoder_weights.h5, vANTELOPE_encoder_weights.h5)
  b) /nevis/katya01/data/users/$USERNAME/svj-vae/results/grid\_sept26/$FOLDERNAME/applydir
    where HDF5 files with evaluated scores are saved along with relevant plots in the subdirectory of which name can be set up in pfn_evaluate.py file i.e. in, say hdf5_jet2_width subdirectory, there are a directory called plots, and HDF5 files of evaluated scores of multiple signal files. 
    
## Getting started

Create a conda enviroment from the svj\_env.yml file. Key packages are:
`h5py tensorflow keras numpy matplotlib pandas scikit-learn`

**Note**: uproot is not included in the conda environment. The Columbia group uses a local installation of uprooti (v4.3.5), which may be more efficient


## Usage

Training is done with the file `svj_pfn.py`, in svj conda environment.
```
python svj_pfn.py
```

You can save your trained network, and evaluate using the `pfn_evaluate.py` script.
```
python pfn_evaluate.py
```


