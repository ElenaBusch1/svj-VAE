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

### Additional information on the main files
A) svj\_pfn.py
-read ROOT files -> randomly select events with given seed -> only select jets w/ at least 3 tracks (but no pT selections on tracks) in apply\_TrackSelection() -> in apply\_JetScalingRotation, apply rotations for eta and phi, and scaling for pT and E -> load PFN model (not trained yet) -> split train and test samples -> in apply\_StandardScaling, scale train data and with the same scaler, scale test data -> train the PFN model -> apply the model to the test samples -> make AUC/ROC curve plot, loss function plot, and a plot of PFN scores of signal vs background of test and train 
B) svj\_antelope.py 
- in prepare\_pfn(), load PFN model (trained) -> read from Type 1 HDF5 files (if already existent) -> in apply\_StandardScaling, scale background and signal samples with the scaler used in svj\_pfn.py -> if asked for signal injection, add signals to background samples -> if asked for scaling (or shift), scale (or shift) the PFN latent space variables; the standard case is that neither scaling nor shifting happens for PFN latent space variables -> in evaluate\_vae(), if ANTELOPE model hasn't been trained, call in train\_vae() -> in train\_vae(), train ANTELOPE model -> apply it to the test samples -> in plot\_loss\_dict(), make AUC/ROC curve plot and loss function plot; and a plot of ANTELOPE (a.k.a anomaly score, or in code, multi\_reco\_log10\_sig ) scores 
-- note that there is transformation of log10 and then sigmoid function, which means that if the ANTELOPE score is originally x, we want sigmoid(log10(x)); this ensures that the ANTELOPE score is restricted to \[0,1\] (due to sigmoid function) and their shapes are more visually distinguishable / less squashed (due to log10 function)
-- multi\_reco\_transformed\_log10\_sig is combination of multi\_kl\_transformed\_log10\_sig (KL = KL divergence term) and multi\_mse\_transformed\_log10\_sig (MSE = mean squared error term); it was ultimately chosen to use  multi\_reco instead only either using multi\_kl or multi\_mse for better performance 
C) pfn\_evaluate.py
- in eval\_sig(), evaluate signal files by call\_functions -> load PFN and/or ANTELOPE models -> read from Type 1 HDF5 files (if already existent) -> make plots -> scale or shift if specified -> in write\_hdf5(), create HDF5 files (type 2) with applied scores (ANTELOPE scores incl. 'mse', 'multi\_reco', 'mse\_transformed\_log10\_sig', etc) and other variables from extraVars list -> repeat in eval\_bkg for the background files to be evaluated
- in scan(), to make some signal grid plots, such as SIC, sensitivity, optimal AUC score cut, etc, run(uncomment) grid\_scan() or grid\_s\_sqrt\_b(); or run get\_sig\_contamination()
- in compare\_hist(), make data/MC plots; CR/VR/SR plots;  or other histograms all in one plot using plot\_single\_variable\_ratio() 

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
    $FOLDERNAME here is when the folder is created (specified in svj\_pfn.py and svj\_antelope.py). An example of this is 10\_08\_23\_04\_08 which was created on October 8th, 2023 at 04:08. This structure of organization is useful when comparing different hyperparmeters and going back to old records as these folders are  where most files will be created from running the main files. The most important directory($FOLDERNAME) is 10\_08\_23\_04\_08 which contains the latest version of the ANTELOPE model that was trained on data 
  a)  /nevis/katya01/data/users/$USERNAME/svj-vae/results/grid\_sept26/$FOLDERNAME/architectures\_saved
    where architectures are saved (examples of files and subdirectories vANTELOPE_decoder_arch, vANTELOPE_encoder_arch, vANTELOPE_decoder_weights.h5, vANTELOPE_encoder_weights.h5)
  b) /nevis/katya01/data/users/$USERNAME/svj-vae/results/grid\_sept26/$FOLDERNAME/applydir
    where HDF5 files with evaluated scores are saved along with relevant plots in the subdirectory of which name can be set up in pfn_evaluate.py file i.e. in, say hdf5_jet2_width subdirectory, there are a directory called plots, and HDF5 files of evaluated scores of multiple signal files. 
    
## Getting started
Three options for conda environments are given below (Tips on conda environments also listed below).

Option A:  Using GPU (recommended way):
Simply activate the environment already set up by Gabriel Matos that uses GPU (for faster computing power especially useful when training VAE model and evaluating events based on the model)
```bash
conda activate /data/users/gpm2117/envs/tf
``` 
and this is all you have to do! (to add packages, ask Gabriel Matos)

Option B: Not using GPU:
You could activate the environment already set up by K Park 
```bash
conda activate /nevis/katya01/data/users/kpark/env/ae-env
```
Option C: Or simply create your new environment by following below:

1) In the directory where the repository was cloned to, open uproot-env.yml
```bash
vim svj_env.yml
```
and inside this file, replace the line below with whichever location you want the environment to be stored in:

prefix: /nevis/katya01/data/users/kpark/env/svj\_env
 
2) Create a conda enviroment from the svj\_env.yml file. 
```bash
conda env create -f svj_env.yml
```

3) and then checking if the environment was installed by:
```bash
conda info --envs
```
4) Now, everytime you open a new terminal tab, type in
```bash
conda activate [prefix]
```
where [prefix] is the prefix you typed in step 1)
e.g. conda activate /nevis/katya01/data/users/kpark/env/uproot-env.

TIP: If you are unsure of your prefix, to retrieve prefix of the current conda environment, type in
```bash
echo $CONDA_PREFIX
```
TIP2: If there are still a few packages that are missing (which is possible), which raises errors when running the code, then try (only after activiating the conda environmnt,
```bash
conda install jupyter --channel conda-forge 
```
The channel specified here to install packages from is conda-forge which is updated often (so many packages are usually up-to-date), but other channels or default option might also work with the command such as 
```bash
conda install [package_name]
```

TIP3: 
If a long prefix shown is annoying, copy the exact command below without changing it:
```bash
conda config --set env_prompt '({name})'
```
This command will change your .condarc usually placed in a home directory (if not, it will be automatically generated from this command), which should include this line:
envi\_prompt: ({name})


More on creating an environment can be found in:
- https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
- https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/02-working-with-environments/index.html

**Note**: uproot is not included in the conda environment. The Columbia group uses a local installation of uprooti (v4.3.5), which may be more efficient


## Usage


Always activate the conda environment first before running any commands (by following one of the three options above).

### Step A
Training is done with the file `svj_pfn.py`.  
Before running the code, however, check the arguments of the init function of the class inside the code and see whether you want different values than the default ones. If you want different values, you could either change the default values or scroll near the bottom of the script and find where the class is instatiated and reset the arguments for this instance. For example, 
param1=Param(sig\_events=502000, bkg\_events=502000, learning\_rate=0.002). With your chosen settings, now, you can run 
```bash
python svj_pfn.py
```
Notice, then, the output files are all saved in self.all\_dir of the class.

### Step B
To train the ANTELOPE on the already trained PFN model, first, you can reset the arguments of svj\_antelope.py in the similar manner as specified in Step A above. Afterwards, run 
```bash
python svj_antelope.py
```
### Step C
You can either 
a) apply PFN model OR
b) apply PFN model and then apply ANTELOPE model 
using the `pfn_evaluate.py` script. Here, you also change the arguments of the instance as specificied in step A. The most important thing is change the filedir, where output files will be made (if b) then ANTELOPE model will also be used from this folder, but doesn't apply to case a). For other important parameters to check, look at the init function (e.g. extraVars, bkg\_version, pfn\_model, arch\_dir\_pfn, etc) 
Only after then, run
```
python pfn_evaluate.py
```

