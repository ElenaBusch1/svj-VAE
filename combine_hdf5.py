import numpy as np
import h5py

##  # bkg file list
##  with open("../v12.5/v12p5_bkg_hdf5s.txt", "r") as f:
##    files = []
##    for line in f:
##      line = line.strip()
##      files.append(line)
##  
##  all_bkg_data = []
##  for f in files:
##    with h5py.File("../v12.5/"+f,"r") as f:
##      bkg_data = f.get('data')[:]
##    all_bkg_data.append(bkg_data)
##  
##  bkg_data = np.concatenate(all_bkg_data)
##  with h5py.File("v12p5_PFNv12_bkgALL.hdf5","w") as h5f:
##    dset = h5f.create_dataset("data",data=bkg_data)

##  # signals
##  for dsid in range(515495,515523):
##    all_sig_data = []
##    for mc in ["mc20a", "mc20d", "mc20e"]:
##      try:
##        with h5py.File("../v12.5/v12p5_PFNv12_"+str(dsid)+"."+mc+".hdf5","r") as f:
##          sig_data = f.get('data')[:]
##        all_sig_data.append(sig_data)
##      except:
##        continue
##    if len(all_sig_data) != 3: continue
##  
##    rec_data = np.concatenate(all_sig_data)
##    print("Recording data for "+str(dsid))
##    with h5py.File("../v12.5/v12p5_PFNv12_"+str(dsid)+".hdf5","w") as h5f:
##      dset = h5f.create_dataset("data",data=rec_data)
    
# data
for yr in range(15,19):
  all_data_data = []
  with h5py.File("../v12.5/v12p5_PFNv12_data"+str(yr)+".hdf5","r") as f:
    data_data = f.get('data')[:]
  all_data_data.append(data_data)

rec_data = np.concatenate(all_data_data)
with h5py.File("../v12.5/v12p5_PFNv12_dataALL.hdf5","w") as h5f:
  dset = h5f.create_dataset("data",data=rec_data)

