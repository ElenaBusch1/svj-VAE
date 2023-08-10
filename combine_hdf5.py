import numpy as np
import h5py

all_bkg_data = []
for i in range(515487,515527):
  with h5py.File("../v8.1/v8p1_PFNv6_"+str(i)+".hdf5","r") as f:
    bkg_data = f.get('data')[:]
  all_bkg_data.append(bkg_data)

#with h5py.File("../v8.1/v8p1_PFNv3_QCDskim3_2.hdf5","r") as f:
#  bkg_data2 = f.get('data')[:]
#with h5py.File("../v8.1/v8p1_PFNv2_QCDskim0_3.hdf5","r") as f:
#  bkg_data3 = f.get('data')[:]

bkg_data = np.concatenate(all_bkg_data)
with h5py.File("v9p1_PFNv6_allSignal.hdf5","w") as h5f:
  dset = h5f.create_dataset("data",data=bkg_data)

