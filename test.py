import numpy as np
from eval_helper import scale_phi
from numpy.random import default_rng

bkg_phi=[0,1,3,6, -5]
sig_phi=[-0.3,1,3,10, 0.5]
print(np.histogram(np.hstack((bkg_phi,sig_phi)),bins=10)[1])
sys.exit()
bkg_latent_test = default_rng(42).random((3,4,2))
sig_latent_test = default_rng(41).random((3,3,2))
bkg_latent_train= default_rng(40).random((3,5,2))
latent_test = np.concatenate((bkg_latent_test, sig_latent_test), axis=1)
y_testb = np.concatenate((np.zeros(bkg_latent_test.shape[1]), np.ones(sig_latent_test.shape[1])), axis = 0)
y_evalb_train = np.zeros(bkg_latent_train.shape[1])
print(bkg_latent_test.shape)
print('-')
print(sig_latent_test.shape)
print('-')
print(latent_test.shape)
print('-')
print(y_testb)
print(y_evalb_train)


print('-'*20)
for bool_nonzero in [True, False]:
  phis=default_rng(0).random((5))
  phis=np.concatenate((phis, np.array([0,0])), axis = 0)
  max_phi=np.max(phis)
  print(f'{phis=}')
  print(f'{max_phi=}')
  print(f'{scale_phi(phis, max_phi, bool_nonzero=bool_nonzero)=}')
  print('-'*20)
"""
import matplotlib.pyplot as plt
x=[1,2,3,3.5,2]
bins=[0.5,1.5, 2.5]
first_bin=[.5,1.5)
second_bin=[1.5, 2.5)
third_bin=[2.5,3.5]
third_bin = [3,4]
count, bins=plt.hist(x, bins=bins, label='lala')
print(count) = [1,2]
plt.legend()
#plt.show()
filename=filedir+filename
plt.savefig(filename)

"""

