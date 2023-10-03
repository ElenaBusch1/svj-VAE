import numpy as np
from eval_helper import scale_phi, do_roc
from numpy.random import default_rng


loss_bkg=np.array([1e-4, 1e-5, 1e-3])
loss_sig=np.array([2*1e-4, 3*1e-5, 4*1e-3])
loss_both= np.concatenate((loss_bkg, loss_sig))
max_loss=np.max(loss_both)
min_loss=np.min(loss_both)
print(f'{max_loss=}, {min_loss=}, {loss_bkg[:5]}')
loss_transformed_bkg = (loss_bkg - min_loss)/(max_loss -min_loss)
loss_transformed_sig = (loss_sig - min_loss)/(max_loss -min_loss)
vae_model='test'
method='before'
method='transformed'
plot_dir='/nevis/katya01/data/users/kpark/svj-vae/test/'
step_size=1
auc=do_roc(loss_bkg, loss_sig, tag_file=vae_model+f'_{method}', tag_title=vae_model+ f' (step size={step_size} {method})',make_transformed_plot= False, plot_dir=plot_dir, bool_pfn=False)
auc_tran=do_roc(loss_transformed_bkg, loss_transformed_sig, tag_file=vae_model+f'_{method}', tag_title=vae_model+ f' (step size={step_size} {method})',make_transformed_plot= False, plot_dir=plot_dir, bool_pfn=False)
print(f'auc not transformed: {auc=}, auc transformed: {auc_tran=}')

"""
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

