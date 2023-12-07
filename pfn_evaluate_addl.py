#dsids= ['515502', '515499']
dsids= [ '515499', '515502', '515515', '515518']
from helper import Label
#keys=['multi_reco']
keys=['mse', 'multi_reco', 'multi_kl', 'multi_mse']
for method_scale in keys:
#  hists=[]
  hists_var={}
#  var='jet2_Width'
#  var='jet1_pt'
  h_names=[]
#  method=method_scale
  method=f'{method_scale}_transformed_log10_sig'
  var_ls=[method, 'mT_jj', 'jet2_Width']
  for var in var_ls: hists_var[var]= []
  weight_ls=[]
  #method='multi_reco_transformed'
#  bkgpath=applydir+f"{bkg_file_prefix}dataALL_log10.hdf5"
  bkgpath=applydir+'/'+'hdf5_jet2_width'+'/'+f"{bkg_file_prefix}dataALL_log10.hdf5"
  dsid=bkgpath.split('.')[-2].split('_')[-2]
  #bkgpath=applydir+f"{bkg_file_prefix}QCDskim_log10.hdf5"
  '''
  with h5py.File(bkgpath,"r") as f:
    bkg_data = f.get('data')[:]
  
  loss_fixed=bkg_data[method]
  hists.append(loss_fixed)
  hists_var.append(bkg_data[var])
  if bkg_data['weight'].any(): # if array contains some element other than 0 
    weight_ls.append(bkg_data['weight'])
  else:#if array contains only zeros
    print(np.ones(bkg_data['weight'].shape).shape)
    print(bkg_data['weight'].shape)
    weight_ls.append(np.ones(bkg_data['weight'].shape))
  
  h_names.append(f'dataALL ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}]) ')
  #h_names.append(f'QCD ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}]) ')
  #h_names.append(f'QCD (s.w. test: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}]) ')
  '''
  '''
  for dsid in dsids:
    sigpath=applydir+f"{sig_file_prefix}{dsid}_log10"+".hdf5"
      # sigpath="../v8.1/"+sig_file_prefix+str(dsid)+".hdf5"
    with h5py.File(sigpath,"r") as f:
      sig1_data = f.get('data')[:]
    mass=Label(dsid).get_m(bool_num=True)
    print(mass)
    rinv=Label(dsid).get_rinv(bool_num=True)
  #  loss= np.log(sig1_data[method_scale])
    loss_fixed=sig1_data[method]
    hists.apend(loss_fixed)
    weight_ls.append(sig1_data['weight'])
    #mass = dsid_mass[dsid]
    h_names.append(f'{mass} GeV {rinv} ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}])')
    #h_names.append(f'{mass} GeV {rinv} (s.w. test: [{round(np.min(loss_fixed),1)}, {round(np.max(loss_fixed),1)}])')
  '''
  bkgpath=applydir+'/'+'hdf5_jet2_width'+'/'+f"{bkg_file_prefix}bkgAll_log10_0-67_jet2_width.hdf5"
  #bkgpath=applydir+'/'+'hdf5_jet2_width'+'/'+f"{bkg_file_prefix}bkgAll_log10_0-67.hdf5"
  #bkgpath=applydir+f"{bkg_file_prefix}bkgAll_log10_0-46.hdf5"
  #bkgpath=applydir+f"{bkg_file_prefix}bkgAll_log10.hdf5"
  dsid=bkgpath.split('.')[-2].split('_')[-2]
  #bkgpath=applydir+f"{bkg_file_prefix}QCDskim_log10.hdf5"
  with h5py.File(bkgpath,"r") as f:
    bkg_data = f.get('data')[:]

  cprint(bkg_data.dtype.names, 'yellow')  
  loss_fixed=bkg_data[method]
  for var in var_ls:
    hists_var[var].append(bkg_data[var])
  if bkg_data['weight'].any(): # if array contains some element other than 0 
    weight_ls.append(bkg_data['weight'])
  else:#if array contains only zeros
    print(np.ones(bkg_data['weight'].shape).shape)
    print(bkg_data['weight'].shape)
    weight_ls.append(np.ones(bkg_data['weight'].shape))
  
  h_names.append(f'bkgALL ( log + sigmoid: [{round(np.min(loss_fixed),3)}, {round(np.max(loss_fixed),3)}]) ')
 # plot_single_variable_ratio(hists,h_names=h_names,weights_ls=weight_ls,plot_dir=plot_dir,logy=True, title= f'{method}_comparison', bool_ratio=False)
#  plot_single_variable_ratio(hists,h_names=h_names,weights_ls=weight_ls,plot_dir=plot_dir,logy=True, title= f'{method}_comparison', bool_ratio=True)
  print(weight_ls[-1]) 
  plot_single_variable_ratio([hists_var['mT_jj'][-1],hists_var['mT_jj'][-1],hists_var['mT_jj'][-1], hists_var['mT_jj'][-1]],h_names=['(All)','(CR)', '(VR)', ' (SR)'],weights_ls=[weight_ls[-1],weight_ls[-1], weight_ls[-1], weight_ls[-1]],plot_dir=plot_dir,logy=True, title= f'mT_jj_{method}_comparison_region', bool_ratio=True, hists_cut=[[hists_var['jet2_Width'][-1], hists_var[method][-1]],[hists_var['jet2_Width'][-1], hists_var[method][-1]],[ hists_var['jet2_Width'][-1], hists_var[method][-1]],[ hists_var['jet2_Width'][-1], hists_var[method][-1]]],cut_ls=[[0,0],[0.05, 0], [0.05, 0.7],[0.05, 0.7]], cut_operator = [[True, True],[False,True], [True,  False],[True, True]] , method_cut=[['jet2_Width', method],['jet2_Width', method], ['jet2_Width', method], ['jet2_Width', method]], bin_min=1000, bin_max= 5000)
  #plot_single_variable_ratio([hists_var['mT_jj'][-1],hists_var['mT_jj'][-1], hists_var['mT_jj'][-1]],h_names=['(CR)', '(VR)', ' (SR)'],weights_ls=[weight_ls[-1], weight_ls[-1], weight_ls[-1]],plot_dir=plot_dir,logy=True, title= f'mT_jj_{method}_comparison_region', bool_ratio=True, hists_cut=[[hists_var['jet2_Width'][-1], hists_var[method][-1]],[ hists_var['jet2_Width'][-1], hists_var[method][-1]],[ hists_var['jet2_Width'][-1], hists_var[method][-1]]],cut_ls=[[0.05, 0], [0.05, 0.7],[0.05, 0.7]], cut_operator = [[False,True], [True,  False],[True, True]] , method_cut=[['jet2_Width', method], ['jet2_Width', method], ['jet2_Width', method]], bin_min=0, bin_max= 5000)
#  plot_single_variable_ratio([hists_var[-1],hists_var[-1]],h_names=['bkgALL', 'bkgALL'],weights_ls=[weight_ls[-1], weight_ls[-1]],plot_dir=plot_dir,logy=True, title= f'{var}_{method}_comparison', bool_ratio=True, hists_cut=[hists[-1], hists[-1]],cut_ls=[0.7,0.7], cut_operator = [True, False], method_cut=method, bin_min=1000, bin_max=5000)

