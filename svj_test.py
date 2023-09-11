  def prepare_test(self, bool_flat=False):
    # apply scaling # is it already applied
    # plot vectors
# just the VAE
# choose validation and train
# vae fitting
# vae save
# reconstruct
    (x_evalb, y_evalb), (x_testb, y_testb) = tf.keras.datasets.fashion_mnist.load_data()

    # shuffle before flattening!
    x_evalb, y_evalb=self.shuffle_two(x_evalb, y_evalb)
    x_testb, y_testb=self.shuffle_two(x_testb, y_testb)
    cprint(f'before{x_evalb.shape=}', 'yellow')
    cprint(f'before{type(x_evalb[0])}', 'yellow')
    cprint(f'before{type(y_evalb[0])}', 'yellow')
    print(y_evalb, y_testb)

    #flatten
    if bool_flat:
      x_evalb, x_testb= self.sample_flatten(x_evalb), self.sample_flatten(x_testb)

    else:   x_evalb, x_testb= x_evalb.reshape(x_evalb.shape[0], 28,28, 1), x_testb.reshape(x_testb.shape[0], 28,28, 1)
    x_evalb, x_testb= x_evalb.astype('float32'), x_testb.astype('float32')
    y_evalb, y_testb= y_evalb.astype('float32'), y_testb.astype('float32')
    print(x_evalb.shape, x_testb.shape)
    #x_evalb = x_evalb.reshape(x_evalb.shape[0], 28, 28, 1).astype('float32')
    cprint(f'after{type(x_evalb[0])}', 'yellow')
    cprint(f'after{type(y_evalb[0])}', 'yellow')


    # plot test input
    # select 15 samples since there are 10000
    nsample=9
    x_testb_plt= x_testb[:nsample]
    # reshape
    if bool_flat:
      x_testb_plt=x_testb.reshape(x_testb.shape[0], 28,28, 1) # should be (x, 28, 28,1)
    cprint(f'reshape {x_testb.shape=}')
    fig = plt.figure()

    for i in range(nsample):
      ax = fig.add_subplot(3, 3, i+1)
      ax.axis('off')
#      ax.text(0.5, -0.15, str(label_dict[y_test[i]]), fontsize=10, ha='center', transform=ax.transAxes)
                                                                                                           

      ax.imshow(x_testb_plt[i, :,:,0]*255, cmap = 'gray')


    fig.suptitle('Input Image')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'input.png')
#    plt.show()
    plt.clf()

    print(f'after{x_evalb.shape=}', 'yellow')

    #x_testb = x_testb.astype('float32')
#    cprint(f'after{(x_evalb[0])}', 'yellow')
    print(f'{np.max(x_evalb)=}')
    x_evalb = x_evalb / 255.
    x_testb = x_testb / 255.
    # validation data set manually
    # Prepare the training dataset
    idx = np.random.choice( np.arange(len(x_evalb)), size= round(.2 *len(x_evalb)) , replace=False) # IMPT that replace=False so that event is picked only once
    idx = np.sort(idx)
    # Prepare the validation dataset
    x_evalb_val = x_evalb[idx, :]
    x_evalb_train = np.delete(x_evalb, idx) # doesn't modify input array 
    print(f'{x_evalb_val.shape=}, {x_evalb_train=}')

    x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val = train_test_split(x_evalb, y_evalb, test_size=round(.2 *len(x_evalb)))
    #phi_evalb_train, phi_evalb_val, _, _ = train_testb_split(phi_evalb, phi_evalb, testb_size=round(.2 *len(phi_evalb)))
    return  x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val


  def evaluate_test(self):
    test_labels = y_evalb.astype(bool)
    normal_test = x_evalb[test_labels]
    anomalous_test = x_evalb[~test_labels]
    x_testb, x_evalb_train, x_evalb_val, y_testb, y_evalb_train, y_evalb_val= self.prepare_test(bool_flat=True)
    print('prepare_test')

    try: vae = self.load_vae()
    except:
      print('loading vae not successful so will start the training process')
      vae,h2 = self.train_vae( x_evalb_train, x_evalb_val, y_evalb_train, y_evalb_val)
      print('training successful')

    latent_test=vae.get_layer('encoder').predict(x_testb)
    latent_train=vae.get_layer('encoder').predict(x_evalb_train)
    latent_val=vae.get_layer('encoder').predict(x_evalb_val)


    #latent_test is a list but latent_test[0] is a numpy array
    latent_test, latent_train, latent_val=np.array(latent_test), np.array(latent_train), np.array(latent_val)
    print(f'{latent_test.shape=}')

    print(f'{y_testb=}{y_testb.shape}')
    plot_pca(latent_test[0,:,:], latent_label=np.array(y_testb), nlabel=10,n_components=2, tag_file=self.vae_model+'_test', tag_title=self.vae_model+' Test', plot_dir=self.plot_dir)
    plot_pca(latent_train[0,:,:],latent_label=np.array(y_evalb_train), nlabel=10, n_components=2, tag_file=self.vae_model+'_train', tag_title=self.vae_model+' Train', plot_dir=self.plot_dir)

    # reconstruct output
#    print(f'{latent_test.shape=}')
    latent_test_recon = vae.get_layer('decoder').predict(latent_test[2,:,:])
    print(f'{latent_test_recon.shape=}')
    # select 15 samples since there are 10000
    nsample=9
    latent_test_recon= latent_test_recon[:nsample]
    # reshape
    latent_test_recon=latent_test_recon.reshape(latent_test_recon.shape[0], 28, -1) # should be (x, 28, 28)
    cprint(f'reshape 1 {latent_test_recon.shape=}')
    latent_test_recon=latent_test_recon.reshape(latent_test_recon.shape[0], 28,28, 1) # should be (x, 28, 28,1)
    cprint(f'reshape 2 {latent_test_recon.shape=}')
    fig = plt.figure()

    for i in range(nsample):
      ax = fig.add_subplot(3, 3, i+1)
      ax.axis('off')
#      ax.text(0.5, -0.15, str(label_dict[y_test[i]]), fontsize=10, ha='center', transform=ax.transAxes)

      ax.imshow(latent_test_recon[i, :,:,0]*255, cmap = 'gray')
    #ax.imshow(latent_test_recon[i, :,:,0]*255, cmap = 'gray') 
    fig.suptitle('Reconstructed Image')
    plt.tight_layout()
    plt.savefig(self.plot_dir + f'output.png')
#    plt.show()
    plt.clf()

    latent_test_sigma, latent_train_sigma = self.transform_sigma(latent_test[1,:,:]), self.transform_sigma(latent_train[1, :,:])

#    for k in range(len(latent_test)):
    plot_1D_phi(latent_test[0,:,:],latent_train[0,:,:] , labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_mu', tag_title=self.vae_model +r" $\mu$", ylog=True)
    plot_1D_phi(latent_test_sigma, latent_train_sigma, labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_sigma', tag_title=self.vae_model  +r" $\sigma$", ylog=True)

    plot_1D_phi(latent_test[0,:,:], latent_train[0, :,:], labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_mu_custom', tag_title=self.vae_model +r" $\mu$", bins=np.linspace(-0.0001,0.0001, num=50))
    plot_1D_phi(latent_test_sigma,latent_train_sigma, labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_sigma_custom', tag_title=self.vae_model +r" $\sigma$", bins=np.linspace(0.9998,1.0002,num=50))

    plot_1D_phi(latent_test[2,:,:], latent_train[2, :,:], labels=['test', 'train'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'test_train_sampling', tag_title=self.vae_model +" Sampling",bool_norm=True)

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch 
    try:plot_loss(h2, loss='loss', tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
    except: print('loading vae_model so cannot draw regular loss plot -> no h2')
    #plot_loss(h2, vae_model, "kl_loss")
    #plot_loss(h2, vae_model, "reco_loss")

    #2. Get loss
#    """
    pred_x_test = vae.predict(x_testb)['reconstruction']
    bkg_loss_mse = keras.losses.mse(x_testb, pred_x_test)

#    plot_score(bkg_loss_mse, np.array([]), False, True, tag_file=self.vae_model+'_single_loss', tag_title=self.vae_model+' (MSE)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
#    """
    #start = time.time()
    #step_size=self.batchsize_vae
    bkg_loss,  bkg_kl_loss,  bkg_reco_loss  = get_multi_loss_each(vae, x_testb, step_size=self.step_size)
    #end = time.time()
#    print("Elapsed (with get_multi_loss) = %s" % (end - start))
#    try:cprint(f'{min(bkg_loss)}, {min(sig_loss)}, {max(bkg_loss)}, {max(sig_loss)}', 'yellow')
#    except: cprint(f'{np.min(bkg_loss)}, {np.min(sig_loss)},{np.max(bkg_loss)}, {np.max(sig_loss)}', 'blue')
    print('\n')

    # xlog=True plots
    sig_loss, sig_kl_loss, sig_reco_loss=np.array([]), np.array([]), np.array([])
    plot_score(bkg_loss[bkg_loss>0], sig_loss[sig_loss>0], False, True, tag_file=self.vae_model+'_pos', tag_title=self.vae_model + ' (score > 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], False, True, tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss>0], sig_reco_loss[sig_reco_loss>0], False, True, tag_file=self.vae_model+"_MSE_pos", tag_title=self.vae_model+" MSE (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog=False plots
    plot_score(bkg_loss, sig_loss, False, False, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog= False plots plot only points less than 0 
    plot_score(bkg_loss[bkg_loss<=0], sig_loss[sig_loss<=0], False, False, tag_file=self.vae_model+'_neg', tag_title=self.vae_model+' (score <= 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss<=0], sig_kl_loss[sig_kl_loss<=0], False, False, tag_file=self.vae_model+"_KLD_neg", tag_title=self.vae_model+" KLD (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss<=0], sig_reco_loss[sig_reco_loss<=0], False, False, tag_file=self.vae_model+"_MSE_neg", tag_title=self.vae_model+" MSE (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    # # 3. Signal Sensitivity Score

    auc= {np.nan, np.nan, np.nan}
    bkg_events_num,sig_events_num=np.nan, np.nan


    return self.all_dir, auc, bkg_events_num,sig_events_num

  def evaluate_vae(self):
    graph, scaler = self.load_pfn()
    phi_bkg,phi_testb, phi_evalb_train, phi_evalb_val, phi_sig=  self.prepare_pfn(graph,scaler)
    print('prepare_pfn')

    try: vae = self.load_vae()
    except:
      print('loading vae not successful so will start the training process')
      vae,h2 = self.train_vae( phi_evalb_train, phi_evalb_val)
      print('training successful')

    #complex ae
    #with open(arch_dir+vae_model+"8.1_predstory.json", "w") as f:
    #    json.dump(h2.predstory, f)
    latent_bkg_test=vae.get_layer('encoder').predict(x_testb)
    latent_bkg_train=vae.get_layer('encoder').predict(x_evalb_train)
    latent_bkg_val=vae.get_layer('encoder').predict(x_evalb_val)

    #latent_bkg_test is a list but latent_bkg_test[0] is a numpy array
    latent_bkg_test, latent_bkg_train, latent_bkg_val, latent_sig=np.array(latent_bkg_test), np.array(latent_bkg_train), np.array(latent_bkg_val), np.array(latent_sig)
    print(f'{latent_bkg_test.shape=}')
    latent_bkg_test_sigma, latent_sig_sigma = self.transform_sigma(latent_bkg_test[1,:,:]), self.transform_sigma(latent_sig[1, :,:])


#    for k in range(len(latent_bkg_test)):
    plot_1D_phi(latent_bkg_test[0,:,:], latent_sig[0, :,:], labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_mu', tag_title=self.vae_model +r" $\mu$", ylog=True)
    plot_1D_phi(latent_bkg_test_sigma, latent_sig_sigma, labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_sigma', tag_title=self.vae_model  +r" $\sigma$", ylog=True)

    plot_1D_phi(latent_bkg_test[0,:,:], latent_sig[0, :,:], labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_mu_custom', tag_title=self.vae_model +r" $\mu$", bins=np.linspace(-0.0001,0.0001, num=50))
    plot_1D_phi(latent_bkg_test_sigma,latent_sig_sigma, labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_sigma_custom', tag_title=self.vae_model +r" $\sigma$", bins=np.linspace(0.9998,1.0002,num=50))

    plot_1D_phi(latent_bkg_test[2,:,:], latent_sig[2, :,:], labels=['test QCD', 'SIG'], plot_dir=self.plot_dir, tag_file=self.vae_model+f'qcd_sig_sampling', tag_title=self.vae_model +" Sampling",bool_norm=True)

    ######## EVALUATE SUPERVISED ######
    # # --- Eval plots 
    # 1. Loss vs. epoch 
    try:plot_loss(h2, loss='loss', tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir)
    except: print('loading vae_model so cannot draw regular loss plot -> no h2')
    #plot_loss(h2, vae_model, "kl_loss")
    #plot_loss(h2, vae_model, "reco_loss")

    #2. Get loss
    #bkg_loss, sig_loss = get_single_loss(ae, phi_testb, phi_sig)
#    """
    pred_phi_bkg = vae.predict(phi_testb)['reconstruction']
    pred_phi_sig = vae.predict(phi_sig)['reconstruction']
    bkg_loss_mse = keras.losses.mse(phi_testb, pred_phi_bkg)
    sig_loss_mse = keras.losses.mse(phi_sig, pred_phi_sig)

    plot_score(bkg_loss_mse, sig_loss_mse, False, True, tag_file=self.vae_model+'_single_loss', tag_title=self.vae_model+' (MSE)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    #start = time.time()
#    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig)
    bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig, step_size=self.step_size)
    #end = time.time()
#    print("Elapsed (with get_multi_loss) = %s" % (end - start))
    try:cprint(f'{min(bkg_loss)}, {min(sig_loss)}, {max(bkg_loss)}, {max(sig_loss)}', 'yellow')
    except: cprint(f'{np.min(bkg_loss)}, {np.min(sig_loss)},{np.max(bkg_loss)}, {np.max(sig_loss)}', 'blue')
    print('\n')

    # xlog=True plots 
    plot_score(bkg_loss[bkg_loss>0], sig_loss[sig_loss>0], False, True, tag_file=self.vae_model+'_pos', tag_title=self.vae_model + ' (score > 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], False, True, tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss>0], sig_reco_loss[sig_reco_loss>0], False, True, tag_file=self.vae_model+"_MSE_pos", tag_title=self.vae_model+" MSE (score > 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog=False plots
    plot_score(bkg_loss, sig_loss, False, False, tag_file=self.vae_model, tag_title=self.vae_model, plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss, sig_kl_loss, False, False, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss, sig_reco_loss, False, False, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score

    # xlog= False plots plot only points less than 0 
    plot_score(bkg_loss[bkg_loss<=0], sig_loss[sig_loss<=0], False, False, tag_file=self.vae_model+'_neg', tag_title=self.vae_model+' (score <= 0)', plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_kl_loss[bkg_kl_loss<=0], sig_kl_loss[sig_kl_loss<=0], False, False, tag_file=self.vae_model+"_KLD_neg", tag_title=self.vae_model+" KLD (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    plot_score(bkg_reco_loss[bkg_reco_loss<=0], sig_reco_loss[sig_reco_loss<=0], False, False, tag_file=self.vae_model+"_MSE_neg", tag_title=self.vae_model+" MSE (score <= 0)", plot_dir=self.plot_dir, bool_pfn=False) # anomaly score
    # # 3. Signal Sensitivity Score

    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    sic_vals=do_roc(bkg_loss, sig_loss, tag_file=self.vae_model, tag_title=self.vae_model+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss, sig_kl_loss, tag_file=self.vae_model+"_KLD", tag_title=self.vae_model+" KLD"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_kl=do_roc(bkg_kl_loss[bkg_kl_loss>0], sig_kl_loss[sig_kl_loss>0], tag_file=self.vae_model+"_KLD_pos", tag_title=self.vae_model+" KLD (score > 0)"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    sic_vals_reco=do_roc(bkg_reco_loss, sig_reco_loss, tag_file=self.vae_model+"_MSE", tag_title=self.vae_model+" MSE"+ f'(batch size = {self.batchsize_vae}',make_transformed_plot= False, plot_dir=self.plot_dir, bool_pfn=False)
    auc={sic_vals['auc'], sic_vals_kl['auc'], sic_vals_reco['auc']}
    bkg_events_num,sig_events_num=len(phi_bkg), len(phi_sig)

    # LOG score
    print("Taking log of score...")

    bkg_loss_mse = np.log(bkg_loss_mse)
    sig_loss_mse = np.log(sig_loss_mse)
    score = getSignalSensitivityScore(bkg_loss_mse, sig_loss_mse)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    do_roc(bkg_loss_mse, sig_loss_mse, tag_file=self.vae_model+'_log_MSE', tag_title=self.vae_model+'log (MSE)',make_transformed_plot= True, plot_dir=self.plot_dir,  bool_pfn=False)
    return self.all_dir, auc, bkg_events_num,sig_events_num

                                                                                                                                                              
