import numpy as np
from tensorflow import keras
from sklearn.metrics import roc_auc_score, roc_curve
from plot_helper import plot_loss

#---- REFERENCES 
#- Keras tutorials: https://blog.keras.io/building-autoencoders-in-keras.html
#- https://towardsdatascience.com (for any topic)
#- VRNN code: https://gitlab.cern.ch/dawillia/cxaod-scripts/-/tree/master/train
#- Weakly supervised CWoLa with PFNs: https://github.com/juliagonski/ILCAnomalies

# params
input_dim = 12 #start with N HLVs (from KP's BDT)
encoding_dim = 2
nepochs = 50
batchsize = 16

# model 
input_vars = keras.Input ( shape =(input_dim,))
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_vars)
decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = keras.Model(input_vars, decoded)
autoencoder.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam())

# prepare input events
x = np.random.rand(100, input_dim) #TODO: this is a dummy 100 events modeled by 12 vars, but need a function to pull these from JZW dijet
sig = np.random.rand(100, input_dim) #TODO same function but for SVJ vars
print(x)
print(sig)
x_temp, x_test, _, _ = train_test_split(x, x, test_size=0.05)
x_train, x_valid, _, _ = train_test_split(x_temp,
                                          x_temp,
                                          test_size=0.1)
n_train = len(x_train)
n_valid = len(x_valid)
n_test = len(x_test)
print("Length train :", n_train, ", valid: ", n_valid, ", test: ", n_test)

# train
h = autoencoder.fit(x_train, x_train,
                epochs=nepochs,
                batch_size=batchsize,
                shuffle=True,
                validation_data=(x_valid, x_valid))
# Define anomaly score as loss TODO
#test_data = autoencoder.evaluate(x_test)
#test_signal = autoencoder.evaluate(sig)

# - First eval plots 
# 1. Loss vs. epoch 
plot_loss(h)
# 2. Histogram of reco error (loss) for JZW and evaled SVJ signals (test sets)
# 3. ROCs/AUCs using sklearn functions imported above  
