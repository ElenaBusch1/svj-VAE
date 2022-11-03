from tensorflow import keras

def get_simple_ae(input_dim, encoding_dim):
  input_vars = keras.Input ( shape =(input_dim,))
  encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_vars)
  decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
  autoencoder = keras.Model(input_vars, decoded)
  autoencoder.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam())
  return autoencoder

def get_better_ae(input_dim, encoding_dim):
  input_vars = keras.Input ( shape =(input_dim,))
  encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_vars)
  encoded = keras.layers.Dense(encoding_dim/2, activation='relu')(encoded)
  encoded = keras.layers.Dense(encoding_dim/4, activation='relu')(encoded)

  decoded = keras.layers.Dense(encoding_dim/2, activation='sigmoid')(encoded)
  decoded = keras.layers.Dense(encoding_dim, activation='sigmoid')(decoded)
  decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
  autoencoder = keras.Model(input_vars, decoded)
  autoencoder.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam())
  return autoencoder


def get_vae(input_dim, encoding_dim):
  #TODO: https://keras.io/examples/generative/vae/
  return
