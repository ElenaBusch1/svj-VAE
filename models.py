import tensorflow as tf
from tensorflow import keras
from keras import backend as K

def sampling(args):
  z_mean, z_log_var = args
  #batch = K.shape(z_mean)[0]
  #latent_dim = K.shape(z_mean)[1]
  #epsilon = K.random_normal(shape=(batch,latent_dim), mean=0., stddev=0.1)
  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2), mean=0., stddev=0.1)
  return z_mean + K.exp(z_log_var) * epsilon

def get_simple_ae(input_dim, encoding_dim):
  input_vars = keras.Input ( shape =(input_dim,))
  encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_vars)
  decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
  autoencoder = keras.Model(input_vars, decoded)
  autoencoder.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam())
  return autoencoder

def get_better_ae(input_dim, encoding_dim):
  input_vars = keras.Input ( shape =(input_dim,))
  encoded = keras.layers.Dense(encoding_dim, activation='sigmoid')(input_vars)
  encoded = keras.layers.Dense(encoding_dim/4, activation='sigmoid')(encoded)
  print("in get batter ae")
  decoded = keras.layers.Dense(encoding_dim, activation='sigmoid')(encoded)
  decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
  autoencoder = keras.Model(input_vars, decoded)
  autoencoder.summary()
  autoencoder.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam(learning_rate=0.01))
  return autoencoder

class VAE(keras.Model):
    #direct copy from https://keras.io/examples/generative/vae/
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #z_mean, z_log_var, reconstruction = call
            reconstruction_loss = tf.reduce_mean(
                #tf.reduce_sum(
                    keras.losses.mse(data, reconstruction)#, axis=(1, 2)
                #)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
          data = data[0]
    
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(data, reconstruction)
        )
        #reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, data):
        z_mean,z_log_var,x = self.encoder(data)
        reconstruction = self.decoder(x)
        return {
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction
        }

class PFN_VAE(keras.Model):
    #think about modifying this so it inherits from VAE class
    def __init__(self, pfn, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.pfn = pfn
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.phi_tracker = keras.metrics.Mean(name="sum_phi")

    @property
    def metrics(self):
        return [
            self.phi_tracker,
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            phi = self.pfn(data)
            z_mean, z_log_var, z = self.encoder(phi)
            reconstruction = self.decoder(z)
            #z_mean, z_log_var, reconstruction = call
            reconstruction_loss = tf.reduce_mean(
                #tf.reduce_sum(
                    keras.losses.mse(phi, reconstruction)#, axis=(1, 2)
                #)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            sum_phi = tf.reduce_mean(phi)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.phi_tracker.update_state(sum_phi)
        return {
            "sum_phi": self.phi_tracker.result(),
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        #if isinstance(data, tuple):
        #  data = data[0]
    
        phi = self.pfn(data)
        z_mean, z_log_var, z = self.encoder(phi)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(phi, reconstruction)
        )
        #reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        sum_phi =  tf.reduce_mean(phi)
        return {
            "sum_phi": sum_phi,
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, data):
        phi = self.pfn(data)
        z_mean,z_log_var,x = self.encoder(phi)
        reconstruction = self.decoder(x)
        return {
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction
        }


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class PFN_AE(keras.Model):
    #think about modifying this so it inherits from VAE class
    def __init__(self, pfn, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.pfn = pfn
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            phi = self.pfn(data)
            encoded = self.encoder(phi)
            reconstruction = self.decoder(encoded)
            #z_mean, z_log_var, reconstruction = call
            reconstruction_loss = tf.reduce_mean(
                #tf.reduce_sum(
                    keras.losses.mse(phi, reconstruction)#, axis=(1, 2)
                #)
            )
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

    def test_step(self, data):
        #if isinstance(data, tuple):
        #  data = data[0]
    
        phi = self.pfn(data)
        encoded = self.encoder(phi)
        reconstruction = self.decoder(encoded)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(phi, reconstruction)
        )
        return {
            "reconstruction_loss": reconstruction_loss,
        }

    def call(self, data):
        phi = self.pfn(data)
        encoded = self.encoder(phi)
        reconstruction = self.decoder(encoded)
        return {
            "reconstruction": reconstruction
        }


def get_vae2(input_dim, encoding_dim):
  latent_dim = 2

  encoder_inputs = keras.Input(shape=(input_dim,))
  x = keras.layers.Dense(encoding_dim, activation="relu")(encoder_inputs)
  z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  encoder.summary()

  latent_inputs = keras.Input(shape=(latent_dim,))
  x = keras.layers.Dense(encoding_dim, activation="sigmoid")(latent_inputs)
  decoder_outputs = keras.layers.Dense(input_dim, activation="sigmoid")(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  decoder.summary()

  vae = VAE(encoder, decoder)
  vae.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0005))

  return vae

def pfn_mask_func(X, mask_val=0):
  # map mask_val to zero and return 1 elsewhere
  return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))


def get_pfn(input_dim, phi_dim):
  initializer = keras.initializers.HeNormal()
  loss = keras.losses.CategoricalCrossentropy()
  optimizer = keras.optimizers.Adam() 
 
  input_dim_x = input_dim[0]
  input_dim_y = input_dim[1]

  #input
  pfn_inputs = keras.Input(shape=(None,input_dim_y))
  masked = keras.layers.Lambda(pfn_mask_func, name="mask")(pfn_inputs)

  # Phi network
  dense1 = keras.layers.Dense(100, kernel_initializer=initializer, name="pfn1")
  x = keras.layers.TimeDistributed(dense1, name="tdist_0")(pfn_inputs)
  x = keras.layers.Activation('relu')(x)
  dense2 = keras.layers.Dense(100, kernel_initializer=initializer, name="pfn2") 
  x = keras.layers.TimeDistributed(dense2, name="tdist_1")(x)
  x = keras.layers.Activation('relu')(x)
  dense3 = keras.layers.Dense(phi_dim, kernel_initializer=initializer, name="phi") 
  x = keras.layers.TimeDistributed(dense3, name="tdist_2")(x)
  phi_outputs = keras.layers.Activation('relu')(x)

  # latent space
  sum_phi = keras.layers.Dot(1, name="sum")([masked,phi_outputs])
  
  pfn = keras.Model(pfn_inputs, sum_phi, name="pfn")
  pfn.summary()
  return pfn

def get_encoder(input_dim, encoding_dim):
  latent_dim = 2
  inputs = keras.Input(shape=(input_dim,))
  x = keras.layers.Dense(encoding_dim, activation="relu")(inputs)
  z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  
  encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
  encoder.summary()
  return encoder

def get_decoder(input_dim, encoding_dim):
  latent_dim = 2
  latent_inputs = keras.Input(shape=(latent_dim,))
  x = keras.layers.Dense(encoding_dim, activation="sigmoid")(latent_inputs)
  decoder_outputs = keras.layers.Dense(input_dim, activation="sigmoid")(x)

  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  decoder.summary()
  return decoder

def get_pfn_vae(input_dim, phi_dim, encoding_dim):
  pfn = get_pfn(input_dim, phi_dim)
  encoder = get_encoder(phi_dim, encoding_dim)
  decoder = get_decoder(phi_dim, encoding_dim)

  pfn_vae = PFN_VAE(pfn, encoder, decoder)
  pfn_vae.compile(optimizer=keras.optimizers.Adam())

  return pfn_vae, pfn

def get_pfn_ae(input_dim, phi_dim, encoding_dim):
  encoding_dim_1 = encoding_dim[0]
  encoding_dim_2 = encoding_dim[1]

  pfn = get_pfn(input_dim, phi_dim)

  #encoder
  encoder_inputs = keras.Input(shape=(phi_dim,))
  encoded = keras.layers.Dense(encoding_dim_1, activation='sigmoid')(encoder_inputs)
  encoded = keras.layers.Dense(encoding_dim_2, activation='sigmoid')(encoded)
  encoder = keras.Model(encoder_inputs, encoded, name="encoder")
  encoder.summary()
  
  #decoder
  decoder_inputs = keras.Input(shape=(encoding_dim_2,))
  decoded = keras.layers.Dense(encoding_dim_1, activation='sigmoid')(decoder_inputs)
  decoded = keras.layers.Dense(phi_dim, activation='sigmoid')(decoded)
  decoder = keras.Model(decoder_inputs, decoded, name="decoder")
  decoder.summary()

  pfn_ae = PFN_AE(pfn, encoder, decoder)
  pfn_ae.compile(optimizer=keras.optimizers.Adam())

  return pfn_ae, pfn

