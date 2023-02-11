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

def get_vae(input_dim, encoding_dim):
  # Source 1: https://blog.keras.io/building-autoencoders-in-keras.html <- Primary resource
  # Source 2: https://learnopencv.com/variational-autoencoder-in-tensorflow/
  # Source 3: https://keras.io/examples/generative/vae/

  latent_dim = 2

  # encoding
  input_vars = keras.Input ( shape =(input_dim,))
  encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_vars)

  #sampling layer
  z_mean = keras.layers.Dense(latent_dim, name="z_mean")(encoded)
  z_log_sigma = keras.layers.Dense(latent_dim, name="z_log_sigma")(encoded)
  z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
  encoder_model = keras.Model(input_vars, [z_mean, z_log_sigma, z], name='Encoder')
  encoder_model.summary()

  #decoding
  latent_inputs = keras.Input(shape=(latent_dim,), name = 'z_sampling')
  decoded = keras.layers.Dense(encoding_dim, activation='relu')(latent_inputs)
  decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
  decoder_model = keras.Model(latent_inputs, decoded, name='Decoder')
  decoder_model.summary()

  #vae
  output_vars = decoder_model(encoder_model(input_vars)[2])
  vae = keras.Model(input_vars, output_vars)
  vae.summary()

  #loss
  ## also consider binary cross-entropy
  mse = keras.losses.mean_squared_error(input_vars,output_vars)
  mse *= input_dim # WHY ??????
  kl_loss = -0.5* K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)
  vae_loss = K.mean(mse + kl_loss)
  vae.add_loss(vae_loss)

  #compile
  vae.compile(optimizer = keras.optimizers.Adam(learning_rate=0.01))
  return vae

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

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class PermInvEncoder(tf.keras.Model):
    def __init__(self, num_elements, element_size, encoding_size):
        super(PermInvEncoder, self).__init__()
        self.fc1 = keras.layers.Dense(128, activation='relu')
        self.fc2 = keras.layers.Dense(encoding_size)
    
    def call(self, x):
        mask = tf.cast(tf.reduce_sum(tf.abs(x), axis=-1) > 0, tf.float32)
        x *= mask[..., tf.newaxis]
        print("x:", x)
        print("mask:", mask)

        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
        print("x reshape:", x)
        x = self.fc1(x)
        print("x fc1:", x) 
        x = tf.reduce_max(x, axis=1, keepdims=True)
        print("x pooled: ", x)
        x = self.fc2(x)
        return x


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

def get_gvae(input_dim, encoding_dim):
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
  dense3 = keras.layers.Dense(128, kernel_initializer=initializer, name="phi") 
  x = keras.layers.TimeDistributed(dense3, name="tdist_2")(x)
  phi_outputs = keras.layers.Activation('relu')(x)

  # latent space
  sum_phi = keras.layers.Dot(1, name="sum")([masked,phi_outputs])

  # F network
  x = keras.layers.Dense(100, kernel_initializer=initializer)(sum_phi)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dense(100, kernel_initializer=initializer)(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dense(100, kernel_initializer=initializer)(x)
  x = keras.layers.Activation('relu')(x)

  # output
  x = keras.layers.Dense(2, kernel_initializer=initializer, name="output")(x)
  output = keras.layers.Activation('softmax')(x)

  pfn = keras.Model(inputs=pfn_inputs, outputs=output, name="pfn")

  pfn.compile(loss=loss, optimizer=optimizer)
  pfn.summary()
  return pfn

