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
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
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
