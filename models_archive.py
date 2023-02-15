import tensorflow as tf
from tensorflow import keras
from keras import backend as K

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

class PFN_AE_delete(keras.Model):
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

def pfn_mask_func(X, mask_val=0):
  # map mask_val to zero and return 1 elsewhere
  return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))

def get_full_PFN(input_dim):
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
  dense3 = keras.layers.Dense(8, kernel_initializer=initializer, name="phi") 
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

  encoder = keras.Model(inputs=pfn_inputs, outputs=sum_phi, name="encoder")

  return pfn, encoder

def get_pfn_ae_long(input_dim, phi_dim, encoding_dim):
  #PFN
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

  #encoder
  encoded = keras.layers.Dense(encoding_dim, activation='sigmoid')(sum_phi)
  encoded = keras.layers.Dense(encoding_dim/4, activation='sigmoid')(encoded)
  #decoder
  decoded = keras.layers.Dense(encoding_dim, activation='sigmoid')(encoded)
  decoded = keras.layers.Dense(phi_dim, activation='sigmoid')(decoded)

  autoencoder = keras.Model(pfn_inputs, decoded)
  autoencoder.summary()
  autoencoder.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam())
  pfn = keras.Model(pfn_inputs, sum_phi, name="pfn")
  return autoencoder, pfn


