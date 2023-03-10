import tensorflow as tf
from tensorflow import keras
from keras import backend as K

## ------------------------------------------------------------------------------------
class PFN_VAE(keras.Model):
    #could this be modified to inherit more from the VAE class?
    def __init__(self, pfn, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.pfn = pfn
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reco_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.phi_tracker = keras.metrics.Mean(name="sum_phi")

    @property
    def metrics(self):
        return [
            #self.phi_tracker,
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            phi = self.pfn(data)
            z_mean, z_log_var, z = self.encoder(phi)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.mse(phi, reconstruction)
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
            #"sum_phi": self.phi_tracker.result(),
            "loss": self.total_loss_tracker.result(),
            "reco_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        phi = self.pfn(data)
        z_mean, z_log_var, z = self.encoder(phi)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(phi, reconstruction)
        )
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        sum_phi =  tf.reduce_mean(phi)
        return {
            #"sum_phi": sum_phi,
            "loss": total_loss,
            "reco_loss": reconstruction_loss,
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



def sampling(args):
  z_mean, z_log_var = args
  #batch = K.shape(z_mean)[0]
  #latent_dim = K.shape(z_mean)[1]
  #epsilon = K.random_normal(shape=(batch,latent_dim), mean=0., stddev=0.1)
  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2), mean=0., stddev=0.1)
  return z_mean + K.exp(z_log_var) * epsilon

def get_simple_ae(input_dim, encoding_dim, latent_dim):
  input_vars = keras.Input ( shape =(input_dim,))
  # encode
  x = keras.layers.Dense(encoding_dim, activation='relu')(input_vars)
  x = keras.layers.Dense(latent_dim, activation='relu')(x)
  #decode
  x = keras.layers.Dense(encoding_dim, activation='relu')(x)
  decoded = keras.layers.Dense(input_dim, activation='sigmoid')(x)
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

def pfn_mask_func(X, mask_val=0):
  # map mask_val to zero and return 1 elsewhere
  return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))

class supervisedPFN(keras.Model):
  def __init__(self,graph, classifier):
    super().__init__()
    self.graph = graph
    self.classifier = classifier
  
  def call(self, X, *args):
    graph_rep = self.graph(X)
    result = self.classifier(graph_rep)
    return result


def get_full_PFN(input_dim, phi_dim):
  #initializer = keras.initializers.HeNormal()
  loss = keras.losses.CategoricalCrossentropy()
  optimizer = keras.optimizers.Adam() 

  input_dim_x = input_dim[0]
  input_dim_y = input_dim[1]

  #input
  pfn_inputs = keras.Input(shape=(None,input_dim_y))
  masked = keras.layers.Lambda(pfn_mask_func, name="mask")(pfn_inputs)

  # Phi network
  dense1 = keras.layers.Dense(50, name="pfn1")
  x = keras.layers.TimeDistributed(dense1, name="tdist_0")(pfn_inputs)
  x = keras.layers.Activation('relu')(x)
  dense2 = keras.layers.Dense(50, name="pfn2") 
  x = keras.layers.TimeDistributed(dense2, name="tdist_1")(x)
  x = keras.layers.Activation('relu')(x)
  dense3 = keras.layers.Dense(phi_dim, name="phi") 
  x = keras.layers.TimeDistributed(dense3, name="tdist_2")(x)
  phi_outputs = keras.layers.Activation('relu')(x)

  # latent space
  sum_phi = keras.layers.Dot(1, name="sum")([masked,phi_outputs])
  graph = keras.Model(inputs=pfn_inputs, outputs=sum_phi, name="graph")
  graph.summary()
   
  # F network
  classifier_inputs = keras.Input(shape=(phi_dim,))
  x = classifier_inputs
  x = keras.layers.Dense(50, name = "F1")(classifier_inputs)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dense(50, name = "F2")(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Dense(50, name="F3")(x)
  x = keras.layers.Activation('relu')(x)

  # output
  x = keras.layers.Dense(2, name="output")(x)
  output = keras.layers.Activation('softmax')(x)

  classifier = keras.Model(inputs=classifier_inputs, outputs=output, name="classifier")
  classifier.summary()

  pfn = supervisedPFN(graph, classifier)
  #pfn.summary()
  pfn.compile(optimizer=optimizer, loss=loss)
  return pfn, graph

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


