import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

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

## ------------------------------------------------------------------------------------
class PFN_AE(keras.Model):
    #could this be modified to inherit more from the AE class?
    def __init__(self, pfn, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.pfn = pfn
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="loss"
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
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.mse(phi, reconstruction)#, axis=(1, 2)
            )
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.reconstruction_loss_tracker.result(),
        }

    def test_step(self, data): 
        phi = self.pfn(data)
        encoded = self.encoder(phi)
        reconstruction = self.decoder(encoded)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(phi, reconstruction)
        )
        return {
            "loss": reconstruction_loss,
        }

    def call(self, data):
        phi = self.pfn(data)
        encoded = self.encoder(phi)
        reconstruction = self.decoder(encoded)
        return {
            "reconstruction": reconstruction
        }

## ------------------------------------------------------------------------------------
class PFN_SVM(keras.Model):
    #could this be modified to inherit more from the AE class?
    def __init__(self, pfn, svm, **kwargs):
        super().__init__(**kwargs)
        self.pfn = pfn
        self.svm = svm

    def call(self, data):
        phi = self.pfn(data)
        result = self.svm(phi)
        return {
            "result": result
        }


## ------------------------------------------------------------------------------------
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

## ------------------------------------------------------------------------------------
class OneClassSVMWrapper(keras.layers.Wrapper):
    def __init__(self, svm, **kwargs):
        super(OneClassSVMWrapper, self).__init__(svm, **kwargs)

    def build(self, input_shape=None):
        assert len(input_shape) == 2
        self.input_dim = input_shape[-1]
        self.svm.build((input_shape[0],))
        super(OneClassSVMWrapper, self).build()

    def call(self, x, mask=None):
        assert self.built, 'Layer must be built before being called'
        y = self.svm.call(x)
        return y

## ------------------------------------------------------------------------------------
class OneClassSVMLayer(keras.layers.Layer):
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale', **kwargs):
        super(OneClassSVMLayer, self).__init__(**kwargs)
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.support_vectors_ = None
        self.intercept_ = None
        self.one_class_svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.input_dim = input_shape[-1]
        super(OneClassSVMLayer, self).build((input_shape[0],))

    def call(self, inputs):
        self.one_class_svm.fit(inputs)
        self.support_vectors_ = self.one_class_svm.support_vectors_
        self.intercept_ = self.one_class_svm.intercept_
        return self.one_class_svm.decision_function(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_support_vectors(self):
        return self.support_vectors_

    def get_intercept(self):
        return self.intercept_

## ------------------------------------------------------------------------------------
class OneClassSVM_Layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OneClassSVM_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.model = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        self.built = True

    def call(self, inputs):
        return tf.convert_to_tensor(self.model.fit(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape


## ------------------------------------------------------------------------------------
class SVMLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SVMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.svm = OneClassSVM(gamma='scale', nu=0.01)

    def call(self, inputs):
        svm_input = inputs
        # Apply the SVM to the input
        svm_output = self.svm.predict(svm_input)
        return svm_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

## ------------------------------------------------------------------------------------
## 		Functions
## ------------------------------------------------------------------------------------


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

## ------------------------------------------------------------------------------------
def pfn_mask_func(X, mask_val=0):
  # map mask_val to zero and return 1 elsewhere
  return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))


## ------------------------------------------------------------------------------------
def get_pfn_svm(input_dims, phi_dim):
  initializer = keras.initializers.HeNormal()
  loss = keras.losses.CategoricalCrossentropy()
  optimizer = keras.optimizers.Adam() 
 
  input_dim_x = input_dims[0]
  input_dim_y = input_dims[1]

  #input
  pfn_inputs = keras.Input(shape=(None,input_dim_y))
  masked = keras.layers.Lambda(pfn_mask_func, name="mask")(pfn_inputs)

  # Phi network
  dense1 = keras.layers.Dense(50, kernel_initializer=initializer, name="pfn1")
  x = keras.layers.TimeDistributed(dense1, name="tdist_0")(pfn_inputs)
  x = keras.layers.Activation('relu')(x)
  dense2 = keras.layers.Dense(50, kernel_initializer=initializer, name="pfn2") 
  x = keras.layers.TimeDistributed(dense2, name="tdist_1")(x)
  x = keras.layers.Activation('relu')(x)
  dense3 = keras.layers.Dense(phi_dim, kernel_initializer=initializer, name="phi") 
  x = keras.layers.TimeDistributed(dense3, name="tdist_2")(x)
  phi_outputs = keras.layers.Activation('relu')(x)

  # latent space
  sum_phi = keras.layers.Dot(1, name="sum")([masked,phi_outputs])
  print(sum_phi.shape)
  #pfn = keras.Model(pfn_inputs, sum_phi, name="pfn")
  #pfn.summary()
  
  #svm
  svm_output = OneClassSVM_Layer()(sum_phi)
  #svm_inputs = keras.Input(shape=(phi_dim,))
  #svm = OneClassSVM(gamma='scale', nu=0.01).fit(sum_phi)
  svm = keras.Model(pfn_inputs, svm_output)
  #combine
  #pfn_svm = PFN_SVM(pfn, svm)
  #  pfn_svm = keras.models.Sequential()
  #  pfn_svm.add(keras.layers.Dense(50, kernel_initializer=initializer, name="pfn1"))
  #  pfn_svm.add(keras.layers.TimeDistributed(dense1, name="tdist_0"))
  #  pfn_svm.add(keras.layers.Activation('relu'))
  #  pfn_svm.add(keras.layers.Dense(50, kernel_initializer=initializer, name="pfn2"))
  #  pfn_svm.add(keras.layers.TimeDistributed(dense2, name="tdist_1"))
  #  pfn_svm.add(keras.layers.Activation('relu'))
  #  pfn_svm.add(keras.layers.Dense(phi_dim, kernel_initializer=initializer, name="phi"))
  #  pfn_svm.add(keras.layers.TimeDistributed(dense3, name="tdist_2"))
  #  pfn_svm.add(keras.layers.Activation('relu'))
  #  pfn_svm.add(keras.layers.Lambda(pfn_mask_func, name="mask"))
  #  pfn_svm.add(keras.layers.Dot(1, name="sum"))
  #pfn_svm.add(pfn)
  #pfn_svm.add(svm_layer)
  pfn_svm.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
  return svm


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


