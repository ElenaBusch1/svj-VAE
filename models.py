import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import Dense, Dropout, LeakyReLU
from sklearn.svm import OneClassSVM
arch_dir = "architectures_saved/"

## ------------------------------------------------------------------------------------
##				Classes
## ------------------------------------------------------------------------------------

## ------------------------------------------------------------------------------------
class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
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
        # for this model custom test & train steps are unnecessary
	# but still implemented to be as directly comparable as possible
        with tf.GradientTape() as tape:
            encoded = self.encoder(data)
            reconstruction = self.decoder(encoded)
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.mse(data, reconstruction)
            )
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.reconstruction_loss_tracker.result(),
        }

    def test_step(self, data): 
        encoded = self.encoder(data)
        reconstruction = self.decoder(encoded)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(data, reconstruction)
        )
        return {
            "loss": reconstruction_loss,
        }

    def call(self, data):
        encoded = self.encoder(data)
        reconstruction = self.decoder(encoded)
        return {
            "reconstruction": reconstruction
        }

## ------------------------------------------------------------------------------------
class VAE(keras.Model):
    #developed from https://keras.io/examples/generative/vae/
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reco_loss"
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
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(data, reconstruction)))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss *= 100
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reco_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(data, reconstruction)))

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        kl_loss *= 100
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reco_loss": reconstruction_loss,
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


## ------------------------------------------------------------------------------------
def get_pfn(input_dims, phi_dim):
  initializer = keras.initializers.HeNormal()
  loss = keras.losses.CategoricalCrossentropy()
  optimizer = keras.optimizers.Adam() 
 
  input_dim_x = input_dims[0]
  input_dim_y = input_dims[1]

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

def get_dnn(input_dim):
  input_vars = keras.Input ( shape =(input_dim,))
  # encode
  x = keras.layers.Dense(50, activation='relu')(input_vars)
  x = keras.layers.Dense(50, activation='relu')(x)
  x = keras.layers.Dense(50, activation='relu')(x)
  output = keras.layers.Dense(2, activation='softmax')(x)
  dnn = keras.Model(input_vars, output)
  dnn.summary()
  dnn.compile(loss = keras.losses.CategoricalCrossentropy(), optimizer = keras.optimizers.Adam(learning_rate=0.001))
  return dnn

## ------------------------------------------------------------------------------------
def get_encoder(input_dim, encoding_dim, latent_dim):
  #encoder = tf.keras.models.Sequential(name="encoder")
  #encoder.add(Dropout(0.1, input_shape=(input_dim,)))
  #encoder.add(Dense(32))
  #encoder.add(Dropout(0.1))
  #encoder.add(LeakyReLU(alpha=0.3))
  #encoder.add(Dense(encoding_dim))
  #encoder.add(Dropout(0.1))
  #encoder.add(LeakyReLU(alpha=0.3))
  #encoder.add(Dense(latent_dim))
  #encoder.add(LeakyReLU(alpha=0.3))
  inputs = keras.Input(shape=(input_dim,))
  encoded = keras.layers.Dense(encoding_dim, activation='relu')(inputs)
  encoder_outputs = keras.layers.Dense(latent_dim, activation='relu')(encoded)

  encoder = keras.Model(inputs, encoder_outputs, name="encoder")
  encoder.summary()
  return encoder

## ------------------------------------------------------------------------------------
def get_variational_encoder(input_dim, encoding_dim, latent_dim):
  inputs = keras.Input(shape=(input_dim,))
  x = Dropout(0.1, input_shape=(input_dim,))(inputs)
  x = Dense(32)(x)
  x = Dropout(0.1)(x)
  x = LeakyReLU(alpha=0.3)(x)
  x = Dense(encoding_dim)(x)
  x = Dropout(0.1)(x)
  x = LeakyReLU(alpha=0.3)(x)
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  
  #x = keras.layers.Dense(encoding_dim, activation="relu")(inputs)
  #z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
  #z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
  #z = Sampling()([z_mean, z_log_var])
  
  encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
  encoder.summary()
  return encoder

## ------------------------------------------------------------------------------------
def get_decoder(input_dim, encoding_dim, latent_dim):
  #decoder = tf.keras.models.Sequential(name="decoder")
  #decoder.add(Dense(encoding_dim, input_dim=latent_dim))
  #decoder.add(Dropout(0.1))
  #decoder.add(LeakyReLU(alpha=0.3))
  #decoder.add(Dense(32, input_dim=latent_dim))
  #decoder.add(Dropout(0.1))
  #decoder.add(LeakyReLU(alpha=0.3))
  #decoder.add(Dense(input_dim, activation='sigmoid'))

  latent_inputs = keras.Input(shape=(latent_dim,))
  x = keras.layers.Dense(encoding_dim, activation="relu")(latent_inputs)
  decoder_outputs = keras.layers.Dense(input_dim, activation="sigmoid")(x)

  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  decoder.summary()
  return decoder

## ------------------------------------------------------------------------------------
def get_ae(input_dim, encoding_dim, latent_dim):
  encoder = get_encoder(input_dim, encoding_dim, latent_dim)
  decoder = get_decoder(input_dim, encoding_dim, latent_dim)

  ae = AE(encoder, decoder)
  ae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
  return ae

## ------------------------------------------------------------------------------------
def get_vae(input_dim, encoding_dim, latent_dim):
  encoder = get_variational_encoder(input_dim, encoding_dim, latent_dim)
  decoder = get_decoder(input_dim, encoding_dim, latent_dim)

  vae = VAE(encoder, decoder)
  vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
  return vae

## ------------------------------------------------------------------------------------
def get_pfn_ae(input_dims, phi_dim, encoding_dim, latent_dim):
  pfn = get_pfn(input_dims, phi_dim)
  encoder = get_encoder(phi_dim, encoding_dim, latent_dim)
  decoder = get_decoder(phi_dim, encoding_dim, latent_dim)

  pfn_ae = PFN_AE(pfn, encoder, decoder)
  pfn_ae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))

  return pfn_ae, pfn

## ------------------------------------------------------------------------------------
def get_pfn_vae(input_dims, phi_dim, encoding_dim, latent_dim):
  pfn = get_pfn(input_dims, phi_dim)
  encoder = get_variational_encoder(phi_dim, encoding_dim, latent_dim)
  decoder = get_decoder(phi_dim, encoding_dim, latent_dim)

  pfn_vae = PFN_VAE(pfn, encoder, decoder)
  pfn_vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))
  return pfn_vae, pfn

def get_trained_pfn_ae(input_dims, phi_dim, encoding_dim, latent_dim):
  graph = keras.models.load_model(arch_dir+'PFN_graph_arch')
  graph.load_weights(arch_dir+'PFN_graph_weights.h5')
  graph.compile()
  

## ------------------------------------------------------------------------------------
def get_model(model_name, input_dims, encoding_dim, latent_dim, phi_dim=None):
  if (model_name == "AE"):
    return get_ae(input_dims, encoding_dim, latent_dim)

  elif (model_name == "VAE"):
    return get_vae(input_dims, encoding_dim, latent_dim)

  elif (model_name == "PFN_AE"): 
    return get_pfn_ae(input_dims, phi_dim, encoding_dim, latent_dim)

  elif (model_name == "PFN_VAE"): 
    return get_pfn_vae(input_dims, phi_dim, encoding_dim, latent_dim)

  else:
    print("ERROR: model name", model_name," not recognized")

