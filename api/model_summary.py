import tensorflow as tf
model = tf.kerasmodels.load_model("ncf_model.h5", compile=False)
model.summary()