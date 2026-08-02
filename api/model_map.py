import tensorflow as tf

model = tf.keras.models.load_model("../model/ncf_model.h5", compile=False)

for layer in model.layers:
    weights = layer.get_weights()
    shapes = [w.shape for w in weights]
    print(f"{layer.name:30s} {type(layer).__name__:20s} {shapes}")