import tf_keras as keras
import numpy as np
import tensorflow as tf
from PIL import ImageOps, Image

model = keras.models.load_model('./model/pneumonia_classifier.h5', compile=False)
img = Image.open('../PNEUMONIA/person1_virus_6.jpeg').convert('RGB')
img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
img_array = np.asarray(img)
normalized_img = (img_array.astype(np.float32) / 127.5) - 1
input_tensor = np.expand_dims(normalized_img, axis=0)

# Extract deeply nested models
base_model = model.layers[0].layers[0] # MobileNetV2
global_pool = model.layers[0].layers[1] # GlobalAveragePooling2D
dense_head = model.layers[1] # Sequential (Dense1, Dense2)

last_conv_layer = base_model.get_layer('out_relu')
conv_output = last_conv_layer.output

# Use GradientTape
with tf.GradientTape() as tape:
    # 1. Forward pass through base_model
    base_output = base_model(input_tensor)
    # 2. Forward pass through pool
    pooled_output = global_pool(base_model.output)
    # 3. Forward pass through dense
    classification_output = dense_head(pooled_output)
    
    # We need a model that links input to both conv_output and classification_output
    # To avoid tf_keras tensor mismatch errors, let's create a functional model
    # that encapsulates the full path starting from the base model's input:
    x = base_model.output
    x = global_pool(x)
    y = dense_head(x)
    
grad_model = keras.Model(inputs=base_model.input, outputs=[conv_output, y])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_tensor)
    class_index = tf.argmax(predictions[0])
    class_output = predictions[:, class_index]
    
grads = tape.gradient(class_output, conv_outputs)
if grads is None:
    print('Gradients are None!')
else:
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    print('Heatmap max:', tf.math.reduce_max(heatmap).numpy(), 'Heatmap min:', tf.math.reduce_min(heatmap).numpy(), 'Unique values:', len(np.unique(heatmap.numpy())))
    print('Predictions:', predictions.numpy())
