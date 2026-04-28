import tf_keras as keras
import numpy as np
import tensorflow as tf
from PIL import ImageOps, Image

# Load model
model = keras.models.load_model('./model/pneumonia_classifier.h5', compile=False)

# Get the base MobileNet model
base_model = model.layers[0].layers[0] 

# The name of the last conv layer was found to be out_relu originally
last_conv_layer = base_model.get_layer('out_relu')

# Produce output without using gradients, just using intermediate representation and weights
cam_model = keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)

# Process image
img = Image.open('../PNEUMONIA/person1_virus_6.jpeg').convert('RGB')
img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
img_array = np.asarray(img)
normalized_img = (img_array.astype(np.float32) / 127.5) - 1
input_tensor = np.expand_dims(normalized_img, axis=0)

# Get intermediate output (7x7x1280)
features = cam_model.predict(input_tensor, verbose=0)[0] 

# Get dense weights
# model.layers[1] is the Sequential block containing Dense1 (100 units) and Dense2 (2 units)
dense1 = model.layers[1].layers[0] # The 100-unit layer
dense2 = model.layers[1].layers[1] # The 2-unit layer

# Get the weights
w1 = dense1.get_weights()[0] # Shape: (1280, 100)
b1 = dense1.get_weights()[1] # Shape: (100,)
w2 = dense2.get_weights()[0] # Shape: (100, 2)
b2 = 0 # No bias

# Calculate predictions to verify
pooled = np.mean(features, axis=(0, 1)) # (1280,)
d1_out = np.maximum(0, np.dot(pooled, w1) + b1) # (100,)
d2_out = np.dot(d1_out, w2) + b2 # (2,)

# Apply softmax
exp_preds = np.exp(d2_out - np.max(d2_out))
preds = exp_preds / np.sum(exp_preds)
class_index = np.argmax(preds)

print(f"Predicted class: {class_index} (score: {preds[class_index]})")

# Calculate class activation maps mathematically
# 1. Backpropagate the w2 weights through the active ReLUs (gradient = 1 if >0 else 0)
active_relus = (d1_out > 0).astype(np.float32)
grad_w = np.dot(w1, active_relus * w2[:, class_index]) # (1280,)

# 2. Re-weight the feature maps
heatmap = np.dot(features, grad_w) # (7, 7)

# ReLU and normalize
heatmap = np.maximum(heatmap, 0)
heatmap_max = np.max(heatmap)
if heatmap_max > 0:
    heatmap = heatmap / heatmap_max

print('Heatmap max:', np.max(heatmap), 'Heatmap min:', np.min(heatmap), 'Unique values:', len(np.unique(heatmap)))

# Use PIL instead of cv2 to avoid extra dependencies
heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
heatmap_resized = heatmap_img.resize((224, 224), Image.Resampling.BICUBIC)
heatmap_resized.save('./test_heatmap.jpg')
print("Saved raw heatmap to test_heatmap.jpg")
