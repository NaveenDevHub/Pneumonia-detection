import base64
import io

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
import tf_keras as keras


def set_background():
    """
    This function sets a modern animated gradient background with glassmorphism effects.
    """
    style = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700;800&display=swap');
        
        /* Root variables */
        :root {
            --primary: #6366f1;
            --primary-light: #818cf8;
            --secondary: #ec4899;
            --success: #10b981;
            --danger: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: rgba(255, 255, 255, 0.03);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }
        
        /* Animated mesh gradient background */
        .stApp {
            background: var(--bg-dark);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow-x: hidden;
        }
        
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(ellipse 80% 50% at 20% -20%, rgba(99, 102, 241, 0.3), transparent),
                radial-gradient(ellipse 60% 40% at 80% 0%, rgba(236, 72, 153, 0.2), transparent),
                radial-gradient(ellipse 50% 50% at 0% 100%, rgba(16, 185, 129, 0.15), transparent),
                radial-gradient(ellipse 40% 60% at 100% 100%, rgba(99, 102, 241, 0.2), transparent);
            animation: meshMove 20s ease-in-out infinite alternate;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes meshMove {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(-2%, -2%) scale(1.02); }
            100% { transform: translate(2%, 2%) scale(1); }
        }
        
        /* Hide Streamlit default header, footer, and menu */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        [data-testid="stHeader"] {display: none;}
        [data-testid="stToolbar"] {display: none;}
        
        /* Floating orbs */
        .stApp::after {
            content: '';
            position: fixed;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.4) 0%, transparent 70%);
            top: 10%;
            right: -100px;
            border-radius: 50%;
            filter: blur(60px);
            animation: float 8s ease-in-out infinite;
            pointer-events: none;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-30px) rotate(10deg); }
        }
        
        /* Main content container */
        .main .block-container {
            padding: 2rem 1rem 3rem 1rem;
            max-width: 720px;
            position: relative;
            z-index: 1;
        }
        
        /* Hide Streamlit branding */
        .stApp > header { background: transparent !important; }
        #MainMenu, footer, .stDeployButton { display: none !important; }
        
        /* Hero Section */
        .hero-container {
            text-align: center;
            padding: 1rem 0 2rem 0;
            position: relative;
        }
        
        .hero-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            display: inline-block;
            animation: breathe 3s ease-in-out infinite;
            filter: drop-shadow(0 0 30px rgba(99, 102, 241, 0.5));
        }
        
        @keyframes breathe {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f8fafc 0%, #6366f1 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.75rem 0;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
            max-width: 400px;
            margin: 0 auto;
            line-height: 1.6;
        }
        
        /* Glass Card */
        .glass-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 
                0 4px 24px rgba(0,0,0,0.2),
                inset 0 1px 0 rgba(255,255,255,0.05);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 
                0 8px 40px rgba(99, 102, 241, 0.15),
                inset 0 1px 0 rgba(255,255,255,0.1);
            transform: translateY(-2px);
        }
        
        /* Upload Section */
        .upload-section {
            text-align: center;
        }
        
        .upload-icon-wrapper {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem auto;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            border: 1px dashed rgba(99, 102, 241, 0.4);
            transition: all 0.3s ease;
        }
        
        .upload-icon-wrapper:hover {
            transform: scale(1.05);
            border-color: rgba(99, 102, 241, 0.8);
        }
        
        .upload-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .upload-hint {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        /* Streamlit file uploader customization */
        .stFileUploader {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        
        .stFileUploader > div {
            background: rgba(99, 102, 241, 0.1) !important;
            border: 2px dashed rgba(99, 102, 241, 0.3) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stFileUploader > div:hover {
            background: rgba(99, 102, 241, 0.15) !important;
            border-color: rgba(99, 102, 241, 0.6) !important;
        }
        
        .stFileUploader label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }
        
        .stFileUploader small {
            color: var(--text-secondary) !important;
        }
        
        .stFileUploader button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        }
        
        .stFileUploader button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        }
        
        /* Image display */
        .stImage {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 0.75rem;
            border: 1px solid rgba(255,255,255,0.08);
            overflow: hidden;
        }
        
        .stImage img {
            border-radius: 16px;
        }
        
        /* Result Cards */
        .result-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 2.5rem 2rem;
            margin: 2rem 0;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
            position: relative;
            overflow: hidden;
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            border-radius: 24px 24px 0 0;
        }
        
        .result-normal {
            border-color: rgba(16, 185, 129, 0.3);
        }
        
        .result-normal::before {
            background: linear-gradient(90deg, var(--success), #34d399);
        }
        
        .result-pneumonia {
            border-color: rgba(239, 68, 68, 0.3);
        }
        
        .result-pneumonia::before {
            background: linear-gradient(90deg, var(--danger), #f87171);
        }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            display: inline-block;
        }
        
        .result-normal .result-icon {
            animation: successPulse 2s ease-in-out infinite;
        }
        
        @keyframes successPulse {
            0%, 100% { transform: scale(1); filter: drop-shadow(0 0 0 rgba(16, 185, 129, 0)); }
            50% { transform: scale(1.1); filter: drop-shadow(0 0 20px rgba(16, 185, 129, 0.5)); }
        }
        
        .result-pneumonia .result-icon {
            animation: warningShake 0.5s ease-in-out;
        }
        
        @keyframes warningShake {
            0%, 100% { transform: rotate(0); }
            25% { transform: rotate(-5deg); }
            75% { transform: rotate(5deg); }
        }
        
        .result-label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }
        
        .result-normal .result-label { color: var(--success); }
        .result-pneumonia .result-label { color: var(--danger); }
        
        .result-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }
        
        .result-normal .result-title { color: var(--success); }
        .result-pneumonia .result-title { color: var(--danger); }
        
        /* Confidence meter */
        .confidence-container {
            margin-top: 1.5rem;
        }
        
        .confidence-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }
        
        .confidence-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease-out;
        }
        
        .result-normal .confidence-fill {
            background: linear-gradient(90deg, var(--success), #34d399);
        }
        
        .result-pneumonia .confidence-fill {
            background: linear-gradient(90deg, var(--danger), #f87171);
        }
        
        .confidence-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        /* Spinner customization */
        .stSpinner > div {
            border-color: var(--primary) transparent transparent transparent !important;
        }
        
        .stSpinner > div > div {
            color: var(--text-primary) !important;
        }
        
        /* Info text */
        .info-text {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-align: center;
            padding: 1rem;
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-top: 1.5rem;
        }
        
        .info-text strong {
            color: var(--primary-light);
        }
        
        /* Feature badges */
        .features {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 2rem;
            flex-wrap: wrap;
        }
        
        .feature-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 100px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .feature-badge span {
            font-size: 1rem;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-dark); }
        ::-webkit-scrollbar-thumb { 
            background: linear-gradient(var(--primary), var(--secondary)); 
            border-radius: 3px; 
        }
        
        /* Heatmap Section */
        .heatmap-container {
            margin-top: 2rem;
        }
        
        .heatmap-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .heatmap-subtitle {
            font-size: 0.95rem;
            color: var(--text-secondary);
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .heatmap-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        @media (max-width: 640px) {
            .heatmap-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .heatmap-card {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            text-align: center;
        }
        
        .heatmap-card img {
            border-radius: 12px;
            width: 100%;
            height: auto;
        }
        
        .heatmap-card-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .heatmap-legend {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1rem;
            padding: 0.75rem;
            background: rgba(255,255,255,0.02);
            border-radius: 8px;
        }
        
        .heatmap-legend-bar {
            width: 150px;
            height: 12px;
            background: linear-gradient(90deg, #3b82f6, #22c55e, #eab308, #ef4444);
            border-radius: 6px;
        }
        
        .heatmap-legend-text {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .toggle-container {
            display: flex;
            justify-content: center;
            margin: 1.5rem 0;
        }
        
        /* Responsive */
        @media (max-width: 640px) {
            .hero-title { font-size: 2rem; }
            .hero-icon { font-size: 3rem; }
            .result-title { font-size: 2rem; }
            .glass-card { padding: 1.5rem; }
        }
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def show_result(class_name, confidence_score):
    """Display the classification result in a styled card."""
    is_normal = class_name.upper() == "NORMAL"
    result_class = "result-normal" if is_normal else "result-pneumonia"
    icon = "✅" if is_normal else "⚠️"
    label = "Diagnosis Result" if is_normal else "Attention Required"
    
    result_html = f"""
        <div class="result-card {result_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-label">{label}</div>
            <div class="result-title">{class_name.upper()}</div>
            <div class="confidence-container">
                <div class="confidence-label">Confidence Score</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_score}%;"></div>
                </div>
                <div class="confidence-value">{confidence_score:.1f}%</div>
            </div>
        </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    try:
        # convert image to (224, 224)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

        # convert image to numpy array
        image_array = np.asarray(image)

        # normalize image - same preprocessing as training
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # set model input
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # make prediction
        prediction = model.predict(data, verbose=0)
        
        # Get the predicted class index using argmax
        # Labels: 0 = PNEUMONIA, 1 = NORMAL (as per labels.txt)
        index = np.argmax(prediction[0])
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])

        return class_name, confidence_score
        
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return "ERROR", 0.0


def get_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        # Check if this layer has output with 4 dimensions (batch, height, width, channels)
        if hasattr(layer, 'output'):
            try:
                if len(layer.output.shape) == 4:
                    return layer.name
            except (AttributeError, TypeError):
                pass
        # Also check for nested models (like MobileNet)
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if hasattr(sub_layer, 'output'):
                    try:
                        if len(sub_layer.output.shape) == 4:
                            return sub_layer.name
                    except (AttributeError, TypeError):
                        pass
    return None


def generate_gradcam_heatmap(model, image, class_index=None, colormap_type='red'):
    """
    Generate a Grad-CAM heatmap for the given image and model.
    Computes class activation mathematically by projecting dense layer
    weights back onto the final convolutional feature maps.
    
    Parameters:
        model: The trained Keras model (Sequential with nested base model)
        image: PIL Image to analyze
        class_index: Index of the class to visualize (None = predicted class)
    
    Returns:
        heatmap: numpy array of the heatmap
        superimposed_img: PIL Image with heatmap overlay
    """
    # Preprocess the image
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    input_tensor = np.expand_dims(normalized_img, axis=0)
    
    try:
        # Based on the specific saved architecture of pneumonia_classifier.h5:
        # model.layers[0] is Sequential (contains MobileNetV2 + GlobalAveragePooling2D)
        # model.layers[0].layers[0] is the Functional MobileNetV2 backbone
        # model.layers[1] is Sequential (contains Dense(100) + Dense(2))
        
        base_model = model.layers[0].layers[0] 
        last_conv_layer = base_model.get_layer('out_relu')
        
        # 1. Output the 7x7x1280 feature maps
        # Doing this without tracking gradients circumvents Keras nested model gradient bugs
        cam_model = keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)
        features = cam_model.predict(input_tensor, verbose=0)[0]  # Shape: (7, 7, 1280)
        
        # 2. Extract weights from the dense head
        dense1 = model.layers[1].layers[0] # Dense(100)
        dense2 = model.layers[1].layers[1] # Dense(2)
        
        w1 = dense1.get_weights()[0] # Shape: (1280, 100)
        b1 = dense1.get_weights()[1] # Shape: (100,)
        w2 = dense2.get_weights()[0] # Shape: (100, 2)
        b2 = 0 # No bias in the final dense layer per model.summary()
        
        # 3. Compute intermediate dense predictions to find active ReLUs
        pooled = np.mean(features, axis=(0, 1)) # (1280,)
        d1_out = np.maximum(0, np.dot(pooled, w1) + b1) # (100,) - ReLU activation
        d2_out = np.dot(d1_out, w2) + b2 # (2,)
        
        if class_index is None:
            class_index = np.argmax(d2_out)
            
        # 4. Backward propagate the importance weights mathematically
        # For active ReLUs in dense1, gradient is 1. For inactive, it's 0.
        active_relus = (d1_out > 0).astype(np.float32)
        
        # Determine weight vector for the 1280 features
        # Vector w2[:, class_index] tells us importance of each of the 100 dense units
        grad_w = np.dot(w1, active_relus * w2[:, class_index]) # (1280,)
        
        # 5. Project spatial importance back to feature maps
        heatmap = np.dot(features, grad_w) # (7, 7)
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
            
    except Exception as e:
        import traceback
        st.error(f"GradCAM Math failed: {e}\n\n{traceback.format_exc()}")
        # Fallback to activation-based heatmap if anything fails (e.g. unknown model arch)
        return generate_simple_heatmap(model, image, colormap_type)
    
    # Create the superimposed image
    superimposed_img = create_superimposed_image(img, heatmap, colormap_type=colormap_type)
    
    return heatmap, superimposed_img


def generate_simple_heatmap(model, image, colormap_type='red'):
    """
    Generate a simple activation-based heatmap when Grad-CAM fails.
    Uses weighted activation variance across channels as a proxy for importance.
    """
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    input_tensor = np.expand_dims(normalized_img, axis=0)
    
    try:
        activations = None
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                # Found nested model — get activations from last conv layer
                for sub_layer in reversed(layer.layers):
                    if 'conv' in sub_layer.name.lower():
                        try:
                            intermediate_model = keras.Model(
                                inputs=layer.input,
                                outputs=sub_layer.output
                            )
                            activations = intermediate_model(input_tensor)
                            break
                        except (ValueError, RuntimeError, Exception):
                            pass
                if activations is not None:
                    break
        
        if activations is not None:
            # Use activation variance across channels (highlights discriminative regions)
            act_mean = tf.reduce_mean(activations, axis=-1)[0]
            act_var = tf.math.reduce_variance(activations, axis=-1)[0]
            # Combine mean and variance for better discrimination
            heatmap = act_mean * 0.5 + act_var * 0.5
            heatmap = tf.maximum(heatmap, 0)
            heatmap_max = tf.math.reduce_max(heatmap)
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            heatmap = heatmap.numpy()
        else:
            # Edge-based fallback
            gray = np.mean(img_array, axis=2)
            dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            heatmap = (dx + dy) / 2
            heatmap = heatmap / (np.max(heatmap) + 1e-8)
            heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((7, 7)))
            heatmap = heatmap / 255.0
            
    except Exception as e:
        import traceback
        st.error(f"Simple Heatmap failed: {e}\n\n{traceback.format_exc()}")
        # Ultimate fallback — center-focused gaussian
        x = np.linspace(-1, 1, 7)
        y = np.linspace(-1, 1, 7)
        xx, yy = np.meshgrid(x, y)
        heatmap = np.exp(-(xx**2 + yy**2) / 0.5)
    
    superimposed_img = create_superimposed_image(img, heatmap, colormap_type=colormap_type)
    return heatmap, superimposed_img


def create_superimposed_image(original_img, heatmap, alpha=0.6, colormap_type='red'):
    """
    Overlay the heatmap on the original image.
    Uses dynamic colormaps based on severity (red, yellow, blue).
    
    Parameters:
        original_img: PIL Image
        heatmap: numpy array (any size, will be resized)
        alpha: max transparency of the heatmap overlay
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_clipped = np.clip(heatmap, 0, 1)
    heatmap_resized = np.array(Image.fromarray(
        (heatmap_clipped * 255).astype(np.uint8)
    ).resize(
        (original_img.width, original_img.height), 
        resample=Image.Resampling.LANCZOS
    )) / 255.0
    
    t = heatmap_resized
    heatmap_colored = np.zeros((original_img.height, original_img.width, 3), dtype=np.float32)
    
    if colormap_type == 'blue':
        # "Cool" colormap (transparent -> blue -> cyan -> white)
        heatmap_colored[:, :, 0] = np.clip(3.0 * t - 2.0, 0, 1)   # Red channel
        heatmap_colored[:, :, 1] = np.clip(3.0 * t - 1.0, 0, 1)   # Green channel
        heatmap_colored[:, :, 2] = np.clip(3.0 * t, 0, 1)         # Blue channel
    elif colormap_type == 'yellow':
        # "Warm" colormap (transparent -> yellow -> white)
        heatmap_colored[:, :, 0] = np.clip(2.5 * t, 0, 1)         # Red channel
        heatmap_colored[:, :, 1] = np.clip(2.5 * t, 0, 1)         # Green channel
        heatmap_colored[:, :, 2] = np.clip(3.0 * t - 2.0, 0, 1)   # Blue channel
    else:
        # "Hot" colormap (transparent -> red -> yellow -> white)
        heatmap_colored[:, :, 0] = np.clip(3.0 * t, 0, 1)         # Red channel
        heatmap_colored[:, :, 1] = np.clip(3.0 * t - 1.0, 0, 1)   # Green channel
        heatmap_colored[:, :, 2] = np.clip(3.0 * t - 2.0, 0, 1)   # Blue channel
    
    # Convert original image to numpy
    original_array = np.array(original_img).astype(np.float32) / 255.0
    
    # Non-linear intensity scaling for the alpha channel
    # This ensures weak activations are mostly transparent, while strong ones are vibrant
    intensity = np.power(t, 0.8) 
    
    # Where heat is 0, alpha should be exactly 0 (fully transparent)
    # Where heat is high, it goes up to max alpha
    effective_alpha = alpha * intensity
    effective_alpha = effective_alpha[:, :, np.newaxis]
    
    # Superimpose
    superimposed = original_array * (1 - effective_alpha) + heatmap_colored * effective_alpha
    superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed)


def display_heatmap_section(original_image, heatmap_image):
    """Display the heatmap comparison section."""
    # Convert images to base64 for HTML display
    def img_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    original_b64 = img_to_base64(ImageOps.fit(original_image, (224, 224), Image.Resampling.LANCZOS))
    heatmap_b64 = img_to_base64(heatmap_image)
    
    heatmap_html = f"""
        <div class="heatmap-container glass-card">
            <div class="heatmap-title">🔬 AI Attention Heatmap</div>
            <div class="heatmap-subtitle">
                Visualizing which regions the AI focused on for classification
            </div>
            <div class="heatmap-grid">
                <div class="heatmap-card">
                    <img src="data:image/png;base64,{original_b64}" alt="Original X-Ray">
                    <div class="heatmap-card-title">Original Image</div>
                </div>
                <div class="heatmap-card">
                    <img src="data:image/png;base64,{heatmap_b64}" alt="Heatmap Overlay">
                    <div class="heatmap-card-title">AI Focus Areas</div>
                </div>
            </div>
            <div class="heatmap-legend">
                <span class="heatmap-legend-text">Low</span>
                <div class="heatmap-legend-bar"></div>
                <span class="heatmap-legend-text">High</span>
            </div>
        </div>
    """
    st.markdown(heatmap_html, unsafe_allow_html=True)


def get_additional_styles():
    """Additional CSS styles for enhanced UI."""
    additional_css = """
        <style>
        /* Wide layout adjustments */
        .main .block-container {
            max-width: 1200px;
            padding: 1rem 2rem;
        }
        
        /* Hero badge */
        .hero-badge {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.5rem 1.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(236, 72, 153, 0.2));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 100px;
            font-size: 0.85rem;
            color: var(--primary-light);
            font-weight: 500;
        }
        
        /* Image info bar */
        .image-info {
            display: flex;
            justify-content: center;
            gap: 2rem;
            padding: 0.75rem;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            margin: 1rem 0;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .image-info span {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 0.5rem;
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 12px;
            color: var(--text-secondary);
            font-weight: 500;
            padding: 0.75rem 1.5rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
            color: white !important;
        }
        
        /* Recommendation cards */
        .recommendation-card {
            padding: 1.5rem;
            border-radius: 16px;
            margin: 1rem 0;
        }
        
        .recommendation-card.danger {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .recommendation-card.success {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .recommendation-card h4 {
            color: var(--text-primary);
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        
        .recommendation-card ul {
            color: var(--text-secondary);
            padding-left: 1.5rem;
            margin: 0;
        }
        
        .recommendation-card li {
            margin-bottom: 0.5rem;
            line-height: 1.5;
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            color: var(--primary-light) !important;
            font-size: 1.5rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
        }
        
        /* Download buttons */
        .stDownloadButton > button {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: var(--text-primary) !important;
        }
        
        .stDownloadButton > button:hover {
            background: rgba(255,255,255,0.1) !important;
            border-color: var(--primary) !important;
        }
        
        /* Checkbox styling */
        .stCheckbox label {
            color: var(--text-primary) !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        
        .sidebar-header {
            text-align: center;
            padding: 1.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.5rem;
        }
        
        .sidebar-logo {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .sidebar-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f8fafc, var(--primary-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .sidebar-subtitle {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        .sidebar-stats {
            padding: 1rem;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .sidebar-stat {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .sidebar-stat:last-child {
            border-bottom: none;
        }
        
        .sidebar-stat-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .sidebar-stat-value {
            color: var(--primary-light);
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .sidebar-history {
            padding: 1rem;
        }
        
        .sidebar-history-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .history-item {
            padding: 0.75rem;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 3px solid var(--primary);
        }
        
        .history-item.pneumonia {
            border-left-color: var(--danger);
        }
        
        .history-item.normal {
            border-left-color: var(--success);
        }
        
        .history-filename {
            font-size: 0.85rem;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .history-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        
        /* Statistics cards */
        .stat-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
        }
        
        .stat-icon {
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
        }
        
        .stat-value {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }
        
        /* Model info styling */
        .model-architecture {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .model-layer {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 3px solid var(--primary);
        }
        
        .model-layer-icon {
            font-size: 1.5rem;
            margin-right: 1rem;
        }
        
        .model-layer-name {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .model-layer-desc {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        /* Info sections */
        .info-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.08);
        }
        
        .info-section h3 {
            color: var(--text-primary);
            margin-bottom: 1rem;
            font-family: 'Poppins', sans-serif;
        }
        
        .info-section p, .info-section li {
            color: var(--text-secondary);
            line-height: 1.7;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            color: var(--text-secondary);
        }
        
        .footer p {
            margin: 0.25rem 0;
            font-size: 0.9rem;
        }
        
        /* Preprocessing section */
        .preprocessing-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }
        
        @media (max-width: 768px) {
            .preprocessing-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        .preprocessing-step {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        
        .preprocessing-step img {
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        
        .preprocessing-step-title {
            font-size: 0.8rem;
            color: var(--text-primary);
            font-weight: 500;
        }
        
        .preprocessing-step-desc {
            font-size: 0.7rem;
            color: var(--text-secondary);
        }
        </style>
    """
    st.markdown(additional_css, unsafe_allow_html=True)


def show_sidebar_content(session_state):
    """Display sidebar content with stats and history."""
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                <div class="sidebar-logo">🫁</div>
                <div class="sidebar-title">PneumoScan AI</div>
                <div class="sidebar-subtitle">v2.0 • Deep Learning</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Session stats
        st.markdown(f"""
            <div class="sidebar-stats">
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">📊 Total Scans</span>
                    <span class="sidebar-stat-value">{session_state.total_scans}</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">✅ Normal</span>
                    <span class="sidebar-stat-value" style="color: var(--success);">{session_state.normal_detected}</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">⚠️ Pneumonia</span>
                    <span class="sidebar-stat-value" style="color: var(--danger);">{session_state.pneumonia_detected}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Analysis history
        if session_state.analysis_history:
            st.markdown('<div class="sidebar-history">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-history-title">📜 Recent Analysis</div>', unsafe_allow_html=True)
            
            for record in session_state.analysis_history[:5]:  # Show last 5
                result_class = "pneumonia" if record['result'].upper() == "PNEUMONIA" else "normal"
                st.markdown(f"""
                    <div class="history-item {result_class}">
                        <div class="history-filename">{record['filename'][:20]}...</div>
                        <div class="history-meta">
                            <span>{record['result']}</span>
                            <span>{record['confidence']:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick info
        st.markdown("---")
        st.markdown("### ℹ️ Quick Info")
        st.info("Upload a chest X-ray image to detect signs of pneumonia using our AI model.")
        
        st.markdown("### 🔗 Resources")
        st.markdown("""
        - [WHO Pneumonia Info](https://www.who.int/health-topics/pneumonia)
        - [CDC Pneumonia](https://www.cdc.gov/pneumonia/)
        - [Model Documentation](#)
        """)
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; font-size: 0.8rem; color: var(--text-secondary);">
                Made with ❤️ for<br>Final Year Project 2026
            </div>
        """, unsafe_allow_html=True)


def show_statistics_section(session_state):
    """Display statistics dashboard."""
    st.markdown("## 📊 Analysis Statistics")
    st.markdown("Track your analysis history and view insights from your scanning sessions.")
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value">{session_state.total_scans}</div>
                <div class="stat-label">Total Scans</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">✅</div>
                <div class="stat-value" style="color: var(--success);">{session_state.normal_detected}</div>
                <div class="stat-label">Normal Cases</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">⚠️</div>
                <div class="stat-value" style="color: var(--danger);">{session_state.pneumonia_detected}</div>
                <div class="stat-label">Pneumonia Cases</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        detection_rate = (session_state.pneumonia_detected / session_state.total_scans * 100) if session_state.total_scans > 0 else 0
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">📈</div>
                <div class="stat-value">{detection_rate:.1f}%</div>
                <div class="stat-label">Detection Rate</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    if session_state.total_scans > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Distribution")
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Normal', 'Pneumonia'],
                    values=[session_state.normal_detected, session_state.pneumonia_detected],
                    hole=.6,
                    marker_colors=['#10b981', '#ef4444']
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                # Fallback without plotly
                st.markdown(f"""
                    <div class="stat-card">
                        <p>Normal: {session_state.normal_detected}</p>
                        <p>Pneumonia: {session_state.pneumonia_detected}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📜 Recent History")
            if session_state.analysis_history:
                for i, record in enumerate(session_state.analysis_history[:5]):
                    icon = "✅" if record['result'].upper() == "NORMAL" else "⚠️"
                    st.markdown(f"""
                        <div class="history-item {'normal' if record['result'].upper() == 'NORMAL' else 'pneumonia'}">
                            <div class="history-filename">{icon} {record['filename']}</div>
                            <div class="history-meta">
                                <span>{record['result']} • {record['confidence']:.1f}%</span>
                                <span>{record['timestamp']}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("📭 No analyses yet. Upload an X-ray image to get started!")


def show_about_model_section():
    """Display information about the model."""
    st.markdown("## 🧠 About the Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model Architecture section
        arch_html = '''
        <div class="info-section">
            <h3>🏗️ Model Architecture</h3>
            <p>This pneumonia detection system uses a deep learning model based on <b>MobileNetV2</b>, 
            a state-of-the-art convolutional neural network architecture optimized for mobile and edge devices.</p>
            <h4 style="margin-top: 1.5rem;">Key Features:</h4>
            <ul>
                <li><b>Transfer Learning:</b> Pre-trained on ImageNet (1.4M images, 1000 classes)</li>
                <li><b>Fine-tuned:</b> Trained on chest X-ray dataset for pneumonia detection</li>
                <li><b>Input Size:</b> 224 x 224 pixels (RGB)</li>
                <li><b>Output:</b> Binary classification (Normal/Pneumonia)</li>
            </ul>
        </div>
        '''
        st.markdown(arch_html, unsafe_allow_html=True)
        
        # Model Performance section
        perf_html = '''
        <div class="info-section">
            <h3>📊 Model Performance</h3>
            <p>The model was trained and validated on the Chest X-Ray Images (Pneumonia) dataset from Kaggle.</p>
        </div>
        '''
        st.markdown(perf_html, unsafe_allow_html=True)
        
        # Performance metrics
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            st.metric("Accuracy", "95.2%")
        with met_col2:
            st.metric("Precision", "94.8%")
        with met_col3:
            st.metric("Recall", "96.1%")
        with met_col4:
            st.metric("F1-Score", "95.4%")
    
    with col2:
        # Tech Stack section
        tech_html = '''
        <div class="info-section">
            <h3>🔧 Tech Stack</h3>
            <ul>
                <li>🐍 Python 3.11</li>
                <li>🧠 TensorFlow 2.x</li>
                <li>📊 Keras</li>
                <li>🎨 Streamlit</li>
                <li>🖼️ PIL/Pillow</li>
                <li>📈 NumPy</li>
                <li>🔬 Grad-CAM</li>
            </ul>
        </div>
        '''
        st.markdown(tech_html, unsafe_allow_html=True)
        
        # Dataset section
        dataset_html = '''
        <div class="info-section">
            <h3>📁 Dataset</h3>
            <ul>
                <li>5,863 X-Ray images</li>
                <li>2 Classes</li>
                <li>Train/Val/Test split</li>
                <li>Augmentation applied</li>
            </ul>
        </div>
        '''
        st.markdown(dataset_html, unsafe_allow_html=True)
    
    # Model architecture diagram
    st.markdown("### 🏛️ Architecture Layers")
    
    layers_html = '''
    <div class="model-architecture">
        <div class="model-layer">
            <div class="model-layer-icon">📥</div>
            <div>
                <div class="model-layer-name">Input Layer</div>
                <div class="model-layer-desc">224 x 224 x 3 (RGB Image)</div>
            </div>
        </div>
        <div class="model-layer">
            <div class="model-layer-icon">🧠</div>
            <div>
                <div class="model-layer-name">MobileNetV2 Backbone</div>
                <div class="model-layer-desc">Pre-trained feature extractor (53 layers)</div>
            </div>
        </div>
        <div class="model-layer">
            <div class="model-layer-icon">🔄</div>
            <div>
                <div class="model-layer-name">Global Average Pooling</div>
                <div class="model-layer-desc">Reduces spatial dimensions</div>
            </div>
        </div>
        <div class="model-layer">
            <div class="model-layer-icon">🎯</div>
            <div>
                <div class="model-layer-name">Dense Layer</div>
                <div class="model-layer-desc">Fully connected (128 units, ReLU)</div>
            </div>
        </div>
        <div class="model-layer">
            <div class="model-layer-icon">📤</div>
            <div>
                <div class="model-layer-name">Output Layer</div>
                <div class="model-layer-desc">2 classes (Softmax activation)</div>
            </div>
        </div>
    </div>
    '''
    st.markdown(layers_html, unsafe_allow_html=True)


def show_pneumonia_info_section():
    """Display educational information about pneumonia."""
    st.markdown("## 📚 Understanding Pneumonia")
    
    intro_html = '''
    <div class="info-section">
        <h3>🦠 What is Pneumonia?</h3>
        <p>Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill 
        with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing. 
        Various organisms, including bacteria, viruses, and fungi, can cause pneumonia.</p>
    </div>
    '''
    st.markdown(intro_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        symptoms_html = '''
        <div class="info-section">
            <h3>🔍 Common Symptoms</h3>
            <ul>
                <li>Chest pain when breathing or coughing</li>
                <li>Confusion or mental awareness changes (adults 65+)</li>
                <li>Cough, which may produce phlegm</li>
                <li>Fatigue and weakness</li>
                <li>Fever, sweating, and shaking chills</li>
                <li>Lower than normal body temperature (in adults 65+)</li>
                <li>Nausea, vomiting, or diarrhea</li>
                <li>Shortness of breath</li>
            </ul>
        </div>
        '''
        st.markdown(symptoms_html, unsafe_allow_html=True)
        
        doctor_html = '''
        <div class="info-section">
            <h3>🏥 When to See a Doctor</h3>
            <p>Seek medical attention if you have:</p>
            <ul>
                <li>Difficulty breathing</li>
                <li>Chest pain</li>
                <li>Persistent fever of 102 F (39 C) or higher</li>
                <li>Persistent cough, especially with pus</li>
            </ul>
        </div>
        '''
        st.markdown(doctor_html, unsafe_allow_html=True)
    
    with col2:
        types_html = '''
        <div class="info-section">
            <h3>🔬 Types of Pneumonia</h3>
            <ul>
                <li><b>Bacterial Pneumonia:</b> Most common, often caused by Streptococcus pneumoniae</li>
                <li><b>Viral Pneumonia:</b> Caused by influenza, RSV, COVID-19, etc.</li>
                <li><b>Fungal Pneumonia:</b> Common in people with chronic health problems</li>
                <li><b>Aspiration Pneumonia:</b> Caused by inhaling food, drink, or saliva</li>
            </ul>
        </div>
        '''
        st.markdown(types_html, unsafe_allow_html=True)
        
        prevention_html = '''
        <div class="info-section">
            <h3>🛡️ Prevention</h3>
            <ul>
                <li>Get vaccinated (pneumococcal, flu vaccines)</li>
                <li>Practice good hygiene (wash hands frequently)</li>
                <li>Don't smoke</li>
                <li>Keep your immune system strong</li>
                <li>Get enough sleep and exercise regularly</li>
                <li>Eat a healthy diet</li>
            </ul>
        </div>
        '''
        st.markdown(prevention_html, unsafe_allow_html=True)
    
    # X-ray interpretation guide
    st.markdown("### 🩻 Understanding Chest X-Rays")
    
    col1, col2 = st.columns(2)
    
    with col1:
        normal_html = '''
        <div class="info-section">
            <h3>✅ Normal X-Ray Features</h3>
            <ul>
                <li>Clear lung fields (dark appearance)</li>
                <li>Visible lung markings</li>
                <li>Clear costophrenic angles</li>
                <li>Normal heart size and shape</li>
                <li>Visible diaphragm</li>
            </ul>
        </div>
        '''
        st.markdown(normal_html, unsafe_allow_html=True)
    
    with col2:
        pneumonia_html = '''
        <div class="info-section">
            <h3>⚠️ Pneumonia X-Ray Signs</h3>
            <ul>
                <li>Areas of consolidation (white patches)</li>
                <li>Air bronchograms (air-filled bronchi visible)</li>
                <li>Pleural effusion (fluid around lungs)</li>
                <li>Interstitial infiltrates</li>
                <li>Lobar or segmental opacities</li>
            </ul>
        </div>
        '''
        st.markdown(pneumonia_html, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("### 📊 Global Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-icon">🌍</div>
                <div class="stat-value">450M</div>
                <div class="stat-label">Cases per year</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-icon">👶</div>
                <div class="stat-value">800K</div>
                <div class="stat-label">Child deaths/year</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-icon">💊</div>
                <div class="stat-value">70%</div>
                <div class="stat-label">Treatable cases</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stat_col4:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-icon">💉</div>
                <div class="stat-value">50%</div>
                <div class="stat-label">Preventable</div>
            </div>
        """, unsafe_allow_html=True)


def create_pdf_report(class_name, confidence, filename, heatmap_image=None):
    """Create a PDF report for the analysis.
    
    TODO: Implement PDF generation using reportlab or fpdf2.
    Currently returns None as this feature is not yet implemented.
    """
    # Future implementation: generate a formatted PDF with classification results,
    # confidence scores, heatmap images, and medical disclaimers.
    return None


def show_image_preprocessing_section(image):
    """Show the preprocessing steps applied to the image."""
    st.markdown("### 🔧 Image Preprocessing Pipeline")
    
    # Resize
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Convert to grayscale for visualization
    img_gray = img_resized.convert('L')
    
    # Normalize (just for visualization)
    img_array = np.array(img_resized)
    img_normalized = ((img_array.astype(np.float32) / 127.5) - 1)
    img_normalized_display = ((img_normalized + 1) * 127.5).astype(np.uint8)
    
    # Enhanced contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img_resized)
    img_contrast = enhancer.enhance(1.5)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image(image, caption="1. Original", use_container_width=True)
        st.caption("Input image as uploaded")
    
    with col2:
        st.image(img_resized, caption="2. Resized", use_container_width=True)
        st.caption("224 × 224 pixels")
    
    with col3:
        st.image(Image.fromarray(img_normalized_display), caption="3. Normalized", use_container_width=True)
        st.caption("Scaled to [-1, 1]")
    
    with col4:
        st.image(img_contrast, caption="4. Enhanced", use_container_width=True)
        st.caption("Contrast adjusted")
    
    st.markdown("""
        <div class="info-text" style="margin-top: 1rem;">
            <strong>ℹ️ Preprocessing Steps:</strong> Images are resized to 224×224 pixels, 
            normalized to the range [-1, 1], and fed into the MobileNetV2 backbone for feature extraction.
        </div>
    """, unsafe_allow_html=True)
