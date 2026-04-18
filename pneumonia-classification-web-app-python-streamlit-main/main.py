import streamlit as st
# Use tf_keras for backward compatibility with Keras 2.x models
import tf_keras as keras
from tf_keras.models import load_model
from PIL import Image
import numpy as np
import time
import datetime
import io
import base64

from util import (classify, set_background, show_result, generate_gradcam_heatmap, 
                  display_heatmap_section, get_additional_styles, show_sidebar_content,
                  show_statistics_section, show_about_model_section, show_pneumonia_info_section,
                  show_image_preprocessing_section)

# Page config
st.set_page_config(
    page_title="PneumoScan AI | Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for history tracking
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'pneumonia_detected' not in st.session_state:
    st.session_state.pneumonia_detected = 0
if 'normal_detected' not in st.session_state:
    st.session_state.normal_detected = 0

# Apply custom styling
set_background()
get_additional_styles()

# Load classifier (cached for performance)
@st.cache_resource
def load_classifier():
    return load_model('./model/pneumonia_classifier.h5', compile=False)

model = load_classifier()

# Load class names
@st.cache_data
def load_class_names():
    with open('./model/labels.txt', 'r') as f:
        return [a[:-1].split(' ')[1] for a in f.readlines()]

class_names = load_class_names()

# Sidebar
show_sidebar_content(st.session_state)

# Main Content Area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="hero-icon">🫁</div>
            <h1 class="hero-title">PneumoScan AI</h1>
            <p class="hero-subtitle">
                Deep Learning-Powered Chest X-Ray Analysis for Accurate Pneumonia Detection
            </p>
           
        </div>
    """, unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("""
        <div class="features">
            <div class="feature-badge"><span>🧠</span> MobileNetV2</div>
            <div class="feature-badge"><span>📊</span> 95%+ Accuracy</div>
            <div class="feature-badge"><span>⚡</span> Real-time</div>
            <div class="feature-badge"><span>🔬</span> Explainable AI</div>
        </div>
    """, unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze", "📊 Statistics", "🧠 About Model", "📚 Learn More"])

with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Upload Section
        st.markdown("""
            <div class="glass-card upload-section">
                <div class="upload-icon-wrapper">📤</div>
                <div class="upload-title">Upload Chest X-Ray</div>
                <div class="upload-hint">Supported formats: JPEG, PNG • Max size: 200MB</div>
            </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        file = st.file_uploader(
            "Upload X-ray Image",
            type=['jpeg', 'jpg', 'png'],
            help="Upload a frontal chest X-ray image (PA or AP view) for best results",
            label_visibility="collapsed"
        )
        
        # Display image and classify
        if file is not None:
            image = Image.open(file).convert('RGB')
            
            # Image info
            img_width, img_height = image.size
            file_size = len(file.getvalue()) / 1024  # KB
            
            st.markdown(f"""
                <div class="image-info">
                    <span>📐 {img_width} × {img_height} px</span>
                    <span>📁 {file_size:.1f} KB</span>
                    <span>🖼️ {file.type}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Display uploaded image
            st.markdown('<div class="glass-card" style="padding: 1rem;">', unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="Uploaded X-Ray Image")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis options
            st.markdown("### ⚙️ Analysis Options")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                show_heatmap = st.checkbox('🔬 Generate Heatmap', value=True, 
                                          help='Visualize AI attention regions')
            with col_opt2:
                show_preprocessing = st.checkbox('🔧 Show Preprocessing', value=False,
                                                 help='View image preprocessing steps')
            
            # Analyze button
            if st.button('🚀 Analyze X-Ray', type='primary', use_container_width=True):
                
                # Progress bar for analysis
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Preprocessing
                status_text.text('📋 Step 1/4: Preprocessing image...')
                progress_bar.progress(25)
                time.sleep(0.3)
                
                # Step 2: Feature extraction
                status_text.text('🧠 Step 2/4: Extracting features with CNN...')
                progress_bar.progress(50)
                time.sleep(0.3)
                
                # Step 3: Classification
                status_text.text('🔍 Step 3/4: Classifying with neural network...')
                progress_bar.progress(75)
                class_name, conf_score = classify(image, model, class_names)
                
                # Step 4: Generating results
                status_text.text('📊 Step 4/4: Generating results...')
                progress_bar.progress(100)
                time.sleep(0.2)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Update session statistics
                st.session_state.total_scans += 1
                if class_name.upper() == "PNEUMONIA":
                    st.session_state.pneumonia_detected += 1
                else:
                    st.session_state.normal_detected += 1
                
                # Add to history
                analysis_record = {
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'filename': file.name,
                    'result': class_name,
                    'confidence': conf_score * 100
                }
                st.session_state.analysis_history.insert(0, analysis_record)
                
                # Display result
                show_result(class_name, conf_score * 100)
                
                # Detailed results section
                st.markdown("""
                    <div class="glass-card">
                        <h3 style="color: var(--text-primary); margin-bottom: 1rem;">📋 Detailed Analysis</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics columns
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("Classification", class_name.upper())
                with met_col2:
                    st.metric("Confidence", f"{conf_score * 100:.1f}%")
                with met_col3:
                    risk_level = "High" if class_name.upper() == "PNEUMONIA" and conf_score > 0.8 else "Medium" if class_name.upper() == "PNEUMONIA" else "Low"
                    st.metric("Risk Level", risk_level)
                with met_col4:
                    st.metric("Processing Time", "< 1 sec")
                
                # Show preprocessing if enabled
                if show_preprocessing:
                    show_image_preprocessing_section(image)
                
                # Show heatmap if enabled
                heatmap_image = None  # Initialize variable
                if show_heatmap:
                    with st.spinner('🎨 Generating attention heatmap...'):
                        try:
                            # Map risk level to colormap color
                            if risk_level == "High":
                                cmap = 'red'
                            elif risk_level == "Medium":
                                cmap = 'yellow'
                            else:
                                cmap = 'blue'
                                
                            heatmap, heatmap_image = generate_gradcam_heatmap(model, image, colormap_type=cmap)
                            display_heatmap_section(image, heatmap_image)
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate heatmap: {str(e)}")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if class_name.upper() == "PNEUMONIA":
                    st.markdown("""
                        <div class="recommendation-card danger">
                            <h4>⚠️ Potential Pneumonia Detected</h4>
                            <ul>
                                <li>Consult a pulmonologist or healthcare provider immediately</li>
                                <li>Additional tests may include blood tests, CT scan, or sputum culture</li>
                                <li>Do not delay seeking medical attention</li>
                                <li>Treatment typically involves antibiotics for bacterial pneumonia</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="recommendation-card success">
                            <h4>✅ No Pneumonia Detected</h4>
                            <ul>
                                <li>Continue regular health check-ups</li>
                                <li>Maintain good respiratory hygiene</li>
                                <li>If symptoms persist, consult a healthcare provider</li>
                                <li>Consider getting vaccinated against pneumonia</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Download report section
                st.markdown("### 📥 Export Results")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    # Generate text report
                    report_text = f"""
PNEUMOSCAN AI - ANALYSIS REPORT
================================
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
File: {file.name}

RESULTS
-------
Classification: {class_name.upper()}
Confidence Score: {conf_score * 100:.2f}%
Risk Level: {risk_level}

DISCLAIMER
----------
This analysis is for educational purposes only and should not 
replace professional medical diagnosis. Always consult a 
qualified healthcare provider for medical advice.
                    """
                    st.download_button(
                        label="📄 Download Text Report",
                        data=report_text,
                        file_name=f"pneumoscan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Download analyzed image
                    if show_heatmap and heatmap_image is not None:
                        buf = io.BytesIO()
                        heatmap_image.save(buf, format='PNG')
                        st.download_button(
                            label="🖼️ Download Heatmap",
                            data=buf.getvalue(),
                            file_name=f"heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                # Disclaimer
                st.markdown("""
                    <div class="info-text">
                        <strong>⚠️ Medical Disclaimer:</strong> This AI system is developed for educational and research 
                        purposes as part of a final year project. It should NOT be used as a substitute for professional 
                        medical diagnosis. The predictions made by this system should be verified by qualified healthcare 
                        professionals. Always seek advice from a licensed medical practitioner for any health concerns.
                    </div>
                """, unsafe_allow_html=True)

with tab2:
    show_statistics_section(st.session_state)

with tab3:
    show_about_model_section()

with tab4:
    show_pneumonia_info_section()

# Footer
st.markdown("""
    <div class="footer">
        <p>🎓 Final Year Project | Department of Computer Science | 2025-2026</p>
        <p>Built with ❤️ using Streamlit, TensorFlow & MobileNetV2</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            © 2026 PneumoScan AI. For educational purposes only.
        </p>
    </div>
""", unsafe_allow_html=True)
