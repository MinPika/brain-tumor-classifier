# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {
        background-color: white;
    }
    .stApp {
        background-color: white;
    }
    .title-text {
        font-size: 48px;
        font-weight: bold;
        color: #0EEAF1;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle-text {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0EEAF1;
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
CLASS_NAMES = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']
CLASS_INFO = {
    'Glioma': {
        'description': 'Cancerous brain tumors originating in glial cells',
        'color': '#FF6B6B',
        'severity': 'High',
        'treatment': 'Surgery, radiation therapy, chemotherapy'
    },
    'Meningioma': {
        'description': 'Usually non-cancerous tumors from the meninges',
        'color': '#4ECDC4',
        'severity': 'Medium',
        'treatment': 'Observation, surgery, radiation therapy'
    },
    'Notumor': {
        'description': 'Normal brain scan without detectable tumors',
        'color': '#45B7D1',
        'severity': 'None',
        'treatment': 'No treatment needed'
    },
    'Pituitary': {
        'description': 'Tumors affecting the pituitary gland',
        'color': '#FFA07A',
        'severity': 'Medium',
        'treatment': 'Medication, surgery, radiation therapy'
    }
}

IMAGE_SIZE = (168, 168)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert to grayscale if RGB
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Resize
    img_resized = cv2.resize(img_array, IMAGE_SIZE)
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add channel and batch dimensions
    img_final = np.expand_dims(img_normalized, axis=-1)
    img_final = np.expand_dims(img_final, axis=0)
    
    return img_final, img_resized

def predict(model, image):
    """Make prediction on preprocessed image"""
    prediction = model.predict(image, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx] * 100
    
    return predicted_class_idx, confidence, prediction[0]

def create_confidence_chart(probabilities):
    """Create interactive confidence bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities * 100,
            y=CLASS_NAMES,
            orientation='h',
            marker=dict(
                color=[CLASS_INFO[name]['color'] for name in CLASS_NAMES],
                line=dict(color='rgba(0,0,0,0.5)', width=2)
            ),
            text=[f'{p*100:.2f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence Scores',
        xaxis_title='Confidence (%)',
        yaxis_title='Tumor Type',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14),
        xaxis=dict(range=[0, 100], gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    return fig

def create_confidence_pie(probabilities):
    """Create interactive pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=CLASS_NAMES,
        values=probabilities * 100,
        marker=dict(colors=[CLASS_INFO[name]['color'] for name in CLASS_NAMES]),
        hole=0.4,
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title='Confidence Distribution',
        height=400,
        paper_bgcolor='white',
        font=dict(size=14)
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown('<p class="title-text">🧠 Brain Tumor MRI Classification System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">AI-Powered Medical Image Analysis | Accuracy: 99.31%</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
        st.title("📋 Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home & Upload", "📊 Batch Analysis", "ℹ️ About", "📚 Help"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_preprocessing = st.checkbox("Show Preprocessing Steps", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_details = st.checkbox("Show Class Details", value=True)
        
        st.markdown("---")
        st.markdown("### 📈 Model Info")
        st.info("""
        **Model:** CNN (4 Conv Blocks)
        
        **Accuracy:** 99.31%
        
        **Classes:** 4
        - Glioma
        - Meningioma  
        - No Tumor
        - Pituitary
        """)
        
        st.markdown("---")
        st.warning(" **Disclaimer:** This tool is for educational purposes only. Always consult medical professionals.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please ensure 'model.keras' is in the same directory.")
        return
    
    # ========================================================================
    # PAGE 1: HOME & UPLOAD
    # ========================================================================
    if page == "🏠 Home & Upload":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload MRI Scan")
            uploaded_file = st.file_uploader(
                "Choose an MRI image (JPG, PNG, JPEG)",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a brain MRI scan for classification"
            )
            
            # Sample images option
            st.markdown("#### Or try a sample image:")
            sample_choice = st.selectbox(
                "Select sample:",
                ["None", "Glioma Sample", "Meningioma Sample", "Normal Sample", "Pituitary Sample"]
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
                
                # Analyze button
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing MRI scan..."):
                        # Preprocess
                        preprocessed_img, display_img = preprocess_image(image)
                        
                        # Predict
                        pred_class_idx, confidence, all_probs = predict(model, preprocessed_img)
                        predicted_class = CLASS_NAMES[pred_class_idx]
                        
                        # Store in session state
                        st.session_state['prediction'] = predicted_class
                        st.session_state['confidence'] = confidence
                        st.session_state['probabilities'] = all_probs
                        st.session_state['display_img'] = display_img
                        st.session_state['analyzed'] = True
        
        with col2:
            if 'analyzed' in st.session_state and st.session_state['analyzed']:
                st.markdown("### 🎯 Analysis Results")
                
                # Prediction box
                pred_class = st.session_state['prediction']
                confidence = st.session_state['confidence']
                
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 32px; margin-bottom: 10px;">
                        {pred_class}
                    </div>
                    <div style="font-size: 18px; opacity: 0.9;">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Preprocessing visualization
                if show_preprocessing:
                    st.markdown("#### 🔧 Preprocessed Image")
                    st.image(st.session_state['display_img'], 
                            caption=f"Grayscale {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}", 
                            use_column_width=True,
                            clamp=True)
                
                # Confidence scores
                if show_confidence:
                    st.markdown("#### 📊 Confidence Breakdown")
                    probs = st.session_state['probabilities']
                    
                    for i, class_name in enumerate(CLASS_NAMES):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.progress(float(probs[i]), text=class_name)
                        with col_b:
                            st.metric("", f"{probs[i]*100:.2f}%")
                
                # Class details
                if show_details:
                    st.markdown("#### 📋 Diagnosis Information")
                    info = CLASS_INFO[pred_class]
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <b>Description:</b> {info['description']}<br>
                        <b>Severity:</b> <span style="color: {'red' if info['severity']=='High' else 'orange' if info['severity']=='Medium' else 'green'};">{info['severity']}</span><br>
                        <b>Treatment:</b> {info['treatment']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download results
                st.markdown("#### 💾 Export Results")
                result_text = f"""
Brain Tumor Classification Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PREDICTION: {pred_class}
CONFIDENCE: {confidence:.2f}%

PROBABILITY BREAKDOWN:
{chr(10).join([f'- {CLASS_NAMES[i]}: {st.session_state["probabilities"][i]*100:.2f}%' for i in range(4)])}

DESCRIPTION: {CLASS_INFO[pred_class]['description']}
SEVERITY: {CLASS_INFO[pred_class]['severity']}
RECOMMENDED TREATMENT: {CLASS_INFO[pred_class]['treatment']}

Note: This is an AI-generated report for educational purposes only.
Always consult with qualified medical professionals for diagnosis and treatment.
                """
                
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=result_text,
                    file_name=f"brain_tumor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # ========================================================================
    # PAGE 2: BATCH ANALYSIS
    # ========================================================================
    elif page == "📊 Batch Analysis":
        st.markdown("### 📊 Batch Image Analysis")
        st.info("Upload multiple MRI scans for batch processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple MRI images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Analyze All Images", type="primary"):
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    
                    image = Image.open(file)
                    preprocessed_img, _ = preprocess_image(image)
                    pred_class_idx, confidence, all_probs = predict(model, preprocessed_img)
                    
                    results.append({
                        'filename': file.name,
                        'prediction': CLASS_NAMES[pred_class_idx],
                        'confidence': confidence
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Analysis Complete!")
                
                # Display results
                st.markdown("### 📊 Batch Results")
                
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    avg_conf = np.mean([r['confidence'] for r in results])
                    st.metric("Avg Confidence", f"{avg_conf:.2f}%")
                with col3:
                    most_common = df['prediction'].mode()[0]
                    st.metric("Most Common", most_common)
                with col4:
                    unique_classes = df['prediction'].nunique()
                    st.metric("Unique Classes", unique_classes)
                
                # Distribution chart
                st.markdown("### 📈 Prediction Distribution")
                pred_counts = df['prediction'].value_counts()
                
                fig = px.bar(
                    x=pred_counts.index,
                    y=pred_counts.values,
                    color=pred_counts.index,
                    color_discrete_map={name: CLASS_INFO[name]['color'] for name in CLASS_NAMES},
                    labels={'x': 'Tumor Type', 'y': 'Count'}
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False
                )
                st.plotly_chart(fig)
                
                # Download batch results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=csv,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # PAGE 3: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.markdown("### ℹ️ About This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 Purpose
            This application uses deep learning to classify brain MRI scans into four categories:
            - **Glioma:** Cancerous brain tumors
            - **Meningioma:** Usually non-cancerous tumors
            - **No Tumor:** Normal brain scans
            - **Pituitary:** Pituitary gland tumors
            
            #### 🧠 Model Architecture
            - **Type:** Convolutional Neural Network (CNN)
            - **Layers:** 4 Convolutional Blocks + 3 Dense Layers
            - **Input:** 168x168 Grayscale Images
            - **Output:** 4-Class Classification
            - **Framework:** TensorFlow 2.16
            
            #### 📊 Performance Metrics
            - **Overall Accuracy:** 99.31%
            - **Glioma F1-Score:** 98.99%
            - **Meningioma F1-Score:** 98.70%
            - **No Tumor F1-Score:** 100.00%
            - **Pituitary F1-Score:** 99.34%
            """)
        
        with col2:
            st.markdown("""
            #### 📚 Dataset
            - **Total Images:** 7,023
            - **Training Set:** 5,712 images
            - **Test Set:** 1,311 images
            - **Sources:** Figshare, Br35H, SARTAJ datasets
            
            #### 🔬 Technology Stack
            - TensorFlow/Keras
            - Streamlit
            - OpenCV
            - Plotly
            - NumPy
            
            #### ⚠️ Important Notes
            - This tool is for **educational purposes only**
            - Not intended for clinical diagnosis
            - Always consult medical professionals
            - Results should be verified by radiologists
            
            #### 👨‍💻 Development
            - Neural Networks Course Project
            - ESL372: Intelligent Techniques for Energy Systems
            - Achieved: 99.31% Test Accuracy
            """)
        
        st.markdown("---")
        
        # Model architecture visualization
        st.markdown("#### 🏗️ Model Architecture")
        st.code("""
Input (168, 168, 1)
    ↓
Conv2D (64 filters, 5x5) → MaxPool (3x3)
    ↓
Conv2D (64 filters, 5x5) → MaxPool (3x3)
    ↓
Conv2D (128 filters, 4x4) → MaxPool (2x2)
    ↓
Conv2D (128 filters, 4x4) → MaxPool (2x2)
    ↓
Flatten → Dense (512) → Dropout (0.25)
    ↓
Dense (256) → Dropout (0.2)
    ↓
Dense (4, softmax) → Output
        """, language="text")
    

    elif page == "📚 Help":
        st.markdown("### 📚 User Guide")
        
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown("""
            1. Navigate to **🏠 Home & Upload** page
            2. Click **Browse files** to upload an MRI image
            3. Click **🔍 Analyze Image** button
            4. View the prediction results and confidence scores
            5. Download the report if needed
            """)
        
        with st.expander("📤 Uploading Images"):
            st.markdown("""
            **Supported formats:** JPG, JPEG, PNG
            
            **Best practices:**
            - Use clear, high-quality MRI scans
            - Ensure the brain region is visible
            - Grayscale or RGB images both work
            - Recommended resolution: 168x168 or higher
            """)
        
        with st.expander("📊 Understanding Results"):
            st.markdown("""
            **Prediction:** The most likely tumor type
            
            **Confidence:** Percentage indicating model certainty (0-100%)
            - **>90%:** High confidence
            - **70-90%:** Medium confidence
            - **<70%:** Low confidence (consult professionals)
            
            **Probability Breakdown:** Shows confidence for all 4 classes
            """)
        
        with st.expander("🔧 Preprocessing Steps"):
            st.markdown("""
            The model automatically:
            1. Converts image to grayscale (if RGB)
            2. Resizes to 168x168 pixels
            3. Normalizes pixel values (0-1 range)
            4. Adds required dimensions for model input
            """)
        
        with st.expander("💾 Batch Processing"):
            st.markdown("""
            1. Go to **📊 Batch Analysis** page
            2. Upload multiple images at once
            3. Click **🚀 Analyze All Images**
            4. View summary statistics and distribution
            5. Download results as CSV file
            """)
        
        with st.expander("⚠️ Troubleshooting"):
            st.markdown("""
            **Model not loading:**
            - Ensure `model.keras` file is in the app directory
            - Check file permissions
            
            **Low confidence predictions:**
            - Image quality might be poor
            - Upload a clearer MRI scan
            - Consult medical professionals
            
            **App running slow:**
            - Large images take longer to process
            - Try resizing images before upload
            - Use batch processing for multiple images
            """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips for Best Results")
        st.info("""
        ✅ Use high-quality MRI scans
        ✅ Ensure proper brain region visibility
        ✅ Check multiple angles if available
        ✅ Compare results with medical reports
        ✅ Always verify with healthcare professionals
        """)

if __name__ == "__main__":
    main()