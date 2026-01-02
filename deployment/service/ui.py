import streamlit as st
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .infected {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .uninfected {
        background-color: #e8f5e9;
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
# Change this if your FastAPI runs on a different port
API_URL = "http://127.0.0.1:8000/detect"

def call_detection_api(image_file):
    """
    Call your FastAPI detection endpoint
    """
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image_file.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Prepare the file for upload
        files = {"im": ("image.png", img_byte_arr, "image/png")}
        
        # Make POST request to your API
        response = requests.post(API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Make sure FastAPI is running on http://localhost:8000"
    except requests.exceptions.Timeout:
        return None, "⏱️ Request timeout. The server took too long to respond."
    except Exception as e:
        return None, f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
    st.title("About")
    st.info("""
    This AI-powered system detects malaria parasites in blood cell images using deep learning and ONNX Runtime.
    
    **How to use:**
    1. Upload a blood cell microscopy image
    2. Click 'Analyze'
    3. View the detection results
    
    **Detection Classes:**
    - Parasitized (Infected)
    - Uninfected
    """)
    
    st.markdown("---")
    st.markdown("### Technology Stack")
    st.markdown("""
    - **Model:** Quantized ONNX
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    - **Inference:** ONNX Runtime
    """)

# Main content
st.title("🔬 Malaria Detection System")
st.markdown("### Upload a blood cell microscopy image for analysis")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a microscopy image of a blood cell"
)

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Input Image")
        
        # Image info
        st.write(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"**Format:** {image.format}")
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        # Analyze button
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Call your FastAPI backend
                result, error = call_detection_api(image)
                
                if error:
                    st.error(error)
                    st.info("💡 **Tip:** Make sure your FastAPI server is running with: `uvicorn service.main:app --reload --host 127.0.0.1 --port 8000`")
                else:
                    # Extract results from your API response
                    prediction = result["prediction"]
                    class_idx = result["class_index"]
                    probabilities = result["probabilities"]
                    
                    # Get confidence for the predicted class
                    confidence = probabilities[class_idx]
                    
                    # Display results
                    is_infected = prediction == "Parasitized"
                    box_class = "infected" if is_infected else "uninfected"
                    
                    st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h2>{'PARASITIZED' if is_infected else 'UNINFECTED'}</h2>
                            <h3>{prediction}</h3>
                            <p style="font-size: 24px; font-weight: bold;">
                                Confidence: {confidence*100:.2f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Show detailed probabilities
                    st.markdown("---")
                    st.markdown("### 📊 Detailed Probabilities")
                    
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric(
                            label="Uninfected", 
                            value=f"{probabilities[0]*100:.2f}%"
                        )
                    with prob_col2:
                        st.metric(
                            label="Parasitized", 
                            value=f"{probabilities[1]*100:.2f}%"
                        )
                    
                    # Additional information
                    st.markdown("---")
                    if is_infected:
                        st.warning("⚠️ **Recommendation:** This sample shows signs of parasitic infection. Please consult with a medical professional for confirmation and treatment.")
                    else:
                        st.success("✅ **Result:** No parasitic infection detected in this sample.")
                    
                    # Download results
                    report = f"""MALARIA DETECTION REPORT
=====================================
Prediction: {prediction}
Confidence: {confidence*100:.2f}%

Detailed Probabilities:
- Uninfected: {probabilities[0]*100:.2f}%
- Parasitized: {probabilities[1]*100:.2f}%

Class Index: {class_idx}
=====================================
Note: This is an AI-assisted diagnosis and should be confirmed by a medical professional.
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name="malaria_detection_report.txt",
                        mime="text/plain"
                    )

else:
    # Show example or instructions when no image is uploaded
    st.info("👆 Please upload a blood cell image to begin analysis")
    
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    **Step 1:** Make sure your FastAPI backend is running:
    ```bash
    uvicorn service.main:app --reload
    ```
    
    **Step 2:** Upload a blood cell microscopy image (JPG, JPEG, or PNG)
    
    **Step 3:** Click the 'Analyze Image' button
    
    **Step 4:** View results and download the report
    """)
    
    # Example placeholders
    st.markdown("### 📋 Sample Results Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://via.placeholder.com/300x300.png?text=Infected+Cell", caption="Example: Infected Cell")
    with col2:
        st.image("https://via.placeholder.com/300x300.png?text=Uninfected+Cell", caption="Example: Uninfected Cell")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>This system is for research and educational purposes. Medical decisions should be made by qualified healthcare professionals.</p>
      
    </div>
""", unsafe_allow_html=True)