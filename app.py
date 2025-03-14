import streamlit as st
import cv2
import numpy as np
from skimage import filters, measure
import matplotlib.pyplot as plt
import neurokit2 as nk

# Set page config for a better layout
st.set_page_config(page_title='ECG Analyzer', page_icon='❤️', layout='wide')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stSidebar {
        background: linear-gradient(135deg, #3bf8ac, #34495e);
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stFileUploader>div>div>div>div>div {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for additional options
with st.sidebar:
    st.title('ECG Analyzer')
    st.markdown('Upload your ECG image to analyze and detect potential heart conditions.')
    st.markdown('---')
    st.markdown('**Developed by:** Anmol Ratan')

# Main content
st.title('ECG Image Processing Dashboard')

uploaded_file = st.file_uploader('Upload an ECG image', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(image, caption='Uploaded ECG Image.', use_column_width=True)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    st.image(output_image, caption='Processed ECG Image with Detected Contours.', use_column_width=True)

    # Display the number of detected contours
    st.write(f'Number of detected contours: {len(contours)}')

    # ECG analysis
    st.subheader('ECG Analysis Results')
    if uploaded_file is not None:
        try:
            # Read and process ECG image
            uploaded_file.seek(0)  # Reset file pointer
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            if len(file_bytes) == 0:
                raise ValueError('Uploaded file is empty')
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('Unable to process the uploaded image')
            # Convert image to signal
            signal = nk.ecg_clean(image.flatten(), sampling_rate=1000)
            # Analyze ECG
            signals, info = nk.ecg_process(signal, sampling_rate=1000)
            # Extract metrics
            heart_rate = np.mean(signals['ECG_Rate'])
            r_peaks = signals['ECG_R_Peaks']
            rr_intervals = np.diff(np.where(r_peaks == 1)[0])
            arrhythmia = 'Detected' if np.std(rr_intervals) > 50 else 'Not detected'
            st.write(f'Heart Rate: {heart_rate:.0f} bpm')
            st.write(f'Arrhythmia: {arrhythmia}')
            st.write('ST Segment: Analysis not available in this version')
        except Exception as e:
            st.error(f'Error analyzing ECG: {str(e)}')
    else:
        st.write('Please upload an ECG image to analyze')
