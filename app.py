import streamlit as st
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import neurokit2 as nk
from fpdf import FPDF
import os
from datetime import datetime
import json

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

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
    st.header('History')
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            if st.button(f'Analysis {i+1} - {entry["timestamp"]}'):
                st.session_state.current_report = entry
    else:
        st.write('No history available')

# Main content
st.title('ECG Image Processing Dashboard')

# Report storage directory
REPORT_DIR = 'ecg_reports'
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_pdf_report(analysis_results, patient_info=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, 'ECG Analysis Report')
    pdf.ln(20)
    pdf.set_font('Arial', '', 12)
    
    # Add patient information
    if patient_info:
        pdf.cell(40, 10, 'Patient Information:')
        pdf.ln(10)
        for key, value in patient_info.items():
            pdf.cell(40, 10, f'{key}: {value}')
            pdf.ln(10)
    
    # Add analysis results
    pdf.cell(40, 10, 'Analysis Results:')
    pdf.ln(10)
    for key, value in analysis_results.items():
        pdf.cell(40, 10, f'{key}: {value}')
        pdf.ln(10)
    
    # Save report
    report_filename = f'{REPORT_DIR}/ecg_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    pdf.output(report_filename)
    return report_filename

uploaded_file = st.file_uploader('Upload an ECG image', type=['png', 'jpg', 'jpeg'])

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

        # Display the uploaded image
        st.image(image, caption='Uploaded ECG Image', use_column_width=True)

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

        # Convert image to signal
        signal = nk.ecg_clean(image.flatten(), sampling_rate=1000)

        # Analyze ECG
        signals, info = nk.ecg_process(signal, sampling_rate=1000)

        # Extract comprehensive metrics
        heart_rate = np.mean(signals['ECG_Rate']) if 'ECG_Rate' in signals else np.nan
        r_peaks = signals['ECG_R_Peaks'] if 'ECG_R_Peaks' in signals else np.array([])
        rr_intervals = np.diff(np.where(r_peaks == 1)[0]) if len(r_peaks) > 1 else np.array([])
        qt_interval = np.mean(signals['ECG_Q_Peaks'] - signals['ECG_T_Peaks']) if 'ECG_Q_Peaks' in signals and 'ECG_T_Peaks' in signals and len(signals['ECG_Q_Peaks']) > 0 and len(signals['ECG_T_Peaks']) > 0 else np.nan
        pr_interval = np.mean(signals['ECG_P_Peaks'] - signals['ECG_R_Peaks']) if 'ECG_P_Peaks' in signals and 'ECG_R_Peaks' in signals and len(signals['ECG_P_Peaks']) > 0 and len(signals['ECG_R_Peaks']) > 0 else np.nan
        
        # Rhythm analysis
        rhythm = 'Regular' if len(rr_intervals) > 1 and np.std(rr_intervals) < 50 else 'Irregular'
        arrhythmia = 'Detected' if len(rr_intervals) > 1 and np.std(rr_intervals) > 50 else 'Not detected'

        # Display analysis results
        st.subheader('Comprehensive ECG Analysis')
        analysis_results = {
            'Heart Rate': f'{heart_rate:.0f} bpm' if not np.isnan(heart_rate) else 'N/A',
            'Rhythm': rhythm,
            'Arrhythmia': arrhythmia,
            'QT Interval': f'{qt_interval:.2f} ms' if not np.isnan(qt_interval) else 'N/A',
            'PR Interval': f'{pr_interval:.2f} ms' if not np.isnan(pr_interval) else 'N/A'
        }
        
        for metric, value in analysis_results.items():
            st.write(f'{metric}: {value}')

        # Generate and store report
        if st.button('Generate Report'):
            report_filename = generate_pdf_report(analysis_results)
            report_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': analysis_results,
                'report_path': report_filename
            }
            st.session_state.history.append(report_entry)
            with open(report_filename, 'rb') as f:
                st.download_button('Download Report', f, file_name=os.path.basename(report_filename))

    except Exception as e:
        st.error(f'Error analyzing ECG: {str(e)}')
else:
    st.write('Please upload an ECG image to analyze')

# Display current report if selected
if 'current_report' in st.session_state:
    st.subheader(f'Report from {st.session_state.current_report["timestamp"]}')
    for metric, value in st.session_state.current_report['analysis'].items():
        st.write(f'{metric}: {value}')
    if st.button('Download This Report'):
        with open(st.session_state.current_report['report_path'], 'rb') as f:
            st.download_button('Download Report', f, file_name=os.path.basename(st.session_state.current_report['report_path']))
