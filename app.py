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
import plotly.express as px
import pandas as pd

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

# Main content with tabs
tab1, tab2, tab3 = st.tabs(['ECG Analysis', 'Patient History', 'Advanced Tools'])

with tab1:
    # Move existing ECG analysis code here
    st.title('ECG Image Processing Dashboard')

    # Report storage directory
    REPORT_DIR = 'ecg_reports'
    os.makedirs(REPORT_DIR, exist_ok=True)

    def generate_pdf_report(analysis_results, patient_info=None, signals=None):
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
            if key != '_signal_data':
                pdf.cell(40, 10, f'{key}: {value}')
                pdf.ln(10)
        
        # Store signal data in JSON format
        if signals is not None:
            signal_data = {
                'ECG_Clean': signals['ECG_Clean'].tolist(),
                'ECG_R_Peaks': signals['ECG_R_Peaks'].tolist()
            }
            analysis_results['_signal_data'] = signal_data
        
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
                report_filename = generate_pdf_report(analysis_results, signals=signals)
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

with tab2:
    st.title('Patient History Dashboard')
    st.markdown('### Detailed Patient Analysis History')
    
    # Search functionality
    search_query = st.text_input('Search history entries')
    
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            if search_query.lower() in entry['timestamp'].lower() or \
               any(search_query.lower() in str(value).lower() for value in entry['analysis'].values()):
                with st.expander(f'Analysis {i+1} - {entry["timestamp"]}'):
                    st.write('### Analysis Results')
                    for metric, value in entry['analysis'].items():
                        if metric != '_signal_data':
                            st.write(f'{metric}: {value}')
                    
                    # Visualization options
                    if st.button(f'Show Visualization {i+1}', key=f'visual_{i}'):
                        try:
                            # Check for signal data in analysis results
                            if '_signal_data' not in entry['analysis']:
                                raise ValueError('No signal data available in this report')
                            
                            # Get signal data
                            signal_data = entry['analysis']['_signal_data']
                            
                            # Create DataFrame for visualization
                            signals = pd.DataFrame({
                                'ECG_Clean': signal_data['ECG_Clean'],
                                'ECG_R_Peaks': signal_data['ECG_R_Peaks']
                            })
                            
                            # Create interactive visualization
                            fig = px.line(signals, y='ECG_Clean', title=f'ECG Signal - Analysis {i+1}')
                            fig.update_layout(xaxis_title='Time (samples)', yaxis_title='Amplitude')
                            
                            # Add R-peak markers
                            r_peaks = np.where(signals['ECG_R_Peaks'] == 1)[0]
                            fig.add_scatter(x=r_peaks, y=signals['ECG_Clean'][r_peaks], mode='markers', name='R Peaks')
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f'Error loading visualization: {str(e)}')
                    
                    if st.button(f'Download Report {i+1}', key=f'download_{i}'):
                        with open(entry['report_path'], 'rb') as f:
                            st.download_button('Download Report', f, file_name=os.path.basename(entry['report_path']))
    else:
        st.write('No history available')

with tab3:
    st.title('Advanced ECG Tools')
    st.markdown('### Additional ECG Processing Features')
    
    # Signal processing options
    st.subheader('Signal Processing')
    col1, col2 = st.columns(2)
    with col1:
        apply_filter = st.checkbox('Apply Butterworth Filter')
        filter_order = st.slider('Filter Order', 1, 10, 2)
    with col2:
        normalize_signal = st.checkbox('Normalize ECG Signal')
        sampling_rate = st.number_input('Sampling Rate (Hz)', 100, 10000, 1000)
    
    # Advanced analysis options
    st.subheader('Analysis Parameters')
    analysis_params = {
        'window_size': st.number_input('Analysis Window Size (ms)', 100, 1000, 500),
        'threshold': st.slider('Detection Threshold', 0.0, 1.0, 0.5),
        'min_interval': st.number_input('Minimum R-R Interval (ms)', 200, 1000, 400)
    }
    
    if st.button('Run Advanced Analysis') and uploaded_file is not None:
        with st.spinner('Processing ECG with advanced techniques...'):
            try:
                # Reset file pointer and read data
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                if len(file_bytes) == 0:
                    raise ValueError('Uploaded file is empty')
                
                # Decode image and validate
                image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError('Unable to process the uploaded image')
                
                # Process ECG signal
                signal = nk.ecg_clean(image.flatten(), sampling_rate=sampling_rate)
                
                # Apply filters if selected
                if apply_filter:
                    signal = nk.signal_filter(signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=40.0, order=filter_order)
                if normalize_signal:
                    signal = nk.standardize(signal)
                
                # Advanced analysis
                signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
                r_peaks = signals['ECG_R_Peaks']
                rr_intervals = np.diff(np.where(r_peaks == 1)[0])
                
                # Calculate advanced metrics
                advanced_metrics = {
                    'Mean RR Interval': f'{np.mean(rr_intervals):.2f} ms',
                    'SDNN': f'{np.std(rr_intervals):.2f} ms',
                    'RMSSD': f'{np.sqrt(np.mean(np.square(np.diff(rr_intervals)))):.2f} ms',
                    'pNN50': f'{np.mean(np.abs(np.diff(rr_intervals)) > 50) * 100:.2f}%',
                    'LF/HF Ratio': 'N/A'  # Placeholder for frequency domain analysis
                }
                
                # Display results
                st.success('Advanced analysis completed!')
                st.subheader('Advanced Metrics')
                for metric, value in advanced_metrics.items():
                    st.write(f'{metric}: {value}')
                
                # Create interactive visualization
                st.subheader('ECG Signal Visualization')
                fig = px.line(signals, y='ECG_Clean', title='Processed ECG Signal')
                fig.update_layout(xaxis_title='Time (samples)', yaxis_title='Amplitude')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f'Error in advanced analysis: {str(e)}')
