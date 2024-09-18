import streamlit as st
import pandas as pd
import os
import plotly.express as px
from utils import file_utils  # Use the existing function get_annotations_from_xml
from utils import constants

# Directories for predictions
predictions_dir = constants.predictions_folder

# Set the layout to wide
st.set_page_config(layout="wide")

# Function to load predictions
def load_predictions(predictions_dir):
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.xml')]
    data = []
    
    for file in prediction_files:
        file_path = os.path.join(predictions_dir, file)
        predictions = file_utils.get_annotations_from_xml(file_path)
        for pred in predictions:
            data.append({
                "Image": file,
                "Flower Class": pred["name"],
                "Prediction Confidence": pred.get("score", 0),
                "Bounding Box": pred["bounding_box"]
            })
    
    return pd.DataFrame(data)

# Load the predictions into a DataFrame
df = load_predictions(predictions_dir)

# CSS for section boundaries and larger font size
st.markdown("""
    <style>
        .section {
            border: 2px solid #d3d3d3;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
        }
        h1 {
            font-size: 80px !important;
        }
        h2 {
            font-size: 60px !important;
        }
        h3 {
            font-size: 40px !important;
        }
        p, div, label, .stText {
            font-size: 25px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1>Toolbox Overview</h1>', unsafe_allow_html=True)
st.markdown('<p>This dashboard provides an overview of the flower classification results on your field.</p>', unsafe_allow_html=True)

# Set up columns for different sections with custom widths
st.markdown('<div class="section">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    # Overview of raw data
    st.markdown('<h2>Flower Classification Results</h2>', unsafe_allow_html=True)
    st.write("Here are the classified data:")
    st.dataframe(df)

with col2:
    # Distribution of predictions by class (using Plotly for smoother visualization)
    st.markdown('<h2>Distribution of Identified Flower Classes</h2>', unsafe_allow_html=True)
    class_count = df['Flower Class'].value_counts().reset_index()
    class_count.columns = ['Flower Class', 'Count']

    # Plotly bar chart
    fig = px.bar(class_count, x='Flower Class', y='Count', title='Distribution of Flower Classes')
    fig.update_layout(title_font_size=25, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Visualization of predictions for a selected image
st.markdown('<div class="section">', unsafe_allow_html=True)
col3, col4 = st.columns([1, 1])

with col3:
    st.markdown('<h3>Predictions for Selected Images</h3>', unsafe_allow_html=True)
    selected_image = st.selectbox("Select an image to view:", df['Image'].unique())

with col4:
    image_predictions = df[df['Image'] == selected_image]
    st.write(f"Predictions for Image: {selected_image}")
    for index, row in image_predictions.iterrows():
        st.write(f"Class: {row['Flower Class']}, Prediction Confidence: {row['Prediction Confidence']}, Bounding Box: {row['Bounding Box']}")

st.markdown('</div>', unsafe_allow_html=True)


# Download options
st.markdown('<div class="section">', unsafe_allow_html=True)
col5, col6 = st.columns([1, 1])

with col5:
    st.markdown('<h3>Download Results</h3>', unsafe_allow_html=True)
    st.write("Download the results as a CSV file:")
    st.download_button(label="Download CSV File", data=df.to_csv(), file_name='flower_classification.csv', mime='text/csv')

st.markdown('</div>', unsafe_allow_html=True)
