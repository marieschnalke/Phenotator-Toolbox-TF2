import streamlit as st
import pandas as pd
import os
import plotly.express as px
from utils import file_utils  # Verwende die vorhandene Funktion get_annotations_from_xml
from utils import constants

# Verzeichnisse für Vorhersagen
predictions_dir = constants.predictions_folder

# Set the layout to wide
st.set_page_config(layout="wide")

# Funktion zum Laden der Vorhersagen
def load_predictions(predictions_dir):
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.xml')]
    data = []
    
    for file in prediction_files:
        file_path = os.path.join(predictions_dir, file)
        predictions = file_utils.get_annotations_from_xml(file_path)
        for pred in predictions:
            data.append({
                "Bild": file,
                "Blütenklasse": pred["name"],
                "Vorhersagewahrscheinlichkeit": pred.get("score", 0),
                "Begrenzungsrahmen": pred["bounding_box"]
            })
    
    return pd.DataFrame(data)

# Lade die Vorhersagen in ein DataFrame
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

# Titel und Einführung
st.markdown('<h1>Toolbox Übersicht</h1>', unsafe_allow_html=True)
st.markdown('<p>Dieses Dashboard gibt einen Überblick über die Ergebnisse der Blütenklassifizierung auf Ihrer Wiese.</p>', unsafe_allow_html=True)

# Set up columns for different sections with custom widths
st.markdown('<div class="section">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    # Übersicht der Rohdaten
    st.markdown('<h2>Ergebnisse der Blütenklassifizierung</h2>', unsafe_allow_html=True)
    st.write("Hier sehen Sie die klassifizierten Daten:")
    st.dataframe(df)

with col2:
    # Verteilung der Vorhersagen nach Klassen (using Plotly for smoother visualization)
    st.markdown('<h2>Verteilung der identifizierten Blütenklassen</h2>', unsafe_allow_html=True)
    class_count = df['Blütenklasse'].value_counts().reset_index()
    class_count.columns = ['Blütenklasse', 'Anzahl']

    # Plotly bar chart
    fig = px.bar(class_count, x='Blütenklasse', y='Anzahl', title='Verteilung der Blütenklassen')
    fig.update_layout(title_font_size=25, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Visualisierung der Vorhersagen für ein ausgewähltes Bild
st.markdown('<div class="section">', unsafe_allow_html=True)
col3, col4 = st.columns([1, 1])

with col3:
    st.markdown('<h3>Vorhersagen für ausgewählte Bilder</h3>', unsafe_allow_html=True)
    selected_image = st.selectbox("Wählen Sie ein Bild zur Ansicht:", df['Bild'].unique())

with col4:
    image_predictions = df[df['Bild'] == selected_image]
    st.write(f"Vorhersagen für Bild: {selected_image}")
    for index, row in image_predictions.iterrows():
        st.write(f"Klasse: {row['Blütenklasse']}, Vorhersagewahrscheinlichkeit: {row['Vorhersagewahrscheinlichkeit']}, Begrenzungsrahmen: {row['Begrenzungsrahmen']}")

st.markdown('</div>', unsafe_allow_html=True)


# Download-Optionen
st.markdown('<div class="section">', unsafe_allow_html=True)
col5, col6 = st.columns([1, 1])

with col5:
    st.markdown('<h3>Ergebnisse herunterladen</h3>', unsafe_allow_html=True)
    st.write("Laden Sie die Ergebnisse als CSV-Datei herunter:")
    st.download_button(label="CSV-Datei herunterladen", data=df.to_csv(), file_name='blütenklassifizierung.csv', mime='text/csv')


st.markdown('</div>', unsafe_allow_html=True)
