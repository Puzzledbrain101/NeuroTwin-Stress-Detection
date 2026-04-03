import streamlit as st
import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
import plotly.graph_objects as go
from nilearn import datasets, surface
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model('stress_xgb.model')
    return model

model = load_model()

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def extract_features_from_window(window, fs=250):
    features = []
    for ch in range(8):
        coeffs = pywt.wavedec(window[:, ch], 'db4', level=4)
        for coeff in coeffs:
            prob = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
            ent = -np.sum(prob * np.log2(prob + 1e-10))
            features.append(ent)
            power = np.sum(coeff**2)
            features.append(power)
    return np.array(features).reshape(1, -1)

# ------------------------------------------------------------
# Brain mapping
# ------------------------------------------------------------
def map_stress_to_brain(stress_level):
    pfc = 1.0 - (stress_level / 3.0)
    amygdala = stress_level / 3.0
    acc = 1.0 - abs(2 * (stress_level / 3.0) - 1.0)
    pfc = max(0, min(1, pfc))
    amygdala = max(0, min(1, amygdala))
    acc = max(0, min(1, acc))
    nerve_pressure = amygdala * (1 - pfc) * 100
    return pfc, amygdala, acc, nerve_pressure

# ------------------------------------------------------------
# Load brain mesh and atlas (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_brain_data():
    fsaverage = datasets.fetch_surf_fsaverage()
    destrieux = datasets.fetch_atlas_surf_destrieux()
    # Use the pial surface for realistic shape
    coords, faces = surface.load_surf_mesh(fsaverage.pial_right)
    return fsaverage, destrieux, coords, faces

fsaverage, destrieux, brain_coords, brain_faces = load_brain_data()

# Region labels
pfc_labels = [
    'G_front_sup', 'G_front_middle', 'G_front_inf-Opercular',
    'G_front_inf-Triangul', 'G_front_inf-Orbital'
]
amygdala_labels = ['Amygdala']
acc_labels = ['G_cingul-Part_insular-cing', 'G_cingul-Part_sup_front']

@st.cache_resource
def get_region_vertices():
    labels = destrieux['labels']
    map_right = destrieux['map_right']
    def get_verts(label_strings):
        verts = []
        for label_str in label_strings:
            if label_str in labels:
                label_idx = labels.index(label_str)
                verts.extend(np.where(map_right == label_idx)[0])
        return np.unique(verts).astype(int)
    return {
        'pfc': get_verts(pfc_labels),
        'amygdala': get_verts(amygdala_labels),
        'acc': get_verts(acc_labels)
    }

region_vertices = get_region_vertices()

# ------------------------------------------------------------
# Compute per-vertex colors (RGBA)
# ------------------------------------------------------------
def get_vertex_colors(pfc, amygdala, acc):
    n_vertices = brain_coords.shape[0]
    # Default gray for non‑region vertices
    colors = np.full((n_vertices, 4), [0.7, 0.7, 0.7, 1.0])  # light gray
    
    # PFC: green, intensity = pfc
    pfc_verts = region_vertices['pfc']
    colors[pfc_verts] = [0, pfc, 0, 1.0]   # green channel scaled by activation
    
    # Amygdala: red
    amy_verts = region_vertices['amygdala']
    colors[amy_verts] = [amygdala, 0, 0, 1.0]
    
    # ACC: blue
    acc_verts = region_vertices['acc']
    colors[acc_verts] = [0, 0, acc, 1.0]
    
    return colors

# ------------------------------------------------------------
# Create Plotly 3D brain figure
# ------------------------------------------------------------
def plot_brain_3d(pfc, amygdala, acc):
    vertex_colors = get_vertex_colors(pfc, amygdala, acc)
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=brain_coords[:,0],
            y=brain_coords[:,1],
            z=brain_coords[:,2],
            i=brain_faces[:,0],
            j=brain_faces[:,1],
            k=brain_faces[:,2],
            vertexcolor=vertex_colors,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
            hoverinfo='none'
        )
    ])
    
    fig.update_layout(
        title=f"PFC: {pfc:.2f} (green) | Amygdala: {amygdala:.2f} (black) | ACC: {acc:.2f} (blue)",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            aspectmode='data'
        ),
        width=600, height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="NeuroTwin: Stress Digital Twin", layout="wide")
st.title("🧠 NeuroTwin: Real‑Time Stress Digital Twin")
st.markdown("Monitor stress in real time using EEG and see the brain's response.")

uploaded_file = st.sidebar.file_uploader("Upload a trial file (.txt)", type="txt")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    eeg = df.iloc[:, 1:9].values
    fs = 250
    window_len = 2 * fs
    step_len = 1 * fs
    n_windows = (len(eeg) - window_len) // step_len + 1

    window_idx = st.sidebar.slider("Select window", 0, n_windows - 1, 0)

    start = window_idx * step_len
    end = start + window_len
    window = eeg[start:end, :]

    features = extract_features_from_window(window)
    dmat = xgb.DMatrix(features)
    pred_proba = model.predict(dmat)
    stress_level = np.argmax(pred_proba[0])
    confidence = float(pred_proba[0][stress_level])
    pfc, amygdala, acc, nerve_pressure = map_stress_to_brain(stress_level)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("EEG Waveform (Channel 1)")
        st.line_chart(window[:, 0])

        st.subheader("3D Brain Model")
        fig = plot_brain_3d(pfc, amygdala, acc)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Stress Level")
        stress_labels = ['Natural', 'Low', 'Mid', 'High']
        st.metric("Prediction", f"{stress_labels[stress_level]}", f"{confidence:.1%} confidence")
        st.progress(confidence)

        st.subheader("Nerve Pressure")
        st.metric("Pressure Index", f"{nerve_pressure:.1f} %")
        if nerve_pressure > 70:
            st.error("⚠️ High nerve pressure detected! Consider a short break.")
        elif nerve_pressure > 40:
            st.warning("⚠️ Moderate nerve pressure – try deep breathing.")
        else:
            st.success("✅ Nerve pressure within normal range.")

        st.subheader("Brain Region Activations")
        st.metric("Prefrontal Cortex", f"{pfc:.2f}")
        st.metric("Amygdala", f"{amygdala:.2f}")
        st.metric("Anterior Cingulate", f"{acc:.2f}")

        st.info("**Colors:** Green = PFC, Black = Amygdala, Blue = ACC. Intensity = activation level.")

    with st.expander("Prediction probabilities"):
        for i, label in enumerate(stress_labels):
            st.write(f"{label}: {pred_proba[0][i]:.3f}")

else:
    st.info("👈 Please upload a `.txt` EEG trial file to start the analysis.")