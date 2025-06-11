import streamlit as st
import numpy as np
import os, zipfile, tempfile
import pydicom, nibabel as nib
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import trimesh
from components.continuous_slider import continuous_slider

@st.cache_data(show_spinner=False)
def load_dicom_series(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    files = sorted([f for f in os.listdir(temp_dir) if f.lower().endswith('.dcm')])
    slices = [pydicom.dcmread(os.path.join(temp_dir, f)) for f in files]
    slices.sort(key=lambda s: float(getattr(s, 'ImagePositionPatient', [0, 0, 0])[2]))
    vol = np.stack([s.pixel_array for s in slices])
    return vol

@st.cache_data(show_spinner=False)
def load_nifti(nifti_file):
    img = nib.load(nifti_file)
    vol = img.get_fdata()
    return vol

@st.cache_data(show_spinner=False)
def volume_stats(vol):
    vmin, vmax = np.percentile(vol, [1, 99]).astype(float)
    return vmin, vmax

st.set_page_config(
    page_title="MirelesMed CT Viewer",
    page_icon="🩻",
    layout="wide",
)

# Hide Streamlit branding for a cleaner look
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Utilidades para cargar datos ---

st.title("MirelesMed CT Viewer")

# --- Controles principales en la parte superior ---
sidebar = st.sidebar

sidebar.header("Carga de volumen y controles 2D")
uploaded = sidebar.file_uploader(
    "Sube un ZIP de DICOM o un archivo NIfTI", type=["zip", "nii", "nii.gz"]
)
axis_name = sidebar.selectbox("Orientación", ["Axial", "Coronal", "Sagital"])
orientation_map = {"Axial": 0, "Coronal": 1, "Sagital": 2}
axis = orientation_map[axis_name]
if sidebar.button("Cargar volumen") and uploaded:
    if uploaded.name.lower().endswith('.zip'):
        vol = load_dicom_series(uploaded)
    else:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
        temp_file.write(uploaded.read())
        temp_file.close()
        vol = load_nifti(temp_file.name)
        os.unlink(temp_file.name)
    st.session_state['volume'] = vol
    st.session_state.pop('mesh', None)
    st.session_state.pop('clipped_mesh', None)

# --- 2D y controles ---
if 'volume' in st.session_state:
    vol = st.session_state['volume']
    vmin, vmax = volume_stats(vol)
    center = (vmax + vmin) / 2
    width = max(vmax - vmin, 1)
    slice_max = vol.shape[axis] - 1

    with sidebar:
        slice_idx = continuous_slider(
            label="Corte",
            min_value=0,
            max_value=slice_max,
            value=int(st.session_state.get("slice_idx", slice_max // 2)),
            step=1,
            key="slice_idx",
        )
        wc = continuous_slider(
            label="Brillo",
            min_value=int(vmin),
            max_value=int(vmax),
            value=int(st.session_state.get("wc", int(center))),
            step=1,
            key="wc",
        )
        ww = continuous_slider(
            label="Contraste",
            min_value=1,
            max_value=int(width),
            value=int(st.session_state.get("ww", int(width))),
            step=1,
            key="ww",
        )
        thr = continuous_slider(
            label="Umbral",
            min_value=int(vmin),
            max_value=int(vmax),
            value=int(st.session_state.get("thr", int(center))),
            step=1,
            key="thr",
        )

    if axis == 0: img = vol[slice_idx]
    elif axis == 1: img = vol[:, slice_idx]
    else: img = vol[:, :, slice_idx]
    mn = wc - ww/2
    mx = wc + ww/2
    imgw = np.clip(img, mn, mx)
    disp = ((imgw - mn) / ww * 255).astype(np.uint8)

    tab2d, tab3d = st.tabs(["Vista 2D", "Vista 3D y Edición STL"])

    with tab2d:
        st.subheader("Vista 2D")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(disp, clamp=True, channels="GRAY", use_container_width=True)
        mask = img > thr
        overlay = np.zeros((*img.shape, 3), dtype=np.uint8)
        overlay[mask] = [255, 0, 0]
        composite = (
            np.clip(
                np.concatenate([disp[..., None]] * 3, axis=-1) * 0.7 + overlay * 0.3,
                0,
                255,
            ).astype(np.uint8)
        )
        with col_b:
            st.image(composite, channels="RGB", use_container_width=True)
        st.caption("Slice con threshold (máscara en color)")

    # --- 3D Preview y flujo pseudo-interactivo ---
    with tab3d:
        st.header("Vista 3D y Edición STL")
        with st.expander("Opciones de 3D / STL"):
            drag_mode = st.selectbox(
                "Modo de interacción",
                ["orbit", "turntable", "pan"],
                format_func=lambda m: m.capitalize(),
            )
            col1, col2 = st.columns(2)
            with col1:
                step = st.slider(
                    "Resolución mesh (step size)", 1, 5, 1,
                    help="Valores mayores generan mallas más ligeras"
                )
                if st.button("Generar/Actualizar STL"):
                    progress = st.progress(0)
                    with st.spinner("Calculando superficie..."):
                        mask3d = (vol > thr).astype(np.uint8)
                        progress.progress(25)
                        verts, faces, _, _ = marching_cubes(mask3d, level=0, step_size=step)
                        progress.progress(75)
                        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                        st.session_state['mesh'] = mesh
                        st.session_state['verts'] = verts
                        st.session_state['faces'] = faces
                        st.session_state.pop('clipped_mesh', None)
                        progress.progress(100)
                    st.success("STL generado.")
            with col2:
                if 'mesh' in st.session_state:
                    plane_axis = st.selectbox("Eje plano de corte", ["X","Y","Z"], key="plane_axis")
                    plane_pos = st.slider("Posición del plano (%)", 0, 100, 50, key="plane_pos")
                    plane_dir = st.selectbox(
                        "Dirección del corte",
                        ["Desde mínimo", "Desde máximo"],
                        key="plane_dir",
                    )
                    if st.button("Aplicar recorte plano"):
                        verts = st.session_state['verts']
                        faces = st.session_state['faces']
                        axis_num = {"X":0,"Y":1,"Z":2}[plane_axis]
                        axis_vals = verts[:, axis_num]
                        if plane_dir == "Desde mínimo":
                            plane_val = axis_vals.min() + np.ptp(axis_vals) * plane_pos / 100
                            faces_keep = np.all(verts[faces][:, :, axis_num] >= plane_val, axis=1)
                        else:
                            plane_val = axis_vals.max() - np.ptp(axis_vals) * plane_pos / 100
                            faces_keep = np.all(verts[faces][:, :, axis_num] <= plane_val, axis=1)
                        faces_clip = faces[faces_keep]
                        mesh_clip = trimesh.Trimesh(vertices=verts, faces=faces_clip)
                        st.session_state['clipped_mesh'] = mesh_clip
                        st.success("Recorte aplicado.")

        # --- Render 3D preview ---
        preview_mesh = None
        if 'clipped_mesh' in st.session_state:
            preview_mesh = st.session_state['clipped_mesh']
        elif 'mesh' in st.session_state:
            preview_mesh = st.session_state['mesh']

        if preview_mesh:
            verts = preview_mesh.vertices
            faces = preview_mesh.faces
            if len(verts) > 0 and len(faces) > 0:
                x, y, z = verts.T
                if faces.ndim > 2:
                    faces_tri = faces.reshape(-1, 3)
                else:
                    faces_tri = faces
                i, j, k = faces_tri.T
                mesh3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="lightgray", opacity=1.0)
                fig = go.Figure(data=[mesh3d])
                fig.update_layout(
                    scene_aspectmode="data",
                    scene_dragmode=drag_mode,
                    margin=dict(l=0, r=0, b=0, t=0),
                    uirevision="mesh"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Controles: arrastra para rotar, rueda para zoom y Ctrl + Click para desplazar"
                )
            else:
                st.warning("Mesh vacía tras clipping.")

            # --- Exportación STL ---
            if st.button("Exportar STL"):
                export_path = os.path.join(tempfile.gettempdir(), "mesh_export.stl")
                preview_mesh.export(export_path)
                with open(export_path, "rb") as f:
                    st.download_button("Descargar STL", f, file_name="mesh_export.stl", mime="application/sla")

else:
    st.info("Carga un volumen DICOM (.zip de .dcm) o NIfTI (.nii/.nii.gz) para empezar.")


