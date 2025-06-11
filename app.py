import streamlit as st
import numpy as np
import os, zipfile, tempfile
import pydicom, nibabel as nib
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import trimesh

st.set_page_config(layout="wide")

# --- Utilidades para cargar datos ---

def load_dicom_series(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    files = sorted([f for f in os.listdir(temp_dir) if f.lower().endswith('.dcm')])
    slices = [pydicom.dcmread(os.path.join(temp_dir, f)) for f in files]
    slices.sort(key=lambda s: float(getattr(s, 'ImagePositionPatient', [0,0,0])[2]))
    vol = np.stack([s.pixel_array for s in slices])
    return vol

def load_nifti(nifti_file):
    img = nib.load(nifti_file)
    vol = img.get_fdata()
    return vol

st.title("Visor/Editor CT 2D+3D STL (Streamlit)")

# --- Barra lateral: carga y controles ---
with st.sidebar:
    st.header("Carga de volumen")
    uploaded = st.file_uploader("Sube un ZIP de DICOM o un archivo NIfTI", type=["zip","nii","nii.gz"])
    axis_name = st.selectbox("Orientación", ["Axial", "Coronal", "Sagital"])
    orientation_map = {'Axial': 0, 'Coronal': 1, 'Sagital': 2}
    axis = orientation_map[axis_name]
    if st.button("Cargar volumen") and uploaded:
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
    vmin, vmax = float(np.percentile(vol, [1,99])[0]), float(np.percentile(vol, [1,99])[1])
    center = (vmax+vmin)/2
    width = max(vmax-vmin, 1)
    slice_max = vol.shape[axis] - 1

    with st.sidebar:
        slice_idx = st.slider("Slice", 0, slice_max, slice_max//2)
        wc = st.slider("Window Center", int(vmin), int(vmax), int(center))
        ww = st.slider("Window Width", 1, int(width), int(width))
        thr = st.slider("Threshold", int(vmin), int(vmax), int(center))

    if axis == 0: img = vol[slice_idx]
    elif axis == 1: img = vol[:, slice_idx]
    else: img = vol[:, :, slice_idx]
    mn = wc - ww/2
    mx = wc + ww/2
    imgw = np.clip(img, mn, mx)
    disp = ((imgw - mn) / ww * 255).astype(np.uint8)

    st.subheader("Vista 2D")
    st.image(disp, clamp=True, channels="GRAY", width=400)
    st.caption("Slice con threshold (máscara en color)")

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
    st.image(composite, channels="RGB", width=400)

    # --- 3D Preview y flujo pseudo-interactivo ---
    st.header("Vista 3D y Edición STL")
    with st.expander("Opciones de 3D / STL"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generar/Actualizar STL"):
                with st.spinner("Calculando superficie..."):
                    mask3d = (vol > thr).astype(np.uint8)
                    verts, faces, _, _ = marching_cubes(mask3d, level=0)
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                    st.session_state['mesh'] = mesh
                    st.session_state['verts'] = verts
                    st.session_state['faces'] = faces
                    st.session_state.pop('clipped_mesh', None)
                st.success("STL generado.")
        with col2:
            if 'mesh' in st.session_state:
                plane_axis = st.selectbox("Eje plano de corte", ["X","Y","Z"], key="plane_axis")
                plane_pos = st.slider("Posición del plano (%)", 0, 100, 50, key="plane_pos")
                if st.button("Aplicar recorte plano"):
                    verts = st.session_state['verts']
                    faces = st.session_state['faces']
                    axis_num = {"X":0,"Y":1,"Z":2}[plane_axis]
                    plane_val = verts[:,axis_num].min() + (np.ptp(verts[:,axis_num])) * plane_pos/100
                    faces_keep = np.all(verts[faces][:,:,axis_num] >= plane_val, axis=1)
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
            fig.update_layout(scene_aspectmode="data", margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, use_container_width=True)
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


