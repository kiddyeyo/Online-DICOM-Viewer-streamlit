import streamlit as st
import numpy as np
import os, zipfile, tempfile
import pydicom, nibabel as nib
from skimage.measure import marching_cubes
import pyvista as pv
import trimesh

st.set_page_config(layout="wide")

# --- Cargar datos ---

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

# --- Interfaz ---

st.title("Visor/Editor CT 2D+3D STL (Streamlit)")

with st.sidebar:
    st.header("Carga de volumen")
    uploaded = st.file_uploader("Sube un ZIP de DICOM o un archivo NIfTI", type=["zip","nii","nii.gz"])
    axis = st.selectbox("Orientación", ["Axial", "Coronal", "Sagital"])
    orientation_map = {'Axial': 0, 'Coronal': 1, 'Sagital': 2}
    axis = orientation_map[axis]
    process = st.button("Cargar volumen")

if 'volume' not in st.session_state:
    st.session_state.volume = None

if uploaded and process:
    if uploaded.name.lower().endswith('.zip'):
        st.session_state.volume = load_dicom_series(uploaded)
    else:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
        temp_file.write(uploaded.read())
        temp_file.close()
        st.session_state.volume = load_nifti(temp_file.name)
        os.unlink(temp_file.name)

vol = st.session_state.volume

# --- 2D View ---
if vol is not None:
    vmin, vmax = np.percentile(vol, [1,99])
    center = (vmin+vmax)/2
    width = max(vmax-vmin, 1)

    slice_max = vol.shape[axis] - 1
    slice_idx = st.sidebar.slider("Slice", 0, slice_max, slice_max//2)
    wc = st.sidebar.slider("Window Center", int(vmin), int(vmax), int(center))
    ww = st.sidebar.slider("Window Width", 1, int(width), int(width))
    thr = st.sidebar.slider("Threshold", int(vmin), int(vmax), int(center))

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
    st.image(np.concatenate([disp[..., None]]*3, axis=-1) * 0.7 + overlay * 0.3, width=400)

    # --- STL Generación y visualización 3D ---
    st.header("Vista 3D y edición STL")
    gen_stl = st.button("Generar STL")
    if gen_stl or "mesh" in st.session_state:
        mask3d = (vol > thr).astype(np.uint8)
        verts, faces, _, _ = marching_cubes(mask3d, level=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        st.session_state.mesh = mesh

        # Visualización 3D (pyvista+plotly)
        cloud = pv.PolyData(verts)
        surf = cloud.delaunay_3d().extract_geometry()
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(surf, color="lightgray")
        pl.camera_position = "xy"
        img_arr = pl.screenshot(None, return_img=True)
        st.image(img_arr, caption="Vista 3D STL (preview)", width=400)
        st.info("Puedes exportar la malla STL resultante debajo.")

        # Edición sencilla: Clipping plane
        st.subheader("Edición (Recorte por plano único)")
        plane_axis = st.selectbox("Eje plano de corte", ["X","Y","Z"])
        plane_pos = st.slider("Posición del plano (%)", 0, 100, 50)
        axis_num = {"X":0,"Y":1,"Z":2}[plane_axis]
        if st.button("Aplicar recorte plano"):
            plane_val = verts[:,axis_num].min() + (verts[:,axis_num].ptp()) * plane_pos/100
            faces_keep = np.all(verts[faces][:,:,axis_num] >= plane_val, axis=1)
            faces_clip = faces[faces_keep]
            mesh_clip = trimesh.Trimesh(vertices=verts, faces=faces_clip)
            st.session_state.mesh = mesh_clip
            st.success("Recorte aplicado.")
        if st.button("Exportar STL"):
            mesh_to_export = st.session_state.mesh
            export_path = os.path.join(tempfile.gettempdir(), "mesh_export.stl")
            mesh_to_export.export(export_path)
            with open(export_path, "rb") as f:
                st.download_button("Descargar STL", f, file_name="mesh_export.stl", mime="application/sla")

else:
    st.info("Carga un volumen DICOM (.zip de .dcm) o NIfTI (.nii/.nii.gz) para empezar.")

