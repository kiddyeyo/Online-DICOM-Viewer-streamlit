from nicegui import ui, app
import numpy as np
import os, tempfile
import pydicom, nibabel as nib
from skimage.measure import marching_cubes
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Global state
volume = None
axis = 0
vmin, vmax = 0, 1
original_polydata = None
current_polydata = None
plane1 = vtk.vtkPlane()
plane2 = vtk.vtkPlane()
dual_mode = False

# UI callbacks

def change_orientation(value):
    global axis
    axis = {'Axial': 0, 'Coronal': 1, 'Sagital': 2}[value]
    if volume is not None:
        ui.get('slice_slider').props(f'min=0 max={volume.shape[axis]-1} value={volume.shape[axis]//2}')
        update_image()


def load_volume(files):
    global volume, axis, vmin, vmax
    paths = []
    for f in files:
        tmp = os.path.join(tempfile.gettempdir(), f.name)
        with open(tmp, 'wb') as out:
            out.write(f.content)
        paths.append(tmp)
    # Detect NIfTI vs DICOM
    if len(paths) == 1 and paths[0].lower().endswith(('.nii', '.nii.gz')):
        vol = nib.load(paths[0]).get_fdata()
    else:
        dcm_files = [p for p in paths if p.lower().endswith('.dcm')]
        slices = [pydicom.dcmread(p) for p in sorted(dcm_files)]
        slices.sort(key=lambda s: float(getattr(s, 'ImagePositionPatient', [0,0,0])[2]))
        vol = np.stack([s.pixel_array for s in slices])
    volume = vol.astype(np.float32)
    # Window defaults
    vmin, vmax = np.percentile(volume, [1, 99])
    center = (vmax + vmin) / 2
    width = max(vmax - vmin, 1)
    ui.get('slice_slider').props(f'min=0 max={volume.shape[axis]-1} value={volume.shape[axis]//2}')
    ui.get('wc_slider').props(f'min={int(vmin)} max={int(vmax)} value={int(center)}')
    ui.get('ww_slider').props(f'min=1 max={int(width)} value={int(width)}')
    ui.get('thr_slider').props(f'min={int(vmin)} max={int(vmax)} value={int(center)}')
    update_image()


def update_image():
    global volume, axis
    if volume is None:
        return
    sl = ui.get('slice_slider').value
    if axis == 0:
        img = volume[sl]
    elif axis == 1:
        img = volume[:, sl]
    else:
        img = volume[:, :, sl]
    c = ui.get('wc_slider').value
    w = max(ui.get('ww_slider').value, 1)
    mn, mx = c - w/2, c + w/2
    imgw = np.clip(img, mn, mx)
    disp = ((imgw - mn) / w * 255).astype(np.uint8)
    thr = ui.get('thr_slider').value
    mask = img > thr
    # Render with matplotlib into base64
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(disp, cmap='gray')
    ax.imshow(np.ma.masked_where(~mask, mask), cmap='jet', alpha=0.3)
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    ui.get('img').update(f'data:image/png;base64,{data}')


def generate_stl():
    global original_polydata, current_polydata
    mask = (volume > ui.get('thr_slider').value).astype(np.uint8)
    verts, faces, _, _ = marching_cubes(mask, level=0)
    # Build VTK PolyData
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(verts))
    poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    for f in faces:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(int(f[0]))
        cells.InsertCellPoint(int(f[1]))
        cells.InsertCellPoint(int(f[2]))
    poly.SetPolys(cells)
    original_polydata = poly
    current_polydata = poly
    ui.notify('STL generado', color='positive')
    ui.run_javascript('loadModel();')


def apply_clip():
    global original_polydata, current_polydata
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(original_polydata)
    clip.SetClipFunction(plane1)
    clip.Update()
    original_polydata = clip.GetOutput()
    current_polydata = original_polydata
    ui.notify('Clipping aplicado', color='positive')
    ui.run_javascript('loadModel();')


def delete_between():
    global original_polydata, current_polydata
    clip1 = vtk.vtkClipPolyData()
    clip1.SetInputData(original_polydata)
    clip1.SetClipFunction(plane1)
    clip1.Update()
    clip2 = vtk.vtkClipPolyData()
    clip2.SetInputData(original_polydata)
    clip2.SetClipFunction(plane2)
    clip2.InsideOutOn()
    clip2.Update()
    appender = vtk.vtkAppendPolyData()
    appender.AddInputData(clip1.GetOutput())
    appender.AddInputData(clip2.GetOutput())
    appender.Update()
    original_polydata = appender.GetOutput()
    current_polydata = original_polydata
    ui.notify('Entre planos eliminado', color='positive')
    ui.run_javascript('loadModel();')


def export_stl():
    global current_polydata
    fn = os.path.join(tempfile.gettempdir(), 'export.stl')
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(fn)
    writer.SetInputData(current_polydata)
    writer.Write()
    return ui.download(fn, 'model.stl')

# FastAPI endpoint for fetching mesh
@app.get('/model')
def get_model():
    verts = vtk_to_numpy(current_polydata.GetPoints().GetData()).tolist()
    faces = []
    polys = current_polydata.GetPolys()
    polys.InitTraversal()
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 3:
            faces.append([idList.GetId(i) for i in range(3)])
    return {'vertices': verts, 'faces': faces}

# Build UI
with ui.row():
    with ui.column().style('width:50%'):
        ui.upload(load_volume).props('label="Cargar DICOM/NIfTI" multiple accept=".dcm,.nii,.nii.gz"')
        ui.select(['Axial','Coronal','Sagital'], label='Orientación', value='Axial', on_change=lambda e: change_orientation(e.value))
        ui.slider(label='Slice', id='slice_slider', on_change=lambda _: update_image())
        ui.slider(label='Window Center', id='wc_slider', on_change=lambda _: update_image())
        ui.slider(label='Window Width', id='ww_slider', on_change=lambda _: update_image())
        ui.slider(label='Threshold', id='thr_slider', on_change=lambda _: update_image())
        ui.image(id='img')
        ui.button('Generate STL', on_click=generate_stl)
    with ui.column().style('width:50%'):
        ui.html('''
<div id="viewer" style="width:100%; height:600px;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
var scene, camera, renderer, modelMesh;
function init() {
    const viewer = document.getElementById('viewer');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, viewer.clientWidth/viewer.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    viewer.appendChild(renderer.domElement);
    var light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(0,1,1).normalize(); scene.add(light);
    camera.position.z = 100;
    loadModel(); animate();
}
function loadModel() {
    fetch('/model').then(res=>res.json()).then(data=>{
        if (modelMesh) scene.remove(modelMesh);
        var geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices.flat(), 3));
        geometry.setIndex(data.faces.flat()); geometry.computeVertexNormals();
        var material = new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, clippingPlanes: [] });
        modelMesh = new THREE.Mesh(geometry, material); scene.add(modelMesh);
    });
}
function animate() { requestAnimationFrame(animate); renderer.render(scene, camera); }
init();
</script>
''')
        ui.button('Enable Clip', on_click=lambda: ui.notify('Preview clip not implemented'))
        ui.button('Apply Clip', on_click=apply_clip)
        ui.button('Dual Planes', on_click=lambda: ui.notify('Dual preview not implemented'))
        ui.button('Delete Between', on_click=delete_between)
        ui.button('Export STL', on_click=export_stl)

ui.run()
