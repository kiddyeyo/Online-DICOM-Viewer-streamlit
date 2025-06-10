import sys
import os
import tempfile
import numpy as np
import pydicom
import nibabel as nib
from skimage.measure import marching_cubes
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.numpy_support import numpy_to_vtk

class CTSTLEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT 2D Viewer + 3D STL Editor")
        self.volume = None
        self.axis = 0
        self.window_center = 0
        self.window_width = 1
        self.mesh_polydata = None
        self.original_polydata = None
        self.clipped_polydata = None
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: 2D Matplotlib viewer
        left = QWidget()
        left_layout = QVBoxLayout(left)
        main_layout.addWidget(left, 1)
        # Canvas
        self.canvas = FigureCanvas(Figure(figsize=(4,4)))
        self.ax = self.canvas.figure.subplots()
        left_layout.addWidget(self.canvas)
        # Controls
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        left_layout.addWidget(ctrl)
        # Input
        in_btn = QPushButton("Cargar DICOM/NIfTI")
        in_btn.clicked.connect(self.load_volume)
        ctrl_layout.addWidget(in_btn)
        # Orientation
        ori_layout = QHBoxLayout()
        ori_layout.addWidget(QLabel("Orientación:"))
        self.orient_cb = QComboBox()
        self.orient_cb.addItems(["Axial","Coronal","Sagital"])
        self.orient_cb.currentIndexChanged.connect(self.change_orientation)
        ori_layout.addWidget(self.orient_cb)
        ctrl_layout.addLayout(ori_layout)
        # Slice slider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.update_image)
        ctrl_layout.addWidget(QLabel("Slice"))
        ctrl_layout.addWidget(self.slice_slider)
        # WL sliders
        self.wc_slider = QSlider(Qt.Horizontal)
        self.wc_slider.valueChanged.connect(self.update_image)
        ctrl_layout.addWidget(QLabel("Window Center"))
        ctrl_layout.addWidget(self.wc_slider)
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.valueChanged.connect(self.update_image)
        ctrl_layout.addWidget(QLabel("Window Width"))
        ctrl_layout.addWidget(self.ww_slider)
        # Threshold slider
        self.thr_slider = QSlider(Qt.Horizontal)
        ctrl_layout.addWidget(QLabel("Threshold"))
        ctrl_layout.addWidget(self.thr_slider)
        # Generate STL
        stl_btn = QPushButton("Generate STL")
        stl_btn.clicked.connect(self.generate_stl)
        ctrl_layout.addWidget(stl_btn)

        # Right: VTK 3D Editor
        right = QWidget()
        right_layout = QVBoxLayout(right)
        main_layout.addWidget(right, 1)
        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(right)
        right_layout.addWidget(self.vtk_widget)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()
        # Buttons for clipping and export
        clip_layout = QHBoxLayout()
        self.enable_clip = QPushButton("Enable Clip")
        self.enable_clip.clicked.connect(self.toggle_clip_plane)
        clip_layout.addWidget(self.enable_clip)
        self.apply_clip = QPushButton("Apply Clip")
        self.apply_clip.setEnabled(False)
        self.apply_clip.clicked.connect(self.apply_clipping)
        clip_layout.addWidget(self.apply_clip)
        right_layout.addLayout(clip_layout)

        dual_layout = QHBoxLayout()
        self.enable_dual = QPushButton("Dual Planes")
        self.enable_dual.clicked.connect(self.toggle_dual_planes)
        dual_layout.addWidget(self.enable_dual)
        self.delete_between = QPushButton("Delete Between")
        self.delete_between.setEnabled(False)
        self.delete_between.clicked.connect(self.delete_between_planes)
        dual_layout.addWidget(self.delete_between)
        right_layout.addLayout(dual_layout)

        export_btn = QPushButton("Export STL")
        export_btn.clicked.connect(self.export_stl)
        right_layout.addWidget(export_btn)

        self.plane_widget = None
        self.plane_widget2 = None
        self.plane1 = vtk.vtkPlane()
        self.plane2 = vtk.vtkPlane()
        self.dual_mode = False
        self.clipping_active = False
        self.resize(1400, 800)

    def load_volume(self):
        path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder or NIfTI File")
        if not path:
            file, _ = QFileDialog.getOpenFileName(self, "Select NIfTI", filter="NIfTI (*.nii *.nii.gz)")
            path = file or None
        if not path: return
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path) if f.lower().endswith('.dcm'))
            slices = [pydicom.dcmread(os.path.join(path, f)) for f in files]
            slices.sort(key=lambda s: float(getattr(s,'ImagePositionPatient',[0,0,0])[2]))
            vol = np.stack([s.pixel_array for s in slices])
        else:
            vol = nib.load(path).get_fdata()
        self.volume = vol
        vmin, vmax = np.percentile(vol, [1,99])
        center = (vmax+vmin)/2; width = max(vmax-vmin,1)
        # configure sliders
        self.slice_slider.setRange(0, vol.shape[self.axis]-1)
        self.slice_slider.setValue(vol.shape[self.axis]//2)
        self.wc_slider.setRange(int(vmin),int(vmax)); self.wc_slider.setValue(int(center))
        self.ww_slider.setRange(1,int(width)); self.ww_slider.setValue(int(width))
        self.thr_slider.setRange(int(vmin),int(vmax)); self.thr_slider.setValue(int(center))
        self.update_image()

    def change_orientation(self, index):
        """Update viewing axis when orientation combo is changed."""
        self.axis = index
        if self.volume is not None:
            self.slice_slider.setRange(0, self.volume.shape[self.axis]-1)
            self.slice_slider.setValue(self.volume.shape[self.axis]//2)
        self.update_image()

    def update_image(self):
        if self.volume is None: return
        sl = self.slice_slider.value()
        if self.axis == 0: img = self.volume[sl]
        elif self.axis == 1: img = self.volume[:,sl]
        else: img = self.volume[:,:,sl]
        c = self.wc_slider.value(); w = max(self.ww_slider.value(),1)
        mn = c - w/2; mx = c + w/2
        imgw = np.clip(img, mn, mx)
        disp = ((imgw - mn)/w*255).astype(np.uint8)
        thr = self.thr_slider.value()
        mask = img > thr
        self.ax.clear(); self.ax.imshow(disp,cmap='gray')
        self.ax.imshow(np.ma.masked_where(~mask,mask),cmap='jet',alpha=0.3)
        self.ax.axis('off'); self.canvas.draw()

    def generate_stl(self):
        if self.volume is None: return
        mask = (self.volume > self.thr_slider.value()).astype(np.uint8)
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
        self.original_polydata = poly
        self.mapper = vtk.vtkPolyDataMapper(); self.mapper.SetInputData(poly)
        if hasattr(self,'actor'): self.renderer.RemoveActor(self.actor)
        self.actor = vtk.vtkActor(); self.actor.SetMapper(self.mapper)
        self.renderer.AddActor(self.actor); self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def create_plane_widget(self, plane):
        """Create an implicit plane widget bound to *plane* and return it."""
        w = vtk.vtkImplicitPlaneWidget()
        w.SetInteractor(self.interactor)
        w.SetPlaceFactor(1.0)
        w.SetInputData(self.original_polydata)
        w.PlaceWidget()
        w.GetPlane(plane)
        w.AddObserver('InteractionEvent', lambda o, e: self.update_clip(o, plane))
        w.On()
        # Attach the clipping plane to the mapper so rendering reflects its
        # position without modifying the input polydata
        self.mapper.AddClippingPlane(plane)
        return w

    def toggle_clip_plane(self):
        if not hasattr(self,'actor'): return
        if not self.clipping_active:
            self.plane_widget = self.create_plane_widget(self.plane1)
            self.clipping_active = True
            self.apply_clip.setEnabled(True)
        else:
            if self.plane_widget:
                self.plane_widget.Off()
                self.plane_widget = None
            # Remove the plane from the mapper so the mesh is shown uncut
            self.mapper.RemoveAllClippingPlanes()
            self.clipping_active = False
            self.apply_clip.setEnabled(False)
            self.vtk_widget.GetRenderWindow().Render()

    def update_clip(self, widget, plane):
        """Update *plane* from *widget* and redraw the scene."""
        widget.GetPlane(plane)
        self.vtk_widget.GetRenderWindow().Render()

    def apply_clipping(self):
        """Permanently apply the current clipping plane to the mesh."""
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(self.original_polydata)
        clip.SetClipFunction(self.plane1)
        clip.Update()
        self.original_polydata = clip.GetOutput()
        self.mapper.SetInputData(self.original_polydata)
        # Remove widget and associated clipping plane from the mapper
        if self.plane_widget:
            self.plane_widget.Off()
            self.plane_widget = None
        self.mapper.RemoveAllClippingPlanes()
        self.apply_clip.setEnabled(False)
        self.clipping_active = False
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_dual_planes(self):
        if not hasattr(self,'actor'): return
        if not self.dual_mode:
            self.plane_widget = self.create_plane_widget(self.plane1)
            self.plane_widget2 = self.create_plane_widget(self.plane2)
            self.dual_mode = True
            self.delete_between.setEnabled(True)
        else:
            if self.plane_widget:
                self.plane_widget.Off()
                self.plane_widget = None
            if self.plane_widget2:
                self.plane_widget2.Off()
                self.plane_widget2 = None
            self.dual_mode = False
            self.delete_between.setEnabled(False)
            # Remove planes from the mapper so clipping is disabled
            self.mapper.RemoveAllClippingPlanes()
            self.vtk_widget.GetRenderWindow().Render()

    def delete_between_planes(self):
        clip1 = vtk.vtkClipPolyData()
        clip1.SetInputData(self.original_polydata)
        clip1.SetClipFunction(self.plane1)
        clip1.Update()
        clip2 = vtk.vtkClipPolyData()
        clip2.SetInputData(self.original_polydata)
        clip2.SetClipFunction(self.plane2)
        clip2.InsideOutOn()
        clip2.Update()
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(clip1.GetOutput())
        appender.AddInputData(clip2.GetOutput())
        appender.Update()
        self.original_polydata = appender.GetOutput()
        self.mapper.SetInputData(self.original_polydata)
        if self.plane_widget:
            self.plane_widget.Off()
            self.plane_widget = None
        if self.plane_widget2:
            self.plane_widget2.Off()
            self.plane_widget2 = None
        # Clear all clipping planes since the geometry now contains the result
        self.mapper.RemoveAllClippingPlanes()
        self.dual_mode = False
        self.delete_between.setEnabled(False)
        self.vtk_widget.GetRenderWindow().Render()

    def export_stl(self):
        if not hasattr(self,'actor'): return
        fn, _ = QFileDialog.getSaveFileName(self, 'Export STL', '', filter='STL (*.stl)')
        if fn:
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(fn)
            writer.SetInputData(self.mapper.GetInput())
            writer.Write()
            QMessageBox.information(self, 'Saved', f'STL saved to {fn}')

if __name__=='__main__':
    app = QApplication(sys.argv)
    win = CTSTLEditor()
    win.show()
    sys.exit(app.exec_())
