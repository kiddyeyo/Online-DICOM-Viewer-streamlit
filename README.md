# Online-DICOM-Viewer

Esta aplicación de Streamlit permite cargar series DICOM o archivos NIfTI y
visualizar cortes 2D y una malla 3D generada con *marching cubes*.  Incluye
herramientas básicas de clipping y exportación a STL.

### Mejoras recientes

- Se añadieron cachés para acelerar la carga de los volúmenes y el cálculo de
  estadísticas.
- El panel de opciones 3D ahora permite escoger la resolución de la malla
  (``step size``) para generar modelos más ligeros y fluidos.
- El visor 3D mantiene la cámara entre interacciones para evitar saltos
  molestos al actualizar.
