# Explicación técnica de la aplicación

La aplicación `app.py` implementa un visor de volúmenes DICOM y NIfTI en Streamlit. A continuación se describen los bloques más relevantes del código.

## Importaciones y utilidades iniciales
El script importa librerías para manejo de datos médicos (`pydicom`, `nibabel`), procesamiento de imágenes (`scikit-image`, `scipy`), visualización (`plotly`) y manejo de mallas 3D (`trimesh`).

Se definen funciones auxiliares:

- `bounding_box(mask)` calcula la región mínima que contiene todos los valores verdaderos de una máscara 3D.
- `largest_connected_region(mask)` obtiene la región conectada más grande dentro de una máscara binaria.
- `load_dicom_series(zip_file)` extrae un ZIP con archivos DICOM y genera un volumen ordenado por la posición de cada corte.
- `load_nifti(nifti_file)` carga un archivo NIfTI y devuelve su volumen.
- `volume_stats(vol)` estima el rango de intensidades útil para ajustar brillo y contraste en pantalla.

Estas funciones usan `@st.cache_data` para que los resultados se almacenen en caché y acelerar la interacción.

## Configuración de la página
Se configura la página de Streamlit con `st.set_page_config` y se ocultan los elementos de interfaz predeterminados mediante CSS incrustado. También se define el título principal de la aplicación.

## Carga del volumen y controles 2D
En la barra lateral se encuentran los controles principales:

1. **Subida de archivos**: se acepta un ZIP con archivos DICOM o un `.nii/.nii.gz`.
2. **Orientación de visualización**: axial, coronal o sagital.
3. Al presionar *Cargar volumen* se lee el archivo y se guarda el volumen en `st.session_state`.

Se calcula el rango de intensidades y se muestran deslizadores para elegir corte, brillo y contraste. También se permite fijar un umbral de visualización mediante presets (hueso, pulmones, tejido blando, tumor) o valores personalizados.

## Vista 2D
Dependiendo de la orientación seleccionada se extrae el corte correspondiente del volumen. Se aplican las ventanas de brillo y contraste para obtener la imagen en escala de grises y se genera una máscara binaria según el umbral. Se muestran dos imágenes lado a lado: el corte original y el corte con la máscara sobrepuesta en rojo.

## Generación y edición de la malla 3D
En la pestaña **Vista 3D y Edición STL** se dispone de controles para crear y modificar una malla 3D basada en el volumen:

1. **Generar/Actualizar STL**: crea una máscara 3D a partir del rango de intensidades elegido. Se aplican operaciones morfológicas (cierre, apertura, eliminación de objetos pequeños) y se selecciona la región conectada más grande. Luego se calcula la superficie con `marching_cubes`, se construye una malla con `trimesh` y se suaviza. Los vértices y caras resultantes se guardan en `st.session_state`.
2. **Corte por plano**: si existe una malla, se puede definir un plano de corte (eje, posición y dirección). Al aplicar el recorte se eliminan las caras de la malla a un lado del plano y se guarda una versión recortada.

## Visualización interactiva en 3D
El panel de vista muestra la malla activa (recortada o completa) utilizando `plotly`. Si hay controles de recorte visibles, se dibuja también el plano de corte en color rojo como vista previa. La gráfica permite rotar, hacer zoom y desplazar la malla.

## Exportación de STL
Finalmente, es posible exportar la malla mostrada a un archivo STL descargable. Se usa `tempfile` para guardar temporalmente el archivo y `st.download_button` para ofrecerlo al usuario.

En ausencia de un volumen cargado, la aplicación indica que es necesario subir un archivo DICOM o NIfTI para comenzar.

