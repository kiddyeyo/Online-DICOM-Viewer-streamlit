# Dockerfile para CT Scan Viewer Web con NiceGUI
FROM python:3.10-slim

# Evitar la creación de archivos pycache y buffer
ENV PYTHONUNBUFFERED=1
ENV COMPOSE_BAKE=true
# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para VTK y otras
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgl1 \
       libglib2.0-0 \
       libxrender1 \
       libxext6 \
       libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar dependencias de Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app.py ./

# Exponer el puerto por defecto de NiceGUI
EXPOSE 8080

# Arrancar la aplicación
CMD ["python", "app.py"]