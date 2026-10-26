# 1. Usar una imagen base ligera de Python (versión fija como pide la rúbrica)
FROM python:3.11-slim

# 2. Crear un usuario no root por seguridad y buenas prácticas
RUN useradd -m appuser

# 3. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiar e instalar las dependencias primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código del proyecto
COPY src/ ./src/
COPY prompts/ ./prompts/

# 6. Cambiar los permisos de la carpeta al usuario no root
RUN chown -R appuser:appuser /app
USER appuser

# 7. Comando para ejecutar el script apuntando al Ollama de la máquina host
ENTRYPOINT ["python", "src/inference.py", "--model", "mistral:7b-instruct-v0.3-q4_K_M"]
