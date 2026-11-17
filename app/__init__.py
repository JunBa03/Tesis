"""Aplicación de demostración del sistema de análisis de radiografías de tórax.

La demostración está pensada como base sobre la que seguir construyendo, no como
una pantalla de una sola vez. Por eso el motor (``app.core``) no sabe nada de la
interfaz: recibe imágenes y modelos y devuelve estructuras de datos. La capa de
Streamlit (``app.pages``) es una vista delgada sobre ese motor, y mañana podría
reemplazarse por una API o una web sin tocar la lógica.
"""
