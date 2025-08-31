# Lumanera Camera Python Image Acquisition Software

A Python application for **acquiring and processiong images from Lumenera cameras** to study magentic domains on thin films using the LuCam API by [Christoph Gohlke](https://github.com/cgohlke).

---
## Features
- Live preview with adjustable camera properties (FPS, brightness, contrast, etc.) and parameter persistence via JSON files
- Image capture workspace supporting ROI selection, background subtraction, blur, averaging or median filtering, and automatic exports for raw/processed images
- Edge‑analysis tab powered by a custom ImageEnhancer that subtracts background, enhances contrast, binarizes, and extracts contours
- Simulation mode when hardware is absent, generating synthetic noisy frames labeled “SIMULATED”
- Threaded workers for non‑blocking preview and capture operations
- ROI selection widget drawing rectangles on the preview to define capture regions
- Centralized logging to both GUI consoles and log.txt

## Authors

- [**Bautista Salvatierra Pérez**](https://github.com/bautisalva)
- [**Tomás Rodríguez Bouhier**](https://github.com/totorod1120)

---

## Requirements

- Python ≥ 3.8
- [NumPy](https://numpy.org/)
- [scikit-image](https://scikit-image.org/)
- [PyQt5](https://pypi.org/project/PyQt6/)
- (plus [Lumenera USB camera and drivers 5.0](https://www.lumenera.com/))

You can also install all at once with:

```bash
pip install -r requirements.txt
```
---
## Usage
```bash
python test.py
```
---

## Missing
- el script 'rayas.py' arregla el problema de las rayas en las imagenes restando la media por fila sobre cada imagen pero esto no queremos hacerlo pues estaríamos perdiendo resolución en ls bordes de los dominio. Necesitamos trabajar en mejorar el contraste de la camara 'desde el setup' mejorando el enfoque y la alineación de los instrumentos
- agregrar un mini histograma en la secion de 'captura'  y tambien uno mas grande en el sector analisis
- ver que hacemos con las manchas cuando quedan en el borde del dominio, las guardamos? inventamos los puntos despues? que hacemos con eso
- incluir mejor la pestaña 'análisis'
- arreglar error al capturar fondo que muestra la imagen 'recortada'
---


## 🔗 References

- [`lucam.py`](https://github.com/cgohlke/lucam) — Python wrapper for the LuCam API, by [Christoph Gohlke](https://github.com/cgohlke).
- Lumenera USB Camera API Reference Manual Release 5.0. Lumenera Corporation.

---



