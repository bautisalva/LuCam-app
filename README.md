# Lumanera Camera Python Image Acquisition Software
LuCam-app
A modular Python application for image acquisition and analysis with Lumenera USB cameras. It offers a cross-platform PyQt5 interface for live preview, high-quality capture, and magnetic-domain edge analysis, with simulation and logging built in.

Features
Live preview with adjustable camera properties (FPS, brightness, contrast, etc.) and parameter persistence via JSON files

Image capture workspace supporting ROI selection, background subtraction, blur, averaging or median filtering, and automatic exports for raw/processed images

Edge-analysis tab powered by a custom ImageEnhancer that subtracts background, enhances contrast, binarizes, and extracts contours

Simulation mode when hardware is absent, generating synthetic noisy frames labeled “SIMULATED”

Threaded workers for non-blocking preview and capture operations

ROI selection widget drawing rectangles on the preview to define capture regions

Centralized logging to both GUI consoles and log.txt

Project Structure
LuCam-app/
│
├── app/
│   ├── modular/          # Main application modules
│   │   ├── main.py       # Entry point
│   │   ├── camera_app.py # Top-level GUI with tabbed interface
│   │   ├── common.py     # Workers, simulated camera, ROI label
│   │   ├── tabs/
│   │   │   ├── preview_tab.py
│   │   │   ├── capture_tab.py
│   │   │   └── analysis_tab.py
│   │   ├── utils.py
│   │   └── lucam.py      # LuCam API wrapper by Christoph Gohlke
│   ├── beta/             # Earlier monolithic prototype & ImageEnhancer
│   └── toto/             # Experimental scripts
│
├── analisis/             # Standalone analysis scripts (e.g., velocity fields)
├── params/               # Sample JSON configs for preview/capture
├── docs/                 # License and Teledyne API manual
└── requirements.txt
Installation
git clone https://github.com/bautisalva/LuCam-app.git
cd LuCam-app
pip install -r requirements.txt
Requirements

Python ≥ 3.8

NumPy

PyQt5

scikit-image

Lumenera USB camera with drivers 5.0 for hardware mode

Usage
Launch the GUI:

python app/modular/main.py
When a Lumenera camera is detected, it runs in hardware mode; otherwise it falls back to a simulated camera

A log file log.txt is created automatically and messages from all tabs are aggregated

Tabs Overview
Preview
Adjust FPS and sensor properties, save/load settings, and refresh from the camera in real time

Capture
Capture bursts of frames with averaging or median modes, apply Gaussian blur, enable/disable background subtraction, select ROI with the mouse, and auto-save multiple image variants (raw, blurred, background-subtracted) to a working directory

Analysis
Run edge-detection analysis on the last captured or loaded image. Parameters include smoothing, contour percentile, peak distance, and contour method (Sobel vs. binarization). Results (binary image and contour coordinates) can be exported to disk

Configuration
Use “Save parameters” buttons in the Preview and Capture tabs to export current settings to JSON; load them later to reproduce experiments. Example files live in params/.

ROI coordinates, blur strength, background gain/offset, capture mode, and other options are all persisted through these files

Additional Analysis Scripts
The analisis/ directory contains standalone scripts for deeper processing:

campo_velocidades.py – interpolates contours over time to compute normal and tangential velocities

desplazamiento_area_perimetro.py, contraste.py, bordes.py, etc., provide complementary metrics and visualization tools for magnetic domain research.

References
The bundled lucam.py is the LuCam API wrapper by Christoph Gohlke (BSD license)

Teledyne Lumenera USB Camera API Reference Manual (see docs/Teledyne_Lumenera-USB_Camera-API_Reference_Manual.pdf)

License
Released under the MIT License. See docs/LICENSE for details.

Authors
Bautista Salvatierra Pérez

Tomás Rodríguez Bouhier
---

## Description

A Python application for **acquiring and processiong images from Lumenera cameras** to study magentic domains on thin films using the LuCam API by [Christoph Gohlke](https://github.com/cgohlke).

---

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



