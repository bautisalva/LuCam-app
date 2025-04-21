# Lumanera Camera Python Image Acquisition Software

---

## Description

A Python application for **acquiring and processiong images from Lumenera cameras** using the LuCam API by [Christoph Gohlke](https://github.com/cgohlke). Features a graphical user interface.

---

## Authors

- [**Bautista Salvatierra Pérez**](https://github.com/bautisalva)
- [**Tomás Rodríguez Bouhier**](https://github.com/totorod1120)

---

## Requirements

- Python ≥ 3.8
- [NumPy](https://numpy.org/)
- [OpenCV (cv2)](https://opencv.org/)
- [PyQt6](https://pypi.org/project/PyQt6/)
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
- asegurarse de capturar y procesar en 16bits
- Arreglar el problema de las rayas en las imagenes
- Profundizar el analisis de imagenes (ver [Link](https://biapol.github.io/Image-data-science-with-Python-and-Napari-EPFL2022/day2c_Image_Filters/09_Filters.html))
  - noise reduction para capturar las imagenes: moving average, gaussian, median
  - mejorar el contraste de la imagen restada: filtrar con una tangente, rolling ball algorithm, tophat filter, otro gaussian blur o gaussian difference
    
---


## 🔗 References

- [`lucam.py`](https://github.com/cgohlke/lucam) — Python wrapper for the LuCam API, by [Christoph Gohlke](https://github.com/cgohlke).
- Lumenera USB Camera API Reference Manual Release 5.0. Lumenera Corporation.

---
![image](https://github.com/user-attachments/assets/982437cf-5599-43d9-a4dd-87b9221eee4f)
