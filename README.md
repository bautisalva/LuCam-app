# Lumanera Camera Python Image Acquisition Software

---

## Description

A Python application for **acquiring and processiong images from Lumenera cameras** using the LuCam API. Features a graphical user interface.

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
## 🧑‍💻 Usage
```bash
python test.py
```
---

## 📊 Capture Modes

| Mode      | Description                          |
|-----------|--------------------------------------|
| Promedio  | Computes the pixel-wise average      |
| Mediana   | Computes the pixel-wise median       |

---

## 🧠 Background Subtraction

The app performs subtraction using:

    result = clip(a * (I - B + b) + 128, 0, 255)

Where:
- I : captured image  
- B : background image  
- a : gain (scaling factor)  
- b : offset (bias)

---
---
## Missing
- tener cuidado con la branch de toto porque saca los botones separados de guardar parametros de preview y captura. Fijarse de agregar la branch de toto con el offset sin borrar esos botones (y que ningun otro cambio borre algo)
- manejo del guardado de fotos (carpetas, guardar el raw de las fotos y eso, etc.)
- revisar el tema del ‘toggle_background’ que parece que anda mal el botón
---


## 🔗 References

- [`lucam.py`](https://github.com/cgohlke/lucam) — Python wrapper for the LuCam API, by [Christoph Gohlke](https://github.com/cgohlke).
- Lumenera USB Camera API Reference Manual Release 5.0. Lumenera Corporation.

---
![image](https://github.com/user-attachments/assets/982437cf-5599-43d9-a4dd-87b9221eee4f)
