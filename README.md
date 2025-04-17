# LuCam Python Image Acquisition Software

---

## Description

A Python application for **acquiring and processiong images from Lumenera cameras** using the LuCam API. Features a graphical user interface.

---

## Authors

- [**Bautista Salvatierra Pérez**](https://github.com/bautisalva)
- [**Tomás Rodríguez Bouhier**](https://github.com/totorod1120)

---

## Requirements

Install the following Python packages:

```bash
pip install numpy opencv-python PyQt6
```

- `numpy`
- `opencv-python`
- `PyQt6`
- (plus [Lumenera USB camera and drivers 5.0](https://www.lumenera.com/))

You can also install all at once with:

```bash
pip install -r requirements.txt
```
---
## Missing
- tener cuidado con la branch de toto porque saca los botones separados de guardar parametros de preview y captura. Fijarse de agregar la branch de toto con el offset sin borrar esos botones (y que ningun otro cambio borre algo)
- manejo del guardado de fotos (carpetas, guardar el raw de las fotos y eso, etc.)
- revisar el tema del ‘toggle_background’ que parece que anda mal el botón
---

## 📝 Notes

- If the `lucam` module is not found (camera disconnected or driver missing), the application automatically **falls back to a simulated camera** for testing.
- More notes

---

## 🔗 References

- [`lucam.py`](https://github.com/cgohlke/lucam) — Python wrapper for the LuCam API, by [Christoph Gohlke](https://github.com/cgohlke).
- Lumenera USB Camera API Reference Manual Release 5.0. Lumenera Corporation.

---
![image](https://github.com/user-attachments/assets/982437cf-5599-43d9-a4dd-87b9221eee4f)
