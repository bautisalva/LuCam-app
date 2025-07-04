# Lumanera Camera Python Image Acquisition Software

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
![image](https://github.com/user-attachments/assets/982437cf-5599-43d9-a4dd-87b9221eee4f)
![Bin-P8139-190Oe-30ms-5Tw-97](https://github.com/user-attachments/assets/3f6d4692-93f4-4548-891b-496643e3728a)
![image](https://github.com/user-attachments/assets/0f8721ff-4443-429b-ab2b-e989d0a0b7f2)



