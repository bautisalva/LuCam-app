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
- [scikit-image](https://scikit-image.org/)
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
- Arreglar el problema de las rayas en las imagenes
    - ver si funciona apagando y prendiendo la camara
    - ver si funciona yendo linea por linea y restandole la media


- arreglar el problema en el guardado de imagenes que se guardan distinto con y sin ROI. La cruda tambien se ve distinta
- pasar el limpio todos los codigos de analisis de imagenes y documentar todo
- agregrar un mini histograma en la secion de 'captura'
- probar el analisis de distintas imagenes para ver si anda
- ver que hacemos con las manchas cuando quedan en el borde del dominio, las guardamos? inventamos los puntos despues? que hacemos con eso
- nos quedamos finalmente con la diferencia de tangentes con la gaussiana y la lineal
    - necesitamos agregar para que se identifiquen los maximos del histograma automaticamente asi no hay que ponerle a mano los parametros a la funcion
- al final no vamos a restar el fondo, asi que borrrar eso
- acordarse que todo el analisis esta hecho en 8bits pero tenemos que pasarlo todo a 16bits para las proximas imagenes
---


## 🔗 References

- [`lucam.py`](https://github.com/cgohlke/lucam) — Python wrapper for the LuCam API, by [Christoph Gohlke](https://github.com/cgohlke).
- Lumenera USB Camera API Reference Manual Release 5.0. Lumenera Corporation.

---
![image](https://github.com/user-attachments/assets/982437cf-5599-43d9-a4dd-87b9221eee4f)
