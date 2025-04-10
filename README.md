# LuCam Python Image Acquisition Software

---

## 📜 Description

A Python application for **acquiring images from Lumenera cameras** using the LuCam API.  
Features a graphical user interface for live preview, image capture (average, median modes), background subtraction, parameter control, and file saving.

---

## 👥 Authors

- **Bautista Salvatierra**
- (Add other collaborators here)

---

## ⚙️ Requirements

Install the following Python packages:

```bash
pip install numpy opencv-python PyQt6
```

- `numpy`
- `opencv-python`
- `PyQt6`
- (plus Lumenera drivers installed on the system)

You can also install all at once with:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- If the `lucam` module is not found (camera disconnected or driver missing), the application automatically **falls back to a simulated camera** for testing.
- Save settings and images are managed via standard dialogs (JSON, TIFF formats).
- This project was tested using Python 3.11.

---

## 🔗 References

- [`lucam.py`](https://github.com/cgohlke/lucam) — Python wrapper for the LuCam API, by Christoph Gohlke.
- [Teledyne Lumenera Official Camera Manual](https://www.teledynelumenera.com/) — Manufacturer's documentation.

---
