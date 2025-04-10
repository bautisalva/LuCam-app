# LuCam Python Image Acquisition Software

Python application for **capturing images using Lumenera cameras** via the LuCam API.  
Includes GUI tools for live preview, capture modes (average, median), background subtraction, and parameter saving.

---

## 📂 Project Structure

```
LuCam-python-image-acquisition-software/
├── docs/
│   └── Teledyne-Camera-Manual.pdf  # Official camera manual
├── external/
│   └── lucam.py                    # LuCam API Python wrapper (from Christoph Gohlke)
├── src/
│   └── test.py                     # Main application with GUI for camera control
├── README.md
├── .gitignore
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/bautisalva/LuCam-python-image-acquisition-software.git
cd LuCam-python-image-acquisition-software
```

### 2. Install Requirements

(Recommended to use a virtual environment.)

```bash
pip install -r requirements.txt
```

**Dependencies include**:
- `numpy`
- `opencv-python`
- `PyQt6`

> You must also have the **Lumenera camera drivers** installed for `lucamapi.dll` to work.

### 3. Running the Application

```bash
cd src
python test.py
```

If the `lucam` module is not found (no camera connected), the app **falls back** to a simulated camera for testing.

---

## 📸 Features

- Live camera preview
- Capture images with **average** or **median** modes
- Adjustable **blur** and camera **properties** (brightness, contrast, etc.)
- Background capture and subtraction
- Save images as `.tif`
- Save and load settings (JSON format)

---

## 📜 Credits

- **lucam.py**: Python wrapper from [Christoph Gohlke's lucam project](https://github.com/cgohlke/lucam)
- **Teledyne PDF**: Official documentation for Lumenera camera usage

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
