import numpy as np
from PIL import Image, ImageDraw

class SimulatedFrameFormat:
    def __init__(self):
        self.pixelFormat = None
        self.width = 640
        self.height = 480
        self.xOffset = 0
        self.yOffset = 0

class SimulatedCamera:
    def TakeSnapshot(self):
        image = np.random.normal(32768, 5000, (480, 640)).astype(np.uint16)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((50, 240), "SIMULATED", fill=65535)
        return np.array(img_pil)

    def set_properties(self, **kwargs):
        pass

    def GetFormat(self):
        return SimulatedFrameFormat(), 15.0

    def SetFormat(self, frameformat, fps):
        pass

    def ContinuousAutoExposureDisable(self):
        pass

    def EnumAvailableFrameRates(self):
        return [7.5, 15.0, 30.0]


# Try to import Lucam
try:
    from lucam import Lucam, API
    LUCAM_AVAILABLE = True
except ImportError:
    LUCAM_AVAILABLE = False
    Lucam = None
    API = None
    print("[WARNING] Módulo 'lucam' no disponible, usando modo simulación.")


def init_camera():
    if LUCAM_AVAILABLE:
        try:
            cam = Lucam()
            cam.CameraClose()
            cam = Lucam()
            return cam, False
        except Exception as e:
            print(f"[ERROR] No se pudo iniciar Lucam: {e}")
    return SimulatedCamera(), True