from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal, Qt, QRect
from PyQt5.QtGui import QPainter, QPen

class ROILabel(QLabel):
    roi_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.start_pos = None
        self.end_pos = None
        self.drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.drawing = True
            self.end_pos = self.start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_pos = event.pos()
            self.update()

            x1 = min(self.start_pos.x(), self.end_pos.x())
            y1 = min(self.start_pos.y(), self.end_pos.y())
            x2 = max(self.start_pos.x(), self.end_pos.x())
            y2 = max(self.start_pos.y(), self.end_pos.y())

            width = x2 - x1
            height = y2 - y1

            self.roi_selected.emit(x1, y1, width, height)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing and self.start_pos and self.end_pos:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_pos, self.end_pos)
            painter.drawRect(rect)