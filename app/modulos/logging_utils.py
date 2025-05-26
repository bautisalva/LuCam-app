import datetime
import os

class Logger:
    def __init__(self, filename="log.txt"):
        self.path = os.path.join(os.getcwd(), filename)
        self.archivo = open(self.path, "a", encoding="utf-8")
        now = datetime.datetime.now()
        self.log(f"=== Inicio {now.strftime('%d/%m/%Y %H:%M:%S')} ===")

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        texto = f"[{timestamp}] {msg}"
        print(texto)
        self.archivo.write(texto + "\n")
        self.archivo.flush()
        return texto

    def close(self):
        self.archivo.close()

