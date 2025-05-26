import json

def guardar_parametros(filepath, params):
    try:
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
        return True, f"Parámetros guardados en: {filepath}"
    except Exception as e:
        return False, f"[ERROR] No se pudo guardar: {e}"

def cargar_parametros(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"[ERROR] No se pudo cargar: {e}"
