import shutil
import os

def delete_iddaw_gtsegpng(base_path):
    weather_conditions = ['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']
    for condition in weather_conditions:
        folder = os.path.join(base_path, condition, 'gtSegPNG')
        if os.path.exists(folder):
            print(f"Deleting: {folder}")
            shutil.rmtree(folder)
        else:
            print(f"Not found (skipped): {folder}")

delete_iddaw_gtsegpng("data/IDD/IDDAW/train")
