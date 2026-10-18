"""Konfigurasi yang mengikuti BAB III proposal Ilsan Hidayatuloh."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'jakarta_house.csv'
OUTPUT_DIR = ROOT / 'outputs'
MODEL_DIR = ROOT / 'models'
TARGET = 'price'
CAT_COLS = ['district', 'city']
NUM_COLS = ['bed_rooms', 'bath_rooms', 'carport', 'land_area', 'building_area']
FEATURES = CAT_COLS + NUM_COLS
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# Per jenis transformasi target (raw dan log1p), per model ensemble.
# Naikkan ke 20 untuk eksplorasi lebih luas; proses akan lebih lama.
SEARCH_ITERATIONS = 8
SEARCH_JOBS = 2  # Cocok sebagai awal pada Mac M1; model memakai satu thread.

MODEL_LABELS = {
    'linear_regression': 'Regresi Linear Berganda',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
}
