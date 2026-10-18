"""Audit aktual dan pembersihan deterministik sebelum split; tidak mem-fit statistik."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CAT_COLS, FEATURES, NUM_COLS, TARGET


def read_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset tidak ditemukan: {path}. Salin jakarta_house.csv ke folder data/.')
    df = pd.read_csv(path, low_memory=False)
    normalized = [str(col).strip().lower() for col in df.columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError('Nama kolom menjadi duplikat setelah penyeragaman huruf.')
    df.columns = normalized
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f'Kolom wajib tidak ditemukan: {missing}. Kolom aktual: {list(df.columns)}')
    return df


def numeric_series(s: pd.Series) -> pd.Series:
    """Parse angka; dukung format ribuan yang jelas, jangan tebak satuan juta/miliar."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)

    def parse_one(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip().replace('\u00a0', '').replace(' ', '')
        t = re.sub(r'^(rp\.?|idr)', '', t, flags=re.I)
        if not t or t.lower() in {'nan', 'none', 'null', '-', '?'}:
            return np.nan
        # 4.500.000.000 atau 4,500,000,000 -> integer rupiah
        if re.fullmatch(r'[+-]?\d{1,3}([.,]\d{3})+', t):
            t = t.replace('.', '').replace(',', '')
        # 4500000000.0, 4.5e9, dsb; bentuk ambigu lain TIDAK dipaksa.
        try:
            return float(t)
        except (TypeError, ValueError):
            return np.nan

    result = s.map(parse_one)
    return pd.to_numeric(result, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _normalize_text(s: pd.Series) -> pd.Series:
    s = s.astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
    s = s.mask(s.str.lower().isin(['', 'nan', 'none', 'null', '-', '?', 'n/a']))
    s = s.str.title()
    # scikit-learn SimpleImputer kompatibel dengan np.nan, bukan pandas.NA
    return s.astype(object).where(s.notna(), np.nan)


def standardized_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[FEATURES + [TARGET]].copy()
    for c in CAT_COLS:
        df[c] = _normalize_text(df[c])
    for c in NUM_COLS + [TARGET]:
        df[c] = numeric_series(df[c])
    return df


def audit_dataset(raw: pd.DataFrame, out: Path) -> dict:
    out = Path(out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    df = standardized_features(raw)

    invalid = {
        'price_kosong_atau_tidak_terbaca': int(df[TARGET].isna().sum()),
        'price_tidak_positif': int((df[TARGET] <= 0).sum()),
        'land_area_tidak_positif': int((df['land_area'] <= 0).sum()),
        'building_area_tidak_positif': int((df['building_area'] <= 0).sum()),
        'bed_rooms_negatif': int((df['bed_rooms'] < 0).sum()),
        'bath_rooms_negatif': int((df['bath_rooms'] < 0).sum()),
        'carport_negatif': int((df['carport'] < 0).sum()),
        'bed_rooms_pecahan': int((df['bed_rooms'].notna() & (df['bed_rooms'] % 1 != 0)).sum()),
        'bath_rooms_pecahan': int((df['bath_rooms'].notna() & (df['bath_rooms'] % 1 != 0)).sum()),
        'carport_pecahan': int((df['carport'].notna() & (df['carport'] % 1 != 0)).sum()),
        'bed_rooms_nol_suspek': int((df['bed_rooms'] == 0).sum()),
        'bath_rooms_nol_suspek': int((df['bath_rooms'] == 0).sum()),
    }
    missing = pd.DataFrame({
        'kolom': df.columns,
        'tipe_asli': [str(raw[c].dtype) for c in df.columns],
        'missing_mentah': [int(raw[c].isna().sum()) for c in df.columns],
        'missing_setelah_parse': [int(df[c].isna().sum()) for c in df.columns],
        'persen_missing_setelah_parse': [round(100 * df[c].isna().mean(), 3) for c in df.columns],
    })
    missing.to_csv(out / 'audit_missing.csv', index=False)
    pd.DataFrame([{'indikator': k, 'jumlah': v} for k, v in invalid.items()]).to_csv(
        out / 'audit_invalid.csv', index=False
    )
    city = df['city'].fillna('Tidak Diketahui').value_counts(dropna=False).rename_axis('city').reset_index(name='jumlah')
    district = df['district'].fillna('Tidak Diketahui').value_counts(dropna=False).rename_axis('district').reset_index(name='jumlah')
    city.to_csv(out / 'audit_city.csv', index=False)
    district.to_csv(out / 'audit_district.csv', index=False)
    combos = (df.dropna(subset=['district', 'city']).groupby('district')['city']
              .nunique().reset_index(name='jumlah_city'))
    combos.query('jumlah_city > 1').to_csv(out / 'audit_location_ambiguity.csv', index=False)
    df[NUM_COLS + [TARGET]].corr(numeric_only=True)[TARGET].rename('korelasi_dengan_price').to_csv(
        out / 'audit_numeric_correlations.csv'
    )
    price = df[TARGET].loc[df[TARGET].gt(0)].dropna()
    summary = {
        'baris_awal': int(len(raw)), 'kolom_awal': int(len(raw.columns)),
        'kolom_asli': list(raw.columns), 'baris_duplikat_seluruh_kolom': int(raw.duplicated().sum()),
        'baris_duplikat_fitur_dan_target': int(df.duplicated(subset=FEATURES + [TARGET]).sum()),
        'jumlah_missing_mentah': int(raw.isna().sum().sum()),
        'jumlah_missing_setelah_parse': int(df.isna().sum().sum()),
        'jumlah_city': int(df['city'].nunique()), 'jumlah_district': int(df['district'].nunique()),
        'district_muncul_di_lebih_satu_city': int((combos['jumlah_city'] > 1).sum()),
        'harga_min': float(price.min()) if len(price) else None,
        'harga_median': float(price.median()) if len(price) else None,
        'harga_max': float(price.max()) if len(price) else None,
        'harga_skewness': float(price.skew()) if len(price) > 2 else None,
        'nilai_tidak_valid': invalid,
        'catatan': ('Audit seluruh data hanya untuk deskripsi kualitas. Pemilihan log, '
                    'imputasi, encoding, dan tuning TIDAK memakai statistik holdout.'),
    }
    (out / 'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    if len(price):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].hist(price / 1e9, bins=50, color='#155A89')
        axes[0].set(title='Distribusi Harga Asli', xlabel='Harga (miliar Rp)', ylabel='Jumlah')
        axes[1].hist(np.log1p(price), bins=50, color='#15947C')
        axes[1].set(title='Distribusi log1p(price) - audit visual', xlabel='log1p(price)')
        axes[2].boxplot(price / 1e9, vert=False)
        axes[2].set(title='Boxplot Harga', xlabel='Harga (miliar Rp)')
        fig.tight_layout()
        fig.savefig(out / 'figures' / 'audit_distribusi_harga.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    return summary


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aturan berlaku sama untuk seluruh baris; tidak pakai quantile/harga holdout."""
    df = standardized_features(raw)
    decisions = {'baris_awal': int(len(df))}
    invalid_target = df[TARGET].isna() | df[TARGET].le(0)
    decisions['hapus_target_hilang_tidak_positif'] = int(invalid_target.sum())
    df = df.loc[~invalid_target].copy()

    # Hanya nilai yang melanggar logika domain. Outlier besar tetap dipertahankan.
    count_invalid = {}
    for c in ['land_area', 'building_area']:
        mask = df[c].le(0)
        count_invalid[c] = int(mask.sum())
        df.loc[mask, c] = np.nan
    for c in ['bed_rooms', 'bath_rooms', 'carport']:
        mask = df[c].lt(0) | (df[c].notna() & df[c].mod(1).ne(0))
        count_invalid[c] = int(mask.sum())
        df.loc[mask, c] = np.nan
    decisions['fitur_tidak_valid_diubah_nan_untuk_imputasi_fold'] = count_invalid

    duplicated = df.duplicated(subset=FEATURES + [TARGET], keep='first')
    decisions['hapus_duplikat_persis_fitur_dan_target'] = int(duplicated.sum())
    df = df.loc[~duplicated].copy()
    decisions['baris_setelah_clean'] = int(len(df))
    decisions['nilai_ekstrem_harga'] = 'Dipertahankan; tidak ada clipping winsorizing atau outlier deletion otomatis.'
    decisions['kolom_index'] = 'Tidak menjadi prediktor.'
    if len(df) < 50:
        raise ValueError('Setelah pembersihan, data <50 baris. Periksa format price dan file dataset.')
    # Jika harga ternyata dalam satuan juta/miliar alih-alih rupiah, hentikan; jangan ubah otomatis.
    if df[TARGET].median() < 1_000_000:
        raise ValueError('Median price < Rp1 juta. Satuan price tampaknya bukan rupiah penuh. '
                         'Periksa CSV dan dokumentasikan konversinya sebelum modeling.')
    return df, decisions
