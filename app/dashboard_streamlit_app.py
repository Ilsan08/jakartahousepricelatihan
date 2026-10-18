
"""Dashboard Streamlit Skripsi — Ilsan Hidayatuloh."""

from __future__ import annotations
from src.config import CAT_COLS, FEATURES, MODEL_LABELS, NUM_COLS
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# ============================================
# IMPORT LIBRARY DASAR
# ============================================
import sys
import json
from pathlib import Path

# ============================================
# MENENTUKAN ROOT PROJECT
# ============================================
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================
# IMPORT LIBRARY DASHBOARD
# ============================================

# ============================================
# IMPORT MODUL PENELITIAN
# ============================================

# Import modul penelitian

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title='Jakarta House Price | Ilsan Hidayatuloh',
                   page_icon='🏠', layout='wide', initial_sidebar_state='expanded')

CSS = '''
<style>streamli
.stApp {background: #F5F7FA; color: #172B4D;}
.block-container {padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1420px;}
h1,h2,h3 {color: #122B4A; letter-spacing: -0.025em;}
.hero {background: linear-gradient(112deg,#0B2949 0%, #134969 58%, #186F7B 100%);
       border-radius: 18px; padding: 31px 36px; color: white; margin-bottom: 16px;}
.hero .eyebrow {text-transform:uppercase; letter-spacing:2.4px; font-size:.72rem;
                color:#B8D9E8; font-weight:700;}
.hero h1 {color: white; font-size:2.35rem; line-height:1.15; margin:10px 0 9px;}
.hero p {color: #D7E5F0; margin:0; font-size:.99rem; max-width: 770px;}
.metric-card {background:white; border:1px solid #E2E8F0; border-radius:13px;
              padding:21px 22px; min-height:118px; box-shadow: 0 5px 20px #12385d09;}
.metric-title {font-size:.75rem; letter-spacing:1px; color:#64748B; text-transform:uppercase; font-weight:700;}
.metric-value {font-size:1.68rem; color:#143F5E; font-weight:750; margin-top:9px;}
.metric-note {font-size:.76rem;color:#718096;margin-top:6px;}
.section-intro {color:#586D81;font-size:.93rem;}
.notice {background:#EAF4F7;border-left:4px solid #248599;padding:15px 18px;
         border-radius:7px;color:#244A60;}
.tag {display:inline-block; padding:5px 11px; color:#D4EDF7; background:#265675;
      border-radius:99px; font-size:.75rem; margin-top:15px;}
[data-testid="stSidebar"] {background: #EAF0F5;}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {
 background:#136B80; border-color:#136B80; color:white; border-radius:9px; }
[data-testid="stMetric"] {background:white;border:1px solid #E2E8F0;border-radius:12px;padding:13px 16px;}
footer {visibility: hidden;}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)


def idr(value, short=False):
    value = float(value)
    sign = '-' if value < 0 else ''
    v = abs(value)
    if short and v >= 1e12:
        return f'{sign}Rp{v / 1e12:,.2f} triliun'.replace(',', '_').replace('.', ',').replace('_', '.')
    if short and v >= 1e9:
        return f'{sign}Rp{v / 1e9:,.2f} miliar'.replace(',', '_').replace('.', ',').replace('_', '.')
    if short and v >= 1e6:
        return f'{sign}Rp{v / 1e6:,.2f} juta'.replace(',', '_').replace('.', ',').replace('_', '.')
    return f'{sign}Rp{v:,.0f}'.replace(',', '.')


@st.cache_data(show_spinner=False)
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def read_model(path):
    return joblib.load(path)  # HANYA file joblib lokal hasil pipeline sendiri.


output = ROOT / 'outputs'
needed = [output / 'metadata.json', output / 'cv_comparison.csv',
          output / 'test_comparison.csv', output / 'test_predictions.csv',
          ROOT / 'models' / 'best_model.joblib']
missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
if missing:
    st.error(
        'Belum ada hasil training. Jalankan `notebooks/pipeline.ipynb` dari atas sampai bawah.')
    st.code('\n'.join(missing))
    st.stop()

metadata = read_json(output / 'metadata.json')
cv = read_csv(output / 'cv_comparison.csv')
test = read_csv(output / 'test_comparison.csv')
preds = read_csv(output / 'test_predictions.csv')
segments = read_csv(output / 'segment_error.csv')
folds = read_csv(output / 'cv_per_fold.csv')
selected = metadata['selected_by_train_cv']
model_name = MODEL_LABELS[selected]
selected_test = test.set_index('model').loc[selected]
cv_selected = cv.set_index('model').loc[selected]

with st.sidebar:
    st.markdown('### JAKARTA HOUSE PRICE')
    st.caption('STUDI KOMPARATIF · 2026')
    st.divider()
    st.markdown('**Peneliti**')
    st.write('Ilsan Hidayatuloh')
    st.caption('NPM 41155050230013')
    st.write('Teknik Informatika')
    st.caption('Universitas Langlangbuana')
    st.divider()
    st.markdown('**Model untuk deployment**')
    st.success(model_name)
    st.caption('Dipilih dari RMSE cross-validation pada data latih.')
    st.markdown('**Evaluasi**')
    st.caption('80% train · 20% holdout · 5-fold CV · seed 42')
    st.divider()
    st.caption(
        'Prediksi merupakan estimasi harga penawaran pada data listing, bukan appraisal atau harga transaksi resmi.')

st.markdown('''<div class="hero">
 <div class="eyebrow">Penelitian Skripsi · Teknik Informatika</div>
 <h1>Jakarta House Price</h1>
 <p>Perbandingan Regresi Linear Berganda, Random Forest, dan XGBoost untuk prediksi harga rumah di DKI Jakarta.</p>
 <span class="tag">BY ILSAN HIDAYATULOH · 41155050230013</span>
</div>''', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Ringkasan', 'Estimasi Harga', 'Perbandingan Model', 'Analisis Kesalahan', 'Audit & Metode'
])

with tab1:
    st.subheader('Hasil eksperimen')
    st.markdown('<p class="section-intro">Model ditetapkan dari CV train; performa holdout disajikan terpisah.</p>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric('Model deployment', model_name)
    b.metric('RMSE holdout', idr(selected_test['RMSE'], short=True))
    c.metric('MAE holdout', idr(selected_test['MAE'], short=True))
    d.metric('R² holdout', f"{selected_test['R2']:.4f}")
    st.write('')
    left, right = st.columns([1.3, 1])
    with left:
        fig = px.bar(test, x='nama_model', y='RMSE', color='nama_model',
                     color_discrete_sequence=['#3279A8', '#198E83', '#6D65A8'],
                     title='RMSE ketiga model pada holdout')
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', xaxis_title='',
                          yaxis_title='RMSE (Rp)', height=370)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('#### Ringkasan rancangan')
        audit = metadata['audit_summary']
        st.write(
            f"**{audit['baris_awal']:,}** baris awal · **{audit['kolom_awal']}** kolom mentah")
        st.write(
            f"**{metadata['train_rows']:,}** data latih · **{metadata['test_rows']:,}** data uji")
        st.write('7 prediktor · 3 model · 2 alternatif transformasi target')
        st.write(
            f"**CV RMSE model pilihan:** {idr(cv_selected['CV_RMSE_mean'], short=True)}")
        st.info('Nilai RMSE terkecil pada holdout bukan dasar untuk mengganti model yang telah dipilih melalui cross-validation.')
    st.markdown('<div class="notice"><b>Batas penggunaan</b> — Model memperkirakan harga penawaran pada dataset Jakarta House Price. Hasil dapat berbeda dari harga negosiasi, transaksi, atau penilaian profesional.</div>', unsafe_allow_html=True)

with tab2:
    st.subheader('Estimasi harga berdasarkan karakteristik rumah')
    st.caption('Masukkan kombinasi lokasi yang dikenal pada data latih. Semua input diproses oleh pipeline model yang sama dengan eksperimen.')
    pairs = pd.DataFrame(metadata['district_city_pairs'])
    if pairs.empty:
        st.error(
            'Tidak ada pasangan lokasi valid pada data latih. Periksa hasil audit.')
    else:
        with st.form('prediction_form', clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                cities = sorted(pairs['city'].dropna().unique().tolist())
                city = st.selectbox('Kota administratif', cities)
            with c2:
                district_list = sorted(pairs.loc[pairs['city'].eq(
                    city), 'district'].dropna().unique().tolist())
                district = st.selectbox(
                    'District / wilayah listing', district_list)
            defaults = metadata['default_numeric']
            x1, x2, x3 = st.columns(3)
            with x1:
                bed = st.number_input('Jumlah kamar tidur', min_value=0, max_value=100,
                                      value=int(round(defaults['bed_rooms'])), step=1)
                land = st.number_input('Luas tanah (m²)', min_value=1.0, max_value=100000.0,
                                       value=max(1.0, float(defaults['land_area'])), step=10.0)
            with x2:
                bath = st.number_input('Jumlah kamar mandi', min_value=0, max_value=100,
                                       value=int(round(defaults['bath_rooms'])), step=1)
                build = st.number_input('Luas bangunan (m²)', min_value=1.0, max_value=100000.0,
                                        value=max(1.0, float(defaults['building_area'])), step=10.0)
            with x3:
                carport = st.number_input('Kapasitas carport', min_value=0, max_value=100,
                                          value=max(0, int(round(defaults['carport']))), step=1)
                st.write('')
                st.caption(
                    'Satuan area mengikuti dataset, m². Semua prediktor digunakan.')
            submit = st.form_submit_button(
                'Hitung estimasi harga', type='primary', use_container_width=True)
        if submit:
            row = pd.DataFrame([{'district': district, 'city': city,
                                 'bed_rooms': bed, 'bath_rooms': bath, 'carport': carport,
                                 'land_area': land, 'building_area': build}], columns=FEATURES)
            with st.spinner('Menghitung estimasi...'):
                best_model = read_model(ROOT / 'models' / 'best_model.joblib')
                estimate = float(best_model.predict(row)[0])
            if not np.isfinite(estimate):
                st.error(
                    'Model menghasilkan prediksi tidak hingga. Periksa data input.')
            elif estimate < 0:
                st.error(
                    f'Prediksi negatif ({idr(estimate)}) tidak dapat ditafsirkan sebagai harga rumah. Kombinasi input berada di luar pola yang dapat dimodelkan dengan baik.')
            else:
                st.success('Estimasi berhasil dihitung')
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">Estimasi harga penawaran</div><div class="metric-value" style="font-size:2.2rem">{idr(estimate)}</div><div class="metric-note">Menggunakan {model_name} · bukan penilaian resmi</div></div>', unsafe_allow_html=True)
                cautions = []
                for name, val in [('bed_rooms', bed), ('bath_rooms', bath), ('carport', carport),
                                  ('land_area', land), ('building_area', build)]:
                    ref = metadata['numeric_ranges_train'][name]
                    if val < ref['p01'] or val > ref['p99']:
                        cautions.append(name)
                if cautions:
                    st.warning('Beberapa fitur berada di luar rentang persentil 1–99 data latih: '
                               + ', '.join(cautions) + '. Prediksi perlu ditafsirkan lebih hati-hati.')

with tab3:
    st.subheader('Perbandingan performa')
    st.caption(
        'Semua algoritma memakai tujuh prediktor dan holdout yang sama. RMSE menjadi metrik utama.')
    st.markdown('#### Skor cross-validation (data latih)')
    cv_display = cv[['nama_model', 'transformasi',
                     'CV_RMSE_mean', 'CV_RMSE_std']].copy()
    cv_display.columns = ['Algoritma', 'Target',
                          'Rata-rata RMSE (Rp)', 'SD RMSE (Rp)']
    st.dataframe(cv_display, use_container_width=True, hide_index=True,
                 column_config={'Rata-rata RMSE (Rp)': st.column_config.NumberColumn(format='%.0f'),
                                'SD RMSE (Rp)': st.column_config.NumberColumn(format='%.0f')})
    st.markdown('#### Evaluasi holdout akhir')
    metric_table = test[['nama_model', 'MAE', 'MSE', 'RMSE', 'R2']].copy()
    st.dataframe(metric_table, use_container_width=True, hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format='%.4g') for k in ['MAE', 'MSE', 'RMSE', 'R2']})
    metric = st.radio('Visualisasi metrik', [
                      'RMSE', 'MAE', 'MSE', 'R2'], horizontal=True)
    fig = px.bar(test, x='nama_model', y=metric, color='nama_model',
                 color_discrete_sequence=['#1D648D', '#188A7E', '#765AA6'])
    fig.update_layout(showlegend=False, xaxis_title='', yaxis_title=metric,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('#### Kestabilan RMSE 5-fold CV')
    fig = px.box(folds, x='model', y='RMSE', points='all', color='model',
                 color_discrete_sequence=['#1D648D', '#188A7E', '#765AA6'])
    fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='RMSE validasi (Rp)',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Variasi antar-fold pada konfigurasi yang dipilih melalui CV yang sama bersifat deskriptif dan dapat optimistis (non-nested CV).')

with tab4:
    st.subheader('Pola kesalahan prediksi')
    choice = st.selectbox('Pilih algoritma', list(MODEL_LABELS),
                          format_func=lambda key: MODEL_LABELS[key], index=list(MODEL_LABELS).index(selected))
    subset = preds.query('model == @choice').copy()
    x, y = st.columns(2)
    with x:
        fig = px.scatter(subset, x='actual', y='prediction', opacity=.43,
                         color_discrete_sequence=['#1E709C'], title='Aktual vs prediksi')
        ceiling = float(max(subset['actual'].max(),
                        subset['prediction'].max()))
        fig.add_trace(go.Scatter(x=[0, ceiling], y=[0, ceiling], mode='lines',
                                 line=dict(color='#BD7D43', dash='dash'), showlegend=False))
        fig.update_layout(xaxis_title='Aktual (Rp)',
                          yaxis_title='Prediksi (Rp)')
        st.plotly_chart(fig, use_container_width=True)
    with y:
        fig = px.histogram(subset, x='residual', nbins=50,
                           color_discrete_sequence=['#1A9582'], title='Distribusi residual')
        fig.update_layout(
            xaxis_title='Residual = aktual - prediksi (Rp)', yaxis_title='Jumlah')
        st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(subset, x='prediction', y='residual', opacity=.38,
                     color_discrete_sequence=['#7D63A0'], title='Residual terhadap nilai prediksi')
    fig.add_hline(y=0, line_dash='dash', line_color='#BF754C')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('#### Kesalahan pada kelompok harga')
    seg = segments[segments['model'].eq(choice)].copy()
    order = ['Rendah', 'Menengah', 'Tinggi']
    seg['segment'] = pd.Categorical(
        seg['segment'], categories=order, ordered=True)
    seg = seg.sort_values('segment')
    st.caption(
        'Batas rendah/menengah/tinggi dihitung dari tertil target pada data latih, bukan dari data uji.')
    st.dataframe(seg[['segment', 'n', 'MAE', 'MSE', 'RMSE', 'R2']],
                 hide_index=True, use_container_width=True)
    fig = px.bar(seg, x='segment', y='RMSE', color='segment',
                 color_discrete_sequence=['#4B82B2', '#217F9A', '#198C78'])
    fig.update_layout(showlegend=False,
                      xaxis_title='Kelompok harga', yaxis_title='RMSE (Rp)')
    st.plotly_chart(fig, use_container_width=True)
    st.download_button('Unduh prediksi holdout (CSV)', subset.to_csv(index=False).encode('utf-8'),
                       file_name=f'holdout_{choice}.csv', mime='text/csv')

with tab5:
    st.subheader('Audit awal dan prosedur penelitian')
    a = metadata['audit_summary']
    ac, bc, cc = st.columns(3)
    ac.metric('Baris awal', f"{a['baris_awal']:,}")
    bc.metric('Missing mentah', a['jumlah_missing_mentah'])
    cc.metric('Duplikat fitur & target', a['baris_duplikat_fitur_dan_target'])
    img = output / 'figures' / 'audit_distribusi_harga.png'
    if img.exists():
        st.image(str(img), caption='Distribusi harga aktual hasil audit file CSV')
    if (output / 'audit_missing.csv').exists():
        st.markdown('#### Missing value per kolom')
        st.dataframe(read_csv(output / 'audit_missing.csv'),
                     use_container_width=True, hide_index=True)
    if (output / 'audit_invalid.csv').exists():
        st.markdown('#### Temuan nilai invalid / perlu verifikasi')
        st.dataframe(read_csv(output / 'audit_invalid.csv'),
                     use_container_width=True, hide_index=True)
    if (output / 'audit_city.csv').exists():
        st.markdown('#### Komposisi wilayah')
        city = read_csv(output / 'audit_city.csv')
        fig = px.bar(city, x='city', y='jumlah', color='city',
                     title='Distribusi listing per kota')
        fig.update_layout(showlegend=False, xaxis_title='',
                          yaxis_title='Jumlah listing')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('#### Penerapan CRISP-DM')
    st.markdown('''
1. **Business Understanding:** membandingkan prediksi harga penawaran pada listing.
2. **Data Understanding:** audit baris/kolom, missing, duplikasi, lokasi, distribusi, nilai invalid.
3. **Data Preparation:** aturan domain yang terdokumentasi, 80:20 split, imputasi + OHE di pipeline per fold.
4. **Modeling:** regresi linear baseline, RF dan XGBoost dituning dengan RandomizedSearchCV 5-fold.
5. **Evaluation:** RMSE kriteria utama, MAE/MSE/R² pendukung, residual, segmen harga, variasi fold.
6. **Deployment:** pipeline model CV-terpilih disimpan utuh dan digunakan di dashboard ini.
''')
    st.warning('Keterbatasan: data listing bukan transaksi; split acak tidak menjamin listing properti serupa tidak tersebar lintas subset jika tidak tersedia ID properti. Tidak ada klaim appraisal resmi.')

st.divider()
st.caption('Jakarta House Price · Penelitian dan implementasi oleh Ilsan Hidayatuloh · Universitas Langlangbuana · 2026')
