"""Pelatihan dan evaluasi berbasis training-only CV + final holdout."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import (CAT_COLS, CV_FOLDS, FEATURES, MODEL_DIR, MODEL_LABELS,
                     NUM_COLS, OUTPUT_DIR, RANDOM_STATE, SEARCH_ITERATIONS,
                     SEARCH_JOBS, TARGET, TEST_SIZE)


def identity(x):
    """Fungsi top-level agar joblib bisa memuat model raw di aplikasi Streamlit."""
    return x


def preprocess_pipeline() -> ColumnTransformer:
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    category = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
    ])
    return ColumnTransformer(
        [('numeric', numeric, NUM_COLS), ('category', category, CAT_COLS)],
        remainder='drop', sparse_threshold=0.3,
    )


def make_estimator(name: str, transform: str):
    if name == 'linear_regression':
        model = LinearRegression()
    elif name == 'random_forest':
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    elif name == 'xgboost':
        model = XGBRegressor(
            objective='reg:squarederror', random_state=RANDOM_STATE,
            n_jobs=1, tree_method='hist', verbosity=0,
        )
    else:
        raise ValueError(f'Model tidak dikenal: {name}')
    pipe = Pipeline([('preprocess', preprocess_pipeline()), ('model', model)])
    if transform == 'log1p':
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False,
        )
    if transform == 'raw':
        return TransformedTargetRegressor(
            regressor=pipe, func=identity, inverse_func=identity, check_inverse=False,
        )
    raise ValueError('Transformasi hanya raw atau log1p.')


RF_SPACE = {
    'regressor__model__n_estimators': [200, 400, 600, 800],
    'regressor__model__max_depth': [None, 10, 20, 30],
    'regressor__model__min_samples_split': [2, 5, 10],
    'regressor__model__min_samples_leaf': [1, 2, 4],
    'regressor__model__max_features': ['sqrt', 'log2', 1.0],
}
XGB_SPACE = {
    'regressor__model__n_estimators': [200, 400, 600, 800],
    'regressor__model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__model__max_depth': [3, 5, 7, 10],
    'regressor__model__subsample': [0.7, 0.8, 0.9, 1.0],
    'regressor__model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'regressor__model__reg_alpha': [0, 0.01, 0.1, 1],
    'regressor__model__reg_lambda': [1, 5, 10],
}


def split_data(clean: pd.DataFrame):
    """Split satu kali; jangan akses X_test/y_test sampai tahap evaluasi final."""
    return train_test_split(
        clean[FEATURES], clean[TARGET], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, shuffle=True,
    )


def _score(estimator, X, y, cv):
    scores = cross_validate(
        estimator, X, y, cv=cv,
        scoring={'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'},
        n_jobs=SEARCH_JOBS, error_score='raise',
    )
    return scores


def train_candidates(X_train: pd.DataFrame, y_train: pd.Series,
                     output: Path = OUTPUT_DIR, n_iter: int = SEARCH_ITERATIONS,
                     folds: int = CV_FOLDS, fast_test: bool = False):
    """Mencari target raw/log dan hiperparameter HANYA dari train via CV.

    fast_test hanya untuk unit smoke-test dataset sintetis, bukan untuk pelaporan skripsi.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    chosen = {}
    comparisons = []
    search_rows = []
    cv_rows = []
    for name in MODEL_LABELS:
        options = []
        for transformation in ['raw', 'log1p']:
            estimator = make_estimator(name, transformation)
            print(f'  - {MODEL_LABELS[name]} | target={transformation}', flush=True)
            if name == 'linear_regression':
                candidate = estimator
                scores = _score(candidate, X_train, y_train, cv)
                rmse_mean = float(-scores['test_rmse'].mean())
                options.append((rmse_mean, transformation, candidate, {}, None))
                search_rows.append({'model': name, 'transformasi': transformation,
                                    'rmse_cv': rmse_mean, 'parameter': '{}'})
            else:
                space = RF_SPACE if name == 'random_forest' else XGB_SPACE
                if fast_test:
                    space = dict(space)
                    space['regressor__model__n_estimators'] = [10]
                    if name == 'random_forest':
                        space['regressor__model__max_depth'] = [5]
                    else:
                        space['regressor__model__max_depth'] = [3]
                search = RandomizedSearchCV(
                    estimator, param_distributions=space, n_iter=n_iter,
                    scoring='neg_root_mean_squared_error', cv=cv,
                    n_jobs=SEARCH_JOBS, random_state=RANDOM_STATE,
                    refit=True, verbose=0, error_score='raise',
                    return_train_score=False,
                )
                search.fit(X_train, y_train)
                candidate = search.best_estimator_
                rmse_mean = float(-search.best_score_)
                options.append((rmse_mean, transformation, candidate, search.best_params_, search))
                search_rows.append({'model': name, 'transformasi': transformation,
                                    'rmse_cv': rmse_mean,
                                    'parameter': json.dumps(search.best_params_, ensure_ascii=False)})
                pd.DataFrame(search.cv_results_).to_csv(
                    output / f'tuning_{name}_{transformation}.csv', index=False
                )
        options.sort(key=lambda x: x[0])
        best_rmse, best_transform, best_estimator, params, search = options[0]
        # Skor per fold untuk konfigurasi terpilih. Ini NON-NESTED CV dan tidak
        # dapat diklaim sebagai estimasi generalisasi yang unbiased.
        if name == 'linear_regression':
            fold_scores = _score(best_estimator, X_train, y_train, cv)
            fold_rmse = -fold_scores['test_rmse']
            fold_mae = -fold_scores['test_mae']
            fold_r2 = fold_scores['test_r2']
        else:
            idx = int(search.best_index_)
            fold_rmse = np.array([-search.cv_results_[f'split{k}_test_score'][idx]
                                  for k in range(folds)], dtype=float)
            fold_mae = np.full(folds, np.nan)
            fold_r2 = np.full(folds, np.nan)
        for i, value in enumerate(fold_rmse, 1):
            cv_rows.append({'model': name, 'transformasi': best_transform, 'fold': i,
                            'RMSE': float(value),
                            'MAE': float(fold_mae[i - 1]) if np.isfinite(fold_mae[i - 1]) else np.nan,
                            'R2': float(fold_r2[i - 1]) if np.isfinite(fold_r2[i - 1]) else np.nan})
        comparisons.append({
            'model': name, 'nama_model': MODEL_LABELS[name], 'transformasi': best_transform,
            'CV_RMSE_mean': float(np.mean(fold_rmse)),
            'CV_RMSE_std': float(np.std(fold_rmse, ddof=1)) if folds > 1 else 0.0,
            'best_params': json.dumps(params, ensure_ascii=False),
        })
        chosen[name] = best_estimator

    cv_table = pd.DataFrame(comparisons).sort_values(['CV_RMSE_mean', 'model']).reset_index(drop=True)
    winner = str(cv_table.loc[0, 'model'])
    pd.DataFrame(search_rows).to_csv(output / 'search_raw_vs_log.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(output / 'cv_per_fold.csv', index=False)
    cv_table.to_csv(output / 'cv_comparison.csv', index=False)
    print(f'\nModel deployment DITENTUKAN via CV train: {MODEL_LABELS[winner]}', flush=True)
    return chosen, cv_table, winner


def _metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'MSE': float(mse), 'RMSE': float(np.sqrt(mse)),
        'R2': float(r2_score(y_true, y_pred)),
    }


def _save_eval_plots(preds: pd.DataFrame, table: pd.DataFrame, output: Path):
    figdir = output / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    models = list(MODEL_LABELS)
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [float(table.set_index('model').loc[m, 'RMSE']) / 1e9 for m in models]
    ax.bar([MODEL_LABELS[x] for x in models], values, color=['#628BB8', '#2B7D93', '#209A7A'])
    ax.set(title='RMSE Holdout per Algoritma', ylabel='RMSE (miliar Rp)')
    ax.tick_params(axis='x', labelrotation=10)
    fig.tight_layout(); fig.savefig(figdir / 'rmse_holdout.png', dpi=160); plt.close(fig)

    for m in models:
        sub = preds[preds['model'].eq(m)]
        actual = sub['actual'].to_numpy(dtype=float) / 1e9
        pred = sub['prediction'].to_numpy(dtype=float) / 1e9
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].scatter(actual, pred, alpha=.25, s=12, c='#176394')
        bounds = [float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))]
        axes[0].plot(bounds, bounds, linestyle='--', color='#BE6C39')
        axes[0].set(xlabel='Aktual (miliar)', ylabel='Prediksi (miliar)', title='Aktual vs Prediksi')
        axes[1].hist(sub['residual'] / 1e9, bins=40, color='#2B9481')
        axes[1].set(xlabel='Aktual - Prediksi (miliar)', title='Distribusi Residual')
        axes[2].scatter(pred, sub['residual'] / 1e9, alpha=.25, s=12, c='#785F9E')
        axes[2].axhline(0, linestyle='--', color='#BE6C39')
        axes[2].set(xlabel='Prediksi (miliar)', ylabel='Residual (miliar)', title='Residual vs Prediksi')
        fig.suptitle(MODEL_LABELS[m]); fig.tight_layout()
        fig.savefig(figdir / f'residual_{m}.png', dpi=150); plt.close(fig)


def _train_input_metadata(X_train: pd.DataFrame) -> dict:
    options = X_train[CAT_COLS].dropna().drop_duplicates()
    options = options.sort_values(['city', 'district'])
    defaults = {col: float(X_train[col].median()) for col in NUM_COLS}
    bounds = {col: {
        'min': float(X_train[col].min()),
        'p01': float(X_train[col].quantile(.01)),
        'median': float(X_train[col].median()),
        'p99': float(X_train[col].quantile(.99)),
        'max': float(X_train[col].max()),
    } for col in NUM_COLS}
    return {'city_options': sorted(X_train['city'].dropna().unique().tolist()),
            'district_city_pairs': options.to_dict(orient='records'),
            'default_numeric': defaults, 'numeric_ranges_train': bounds}


def evaluate_holdout_once(models: dict, cv_table: pd.DataFrame, winner: str,
                          X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series,
                          audit_summary: dict, decisions: dict,
                          output: Path = OUTPUT_DIR, modeldir: Path = MODEL_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hanya panggil SEKALI seusai seluruh pilihan transform & tuning terkunci."""
    output, modeldir = Path(output), Path(modeldir)
    output.mkdir(parents=True, exist_ok=True)
    modeldir.mkdir(parents=True, exist_ok=True)
    # Ambang segmen disepakati dari train saja; tidak menggunakan kuantil data uji.
    q33, q67 = np.quantile(y_train.to_numpy(), [1 / 3, 2 / 3])
    if q33 >= q67:
        raise ValueError('Ambang segmen harga dari y_train tidak unik. Perlu revisi pengelompokan yang terdokumentasi.')
    segment = pd.cut(y_test, [-np.inf, q33, q67, np.inf],
                     labels=['Rendah', 'Menengah', 'Tinggi'], include_lowest=True)
    results, predictions = [], []
    for name, model in models.items():
        # search.best_estimator_ telah fitted oleh refit, baseline belum tentu;
        # refit semua final secara konsisten pada seluruh train.
        model.fit(X_train, y_train)
        test_pred = np.asarray(model.predict(X_test), dtype=float)
        if not np.isfinite(test_pred).all():
            raise ValueError(f'Prediksi {name} memuat NaN/Infinity. Evaluasi dihentikan.')
        scores = _metrics(y_test, test_pred)
        results.append({'model': name, 'nama_model': MODEL_LABELS[name], **scores})
        predictions.append(pd.DataFrame({
            'record_id': X_test.index.to_numpy(),
            'model': name,
            'actual': y_test.to_numpy(dtype=float), 'prediction': test_pred,
            'residual': y_test.to_numpy(dtype=float) - test_pred,
            'abs_error': np.abs(y_test.to_numpy(dtype=float) - test_pred),
            'segment': segment.astype(str).to_numpy(),
        }))
        joblib.dump(model, modeldir / f'{name}.joblib', compress=3)
        if name == winner:
            joblib.dump(model, modeldir / 'best_model.joblib', compress=3)
    table = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
    preds = pd.concat(predictions, ignore_index=True)
    test_winner = str(table.loc[0, 'model'])
    segment_rows = []
    for (name, group), x in preds.groupby(['model', 'segment'], observed=True):
        segment_rows.append({'model': name, 'segment': group, 'n': len(x),
                             **_metrics(x['actual'], x['prediction'])})
    pd.DataFrame(segment_rows).to_csv(output / 'segment_error.csv', index=False)
    table.to_csv(output / 'test_comparison.csv', index=False)
    preds.to_csv(output / 'test_predictions.csv', index=False)
    pd.DataFrame({'record_id': X_train.index}).to_csv(output / 'train_row_ids.csv', index=False)
    pd.DataFrame({'record_id': X_test.index}).to_csv(output / 'test_row_ids.csv', index=False)
    selection = {
        'selected_by_train_cv': winner,
        'selected_by_train_cv_label': MODEL_LABELS[winner],
        'holdout_lowest_rmse_model_for_reporting_only': test_winner,
        'warning': ('Deployment model dipilih dari minimum RMSE CV data latih, bukan dari hasil test. '
                    'Jangan beralih ke model holdout lebih rendah setelah melihat test.'),
        'segment_thresholds_train_rupiah': {'q33': float(q33), 'q67': float(q67)},
        'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
        'train_test_overlap_row_indices': int(len(set(X_train.index) & set(X_test.index))),
    }
    metadata = {**selection, **_train_input_metadata(X_train),
                'audit_summary': audit_summary, 'preprocessing_decisions': decisions,
                'target_unit': 'IDR (rupiah)',
                'target_meaning': 'Harga penawaran dalam listing, bukan harga transaksi final',
                'features': FEATURES, 'split_random_state': RANDOM_STATE,
                'cv_folds': CV_FOLDS,
                'evaluation_note': 'RMSE/MAE/MSE dievaluasi dalam rupiah setelah inverse_transform target.'}
    (output / 'model_selection.json').write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'preprocessing_decisions.json').write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding='utf-8')
    _save_eval_plots(preds, table, output)
    (output / 'RINGKASAN_METODOLOGI.txt').write_text(
        'Penelitian: Ilsan Hidayatuloh, NPM 41155050230013\n'
        'Tiga model, tujuh fitur identik, train-test 80:20, seed 42.\n'
        'Imputasi dan OHE di dalam pipeline per CV fold.\n'
        'Raw/log1p target dan hiperparameter diseleksi HANYA melalui CV train.\n'
        'RMSE CV menjadi kriteria pemilihan deployment sebelum melihat holdout.\n'
        'Holdout 20% dievaluasi sekali untuk laporan ketiga model; tidak untuk retuning.\n'
        'Residual = aktual - prediksi. Ambang segmen ditetapkan dari train.\n'
        'Catatan: random split tidak dapat mengatasi listing properti yang sama bila '
        'muncul dengan atribut hampir identik; tidak tersedia property_id.\n'
        'CV per fold setelah hyperparameter search adalah NON-NESTED, jangan diklaim unbiased.\n'
        'Keamanan: joblib hanya muat dari folder model tepercaya.\n', encoding='utf-8'
    )
    return table, preds
