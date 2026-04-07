
from pathlib import Path
import sys, json, os, shutil, hashlib, traceback, gc
from copy import deepcopy
import pandas as pd
import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

def json_safe(x, _depth=0, _seen=None):
    import dataclasses
    from collections.abc import Mapping
    if _seen is None:
        _seen = set()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    obj_id = id(x)
    if obj_id in _seen:
        return f'<recursive:{type(x).__name__}>'
    if _depth > 4:
        return repr(x)
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist(), _depth=_depth + 1, _seen=_seen)
    if isinstance(x, pd.Series):
        return json_safe(x.to_list(), _depth=_depth + 1, _seen=_seen)
    if isinstance(x, pd.Index):
        return json_safe(x.tolist(), _depth=_depth + 1, _seen=_seen)
    if isinstance(x, pd.DataFrame):
        return {'__class__': 'DataFrame', 'shape': list(x.shape), 'columns': [str(c) for c in x.columns.tolist()]}
    if dataclasses.is_dataclass(x):
        _seen.add(obj_id)
        try:
            return json_safe(dataclasses.asdict(x), _depth=_depth + 1, _seen=_seen)
        finally:
            _seen.discard(obj_id)
    if isinstance(x, Mapping):
        _seen.add(obj_id)
        try:
            return {str(k): json_safe(v, _depth=_depth + 1, _seen=_seen) for k, v in x.items()}
        finally:
            _seen.discard(obj_id)
    if isinstance(x, (list, tuple, set)):
        _seen.add(obj_id)
        try:
            return [json_safe(v, _depth=_depth + 1, _seen=_seen) for v in x]
        finally:
            _seen.discard(obj_id)
    if hasattr(x, '__dict__'):
        _seen.add(obj_id)
        try:
            items = {}
            for k, v in list(vars(x).items())[:100]:
                if str(k).startswith('_'):
                    continue
                items[str(k)] = json_safe(v, _depth=_depth + 1, _seen=_seen)
            if items:
                return {'__class__': type(x).__name__, **items}
        except Exception:
            pass
        finally:
            _seen.discard(obj_id)
    return repr(x)

def set_attr_any(obj, names, value):
    last_err = None
    for name in names:
        if hasattr(obj, name):
            try:
                setattr(obj, name, value)
                return name
            except (AttributeError, TypeError) as e:
                last_err = e
                continue
    for name in names:
        try:
            setattr(obj, name, value)
            return name
        except (AttributeError, TypeError) as e:
            last_err = e
            continue
    if hasattr(obj, '__dict__'):
        obj.__dict__[names[0]] = value
        return names[0]
    raise AttributeError(f'Could not set any of {names} on {type(obj).__name__}. Last error: {last_err}')

def force_set_any(obj, names, value):
    for name in names:
        try:
            setattr(obj, name, value)
            return name
        except Exception:
            pass
    if hasattr(obj, '__dict__'):
        for name in names:
            try:
                obj.__dict__[name] = value
                return name
            except Exception:
                pass
    raise AttributeError(f'Could not force-set any of {names} on {type(obj).__name__}')

def maybe_set_window_dicts(settings, start, end):
    payload = {'start': pd.Timestamp(start), 'end': pd.Timestamp(end), 'start_str': str(pd.Timestamp(start).date()), 'end_str': str(pd.Timestamp(end).date())}
    for name in ['requested_window', 'analysis_window', 'verification_window', 'time_window', 'date_window', 'window', 'window_spec']:
        try:
            force_set_any(settings, [name], payload)
        except Exception:
            pass
    tuple_payload = (pd.Timestamp(start), pd.Timestamp(end))
    list_payload = [pd.Timestamp(start), pd.Timestamp(end)]
    for name in ['time_range', 'date_range', 'window_range', 'analysis_range', 'verification_range']:
        try:
            force_set_any(settings, [name], tuple_payload)
        except Exception:
            try:
                force_set_any(settings, [name], list_payload)
            except Exception:
                pass

def configure_window(settings, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start_s = str(start.date())
    end_s = str(end.date())
    date_windows_payload = [(start_s, end_s)]
    start_ts_names = ['analysis_start', 'window_start', 'start_time', 'start', 'verification_start', 'start_date', 'date_start', 'time_start', 'analysis_start_time', 'window_start_time', 'window_start_date', 'verify_start', 'valid_start', 'truth_start', 'forecast_start', 'sample_start', 'period_start', 'time_min', 'date_min']
    end_ts_names = ['analysis_end', 'window_end', 'end_time', 'end', 'verification_end', 'end_date', 'date_end', 'time_end', 'analysis_end_time', 'window_end_time', 'window_end_date', 'verify_end', 'valid_end', 'truth_end', 'forecast_end', 'sample_end', 'period_end', 'time_max', 'date_max']
    start_str_names = ['analysis_start_str', 'window_start_str', 'start_str', 'start_date_str', 'verification_start_str', 'date_start_str']
    end_str_names = ['analysis_end_str', 'window_end_str', 'end_str', 'end_date_str', 'verification_end_str', 'date_end_str']
    for names, value in [(start_ts_names, start), (end_ts_names, end), (start_str_names, start_s), (end_str_names, end_s)]:
        for name in names:
            try:
                force_set_any(settings, [name], value)
            except Exception:
                pass
    for names, value in [(['date_windows'], date_windows_payload), (['date_window_pairs', 'analysis_windows', 'verification_windows', 'windows'], date_windows_payload), (['requested_window_start'], start), (['requested_window_end'], end), (['requested_window_label'], f'{start_s}_{end_s}')]:
        try:
            force_set_any(settings, names, value)
        except Exception:
            pass
    if hasattr(settings, '__dict__'):
        for key in list(settings.__dict__.keys()):
            lk = str(key).lower()
            try:
                if lk.endswith('date_windows') or lk == 'date_windows':
                    settings.__dict__[key] = date_windows_payload
                elif ('start' in lk or lk.endswith('_min')) and isinstance(settings.__dict__.get(key), (str, pd.Timestamp)):
                    settings.__dict__[key] = start
                elif ('end' in lk or lk.endswith('_max')) and isinstance(settings.__dict__.get(key), (str, pd.Timestamp)):
                    settings.__dict__[key] = end
            except Exception:
                pass
    maybe_set_window_dicts(settings, start, end)
    return settings

def configure_leads(settings, lead_hours):
    lead_hours = [int(x) for x in lead_hours]
    lead_strings = [f'{x}h' for x in lead_hours]
    lead_tds = [pd.Timedelta(hours=x) for x in lead_hours]
    set_attr_any(settings, ['lead_hours', 'forecast_hours'], lead_hours)
    set_attr_any(settings, ['lead_times', 'flagship_leads'], lead_strings)
    set_attr_any(settings, ['lead_timedeltas', 'lead_td', 'forecast_timedeltas'], lead_tds)
    try:
        force_set_any(settings, ['lag'], '12h')
    except Exception:
        pass
    return settings

def configure_regimes(settings, n_regimes=4, n_components=8, n_boot=400, block_length=5):
    for names, value in [(['n_regimes', 'regime_count'], int(n_regimes)), (['n_eof', 'n_components', 'regime_n_components', 'n_eof_components'], int(n_components)), (['bootstrap_samples', 'n_boot', 'n_bootstrap', 'bootstrap_n'], int(n_boot)), (['bootstrap_block_length', 'block_length', 'bootstrap_block'], int(block_length))]:
        try:
            force_set_any(settings, names, value)
        except Exception:
            pass
    return settings

def configure_blocking(settings, threshold=0.1):
    try:
        force_set_any(settings, ['blocking_threshold'], float(threshold))
    except Exception:
        pass
    return settings

def configure_growth(settings, lag_hours=12):
    for names, value in [(['growth_lag_hours', 'lag_hours', 'trajectory_lag_hours'], int(lag_hours)), (['lag'], f'{int(lag_hours)}h')]:
        try:
            force_set_any(settings, names, value)
        except Exception:
            pass
    return settings

def settings_window_snapshot(settings):
    if hasattr(settings, '__dict__'):
        base = {k: v for k, v in settings.__dict__.items() if not str(k).startswith('_')}
    else:
        base = {}
    keys = ['truth_dataset','deterministic_models','ensemble_models','date_windows','leads_hours','variables','n_regimes','n_eof_components','n_eof','bootstrap_n','bootstrap_block','blocking_sectors','blocking_threshold','assume_geopotential','lag','analysis_start','window_start','start_time','start','verification_start','start_date','date_start','time_start','analysis_end','window_end','end_time','end','verification_end','end_date','date_end','time_end','analysis_start_str','window_start_str','start_str','start_date_str','verification_start_str','date_start_str','analysis_end_str','window_end_str','end_str','end_date_str','verification_end_str','date_end_str','date_window_pairs','requested_window_start','requested_window_end','requested_window_label','requested_window','analysis_window','verification_window','time_window','date_window','window','window_spec','time_range','date_range','window_range','analysis_range','verification_range','lead_hours','lead_times','lead_timedeltas','n_eof','bootstrap_samples','bootstrap_block_length','growth_lag_hours','leads']
    snap = {}
    for k in keys:
        try:
            if hasattr(settings, k):
                snap[k] = getattr(settings, k)
            elif k in base:
                snap[k] = base[k]
        except Exception:
            pass
    return json_safe(snap)

def validate_settings_window_snapshot(settings, requested_start, requested_end):
    requested_start = str(pd.Timestamp(requested_start).date())
    requested_end = str(pd.Timestamp(requested_end).date())
    observed = None
    for name in ['date_windows', 'date_window_pairs', 'analysis_windows', 'verification_windows', 'windows']:
        try:
            if hasattr(settings, name):
                observed = getattr(settings, name)
                break
            elif hasattr(settings, '__dict__') and name in settings.__dict__:
                observed = settings.__dict__[name]
                break
        except Exception:
            pass
    observed_safe = json_safe(observed)
    contains = False
    if isinstance(observed_safe, list):
        for item in observed_safe:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if str(item[0])[:10] == requested_start and str(item[1])[:10] == requested_end:
                    contains = True
                    break
    return {'expected_start_date': requested_start, 'expected_end_date': requested_end, 'observed_date_windows': observed_safe, 'date_windows_contains_requested': contains}

def try_parse_datetime_series(series):
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype='datetime64[ns]')
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', None]:
        try:
            if fmt is None:
                ts = pd.to_datetime(s, errors='coerce')
            else:
                ts = pd.to_datetime(s, errors='coerce', format=fmt)
            ts = ts.dropna()
            if not ts.empty:
                return ts
        except Exception:
            continue
    return pd.Series(dtype='datetime64[ns]')

def extract_time_signature_from_csv(path):
    out = {'file': str(path), 'exists': Path(path).exists(), 'n_rows': 0, 'time_col': None, 'n_time_values': 0, 'n_unique_times': 0, 'observed_start': None, 'observed_end': None, 'time_hash': None, 'status': 'missing', 'notes': ''}
    path = Path(path)
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
        out['n_rows'] = len(df)
    except Exception as e:
        out['status'] = f'read_error: {e}'
        return out
    candidates = []
    preferred = ['time', 'date', 'valid_time', 'valid_date', 'timestamp', 'datetime']
    for col in preferred:
        if col in df.columns:
            candidates.append(col)
    for col in df.columns:
        lc = str(col).lower()
        if any(tok in lc for tok in ['time', 'date', 'valid']):
            if col not in candidates:
                candidates.append(col)
    for col in candidates:
        ts = try_parse_datetime_series(df[col])
        if not ts.empty:
            out['time_col'] = col
            out['n_time_values'] = int(ts.shape[0])
            out['n_unique_times'] = int(ts.nunique())
            out['observed_start'] = str(ts.min())
            out['observed_end'] = str(ts.max())
            uniq_sorted = sorted({str(x) for x in ts.dropna().unique()})
            out['time_hash'] = hashlib.md5('|'.join(uniq_sorted).encode()).hexdigest()
            out['status'] = 'ok'
            return out
    out['status'] = 'unparseable_datetime_cols'
    return out

def compare_requested_to_observed(requested_start, requested_end, observed_start, observed_end, max_lead_h=168):
    if observed_start is None or observed_end is None:
        return 'no_observed_window'
    rs = pd.Timestamp(requested_start)
    re = pd.Timestamp(requested_end)
    os_ = pd.Timestamp(observed_start)
    oe = pd.Timestamp(observed_end)
    start_tol = pd.Timedelta(hours=0)
    end_tol = pd.Timedelta(hours=max_lead_h)
    start_ok = abs(os_ - rs) <= start_tol
    end_ok = abs(oe - re) <= end_tol
    if start_ok and end_ok:
        return 'ok'
    pilot_rs = pd.Timestamp('2020-01-01')
    pilot_re = pd.Timestamp('2020-03-31')
    if abs(os_ - pilot_rs) <= pd.Timedelta(hours=24) and abs(oe - (pilot_re + pd.Timedelta(hours=18))) <= pd.Timedelta(days=2):
        return 'pilot_fallback_suspected'
    return 'window_mismatch'

def _validation_priority(sig):
    path = str(sig.get('file') or '')
    name = Path(path).name
    parts = set(Path(path).parts)
    time_col = str(sig.get('time_col') or '')
    score = 0
    if name == 'regime_labels.csv':
        score += 1000
    elif name == 'deterministic_errors.csv':
        score += 900
    elif name == 'growth_errors.csv':
        score += 850
    elif name == 'blocking_errors.csv':
        score += 800
    elif name == 'deterministic_summary.csv':
        score += 400
    elif name == 'blocking_rmse.csv':
        score += 350
    if 'regimes' in parts:
        score += 120
    if 'deterministic' in parts:
        score += 80
    if 'growth' in parts:
        score += 60
    if 'blocking' in parts:
        score += 50
    if 'audit' in parts:
        score -= 1000
    if time_col in {'n_valid_times', 'n_times'}:
        score -= 1200
    if time_col in {'time', 'date', 'valid_time', 'valid_date', 'timestamp', 'datetime'}:
        score += 100
    try:
        score += min(int(sig.get('n_unique_times') or 0), 500) / 10.0
    except Exception:
        pass
    return float(score)

def collect_run_validation(run_root, requested_start, requested_end, max_lead_h=168):
    run_root = Path(run_root)
    preferred_basenames = ['regime_labels.csv', 'deterministic_errors.csv', 'growth_errors.csv', 'blocking_errors.csv', 'deterministic_summary.csv', 'blocking_rmse.csv']
    file_sigs, seen = [], set()
    for basename in preferred_basenames:
        for path in sorted(run_root.rglob(basename)):
            sp = str(path)
            if sp in seen:
                continue
            seen.add(sp)
            file_sigs.append(extract_time_signature_from_csv(path))
    for path in sorted(run_root.rglob('*.csv')):
        sp = str(path)
        if sp in seen:
            continue
        seen.add(sp)
        file_sigs.append(extract_time_signature_from_csv(path))
    ok_sigs = [sig for sig in file_sigs if sig.get('status') == 'ok']
    primary = None
    if ok_sigs:
        ok_sigs = sorted(ok_sigs, key=_validation_priority, reverse=True)
        primary = ok_sigs[0]
    if primary is None:
        status = 'no_valid_time_signature'
        primary = {'file': None, 'time_col': None, 'observed_start': None, 'observed_end': None, 'time_hash': None, 'n_unique_times': 0, 'status': status}
    else:
        status = compare_requested_to_observed(requested_start, requested_end, primary.get('observed_start'), primary.get('observed_end'), max_lead_h=max_lead_h)
    return {'requested_start': str(pd.Timestamp(requested_start)), 'requested_end': str(pd.Timestamp(requested_end)), 'requested_label': f'{pd.Timestamp(requested_start).date()}_{pd.Timestamp(requested_end).date()}', 'primary_file': primary.get('file'), 'primary_time_col': primary.get('time_col'), 'primary_observed_start': primary.get('observed_start'), 'primary_observed_end': primary.get('observed_end'), 'primary_time_hash': primary.get('time_hash'), 'primary_n_unique_times': primary.get('n_unique_times'), 'validation_status': status, 'all_files': file_sigs}

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(json_safe(payload), f, indent=2)

def write_run_sidecars(run_root, requested_payload, settings_snapshot, validation_payload=None):
    run_root = Path(run_root)
    write_json(run_root / 'requested_window.json', requested_payload)
    write_json(run_root / 'settings_snapshot.json', settings_snapshot)
    if validation_payload is not None:
        write_json(run_root / 'window_validation.json', validation_payload)
        pd.DataFrame(validation_payload['all_files']).to_csv(run_root / 'window_validation.csv', index=False)

def wipe_directory_contents(path):
    path = Path(path)
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

def main(job_path):
    job = json.load(open(job_path, 'r'))
    bundle_root = Path(job['bundle_root'])
    src_root = bundle_root / 'src'
    for p in [bundle_root, src_root]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from examples.default_settings import SETTINGS as BASE_SETTINGS
    from flagship_predictability import run_dataset_audit, run_truth_regimes, run_deterministic_atlas, run_regime_sensitivity, run_growth_diagnostics, run_blocking_verification

    tag = job['tag']
    bucket = job['bucket']
    season = job['season']
    start = pd.Timestamp(job['start'])
    end = pd.Timestamp(job['end'])
    run_root = Path(job['run_root'])
    run_root.mkdir(parents=True, exist_ok=True)
    if job.get('wipe_run_root', False):
        wipe_directory_contents(run_root)

    requested_payload = {
        'tag': tag, 'bucket': bucket, 'season': season,
        'requested_start': str(start), 'requested_end': str(end),
        'requested_start_date': str(start.date()), 'requested_end_date': str(end.date()),
        'requested_leads_h': job['sparse_leads_h'],
        'requested_n_regimes': job['default_n_regimes'],
        'requested_n_components': job['default_n_components'],
        'requested_blocking_threshold': job['default_blocking_threshold'],
        'requested_growth_lag_h': job['default_growth_lag_h'],
    }

    settings = deepcopy(BASE_SETTINGS)
    settings = configure_window(settings, start, end)
    settings = configure_leads(settings, job['sparse_leads_h'])
    settings = configure_regimes(settings, job['default_n_regimes'], job['default_n_components'], job['default_bootstraps'], job['default_block_len'])
    settings = configure_blocking(settings, job['default_blocking_threshold'])
    settings = configure_growth(settings, job['default_growth_lag_h'])

    settings_snapshot = settings_window_snapshot(settings)
    prerun_check = validate_settings_window_snapshot(settings, start, end)
    write_run_sidecars(run_root, requested_payload, {**settings_snapshot, '__prerun_window_check__': prerun_check})

    row_out = {
        'tag': tag, 'bucket': bucket, 'season': season,
        'requested_start': str(start.date()), 'requested_end': str(end.date()),
        'run_root': str(run_root),
        'prerun_date_windows_contains_requested': bool(prerun_check['date_windows_contains_requested']),
        'prerun_observed_date_windows': json.dumps(json_safe(prerun_check['observed_date_windows'])),
    }

    if not prerun_check['date_windows_contains_requested']:
        row_out.update({'audit_status': 'skipped', 'regimes_status': 'prerun_window_override_failed', 'deterministic_status': 'skipped', 'growth_status': 'skipped', 'blocking_status': 'skipped', 'regime_sensitivity_status': 'skipped', 'window_validation_status': 'prerun_window_override_failed'})
        write_json(run_root / 'window_result.json', row_out)
        return 0

    try:
        if job['run_dataset_audit']:
            run_dataset_audit(settings, output_root=run_root / 'audit')
            row_out['audit_status'] = 'ok'
        else:
            row_out['audit_status'] = 'skipped'
        gc.collect()
    except Exception as e:
        row_out['audit_status'] = f'error: {e}'

    try:
        if job['run_truth_regimes']:
            run_truth_regimes(settings, output_root=run_root / 'regimes')
            row_out['regimes_status'] = 'ok'
        else:
            row_out['regimes_status'] = 'skipped'
        gc.collect()
    except Exception as e:
        row_out['regimes_status'] = f'error: {e}'

    validation_payload = collect_run_validation(run_root, requested_start=start, requested_end=end, max_lead_h=max(job['sparse_leads_h']))
    row_out['window_validation_status'] = validation_payload['validation_status']
    row_out['primary_time_hash'] = validation_payload['primary_time_hash']
    row_out['primary_observed_start'] = validation_payload['primary_observed_start']
    row_out['primary_observed_end'] = validation_payload['primary_observed_end']
    write_run_sidecars(run_root, requested_payload, {**settings_snapshot, '__prerun_window_check__': prerun_check}, validation_payload=validation_payload)

    if job['stop_on_window_mismatch'] and validation_payload['validation_status'] != 'ok':
        row_out['deterministic_status'] = 'skipped_due_to_window_validation'
        row_out['growth_status'] = 'skipped_due_to_window_validation'
        row_out['blocking_status'] = 'skipped_due_to_window_validation'
        row_out['regime_sensitivity_status'] = 'skipped_due_to_window_validation'
        write_json(run_root / 'window_result.json', row_out)
        return 0

    try:
        if job['run_deterministic_atlas']:
            run_deterministic_atlas(settings, output_root=run_root / 'deterministic')
            row_out['deterministic_status'] = 'ok'
        else:
            row_out['deterministic_status'] = 'skipped'
        gc.collect()
    except Exception as e:
        row_out['deterministic_status'] = f'error: {e}'

    try:
        if job['run_growth']:
            run_growth_diagnostics(settings, output_root=run_root / 'growth')
            row_out['growth_status'] = 'ok'
        else:
            row_out['growth_status'] = 'skipped'
        gc.collect()
    except Exception as e:
        row_out['growth_status'] = f'error: {e}'

    try:
        if job['run_blocking']:
            run_blocking_verification(settings, output_root=run_root / 'blocking')
            row_out['blocking_status'] = 'ok'
        else:
            row_out['blocking_status'] = 'skipped'
        gc.collect()
    except Exception as e:
        row_out['blocking_status'] = f'error: {e}'

    try:
        if tag in set(job['run_regime_sensitivity_for']):
            run_regime_sensitivity(settings, output_root=run_root / 'regime_sensitivity')
            row_out['regime_sensitivity_status'] = 'ok'
        else:
            row_out['regime_sensitivity_status'] = 'skipped'
        gc.collect()
    except Exception as e:
        row_out['regime_sensitivity_status'] = f'error: {e}'

    write_json(run_root / 'window_result.json', row_out)
    return 0

if __name__ == '__main__':
    job_path = sys.argv[1]
    try:
        rc = main(job_path)
    except Exception:
        print(traceback.format_exc())
        try:
            job = json.load(open(job_path, 'r'))
            run_root = Path(job['run_root'])
            run_root.mkdir(parents=True, exist_ok=True)
            with open(run_root / 'window_worker_exception.txt', 'w') as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        rc = 1
    sys.exit(rc)
