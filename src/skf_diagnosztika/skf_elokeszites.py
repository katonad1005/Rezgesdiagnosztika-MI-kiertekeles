# A frekvenciatengelyt és a spektrumjellemzőket előkészíti a tünetvizsgálathoz.

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .skf_jellemzok import Axis
from .skf_fajlkezeles import Spectrum, number
from fordulatszam import AZONOS_MODSZEREK, becsles as fordulat_becsles

STANDARD_ORIGIN = "automatikus_standardprofil: 0 Hz–End Frequency [Hz], mindkét végponttal; feltételezés"


def optional_number(value):
    try:
        return number(value) if value is not None and str(value).strip() else None
    except (ValueError, TypeError):
        return None


def clean_features(values):
    result = {}
    for key, value in dict(values or {}).items():
        if not (key.startswith(("harm_", "csucs_", "spektrum_max_", "dominans_pozitiv_")) or
                key.startswith('fordulat_') or key in {
                    "forgasi_frekvencia_hz", "fordulatszam_perc", "jellemzok_allapot",
                    "frekvencia_jellemzok_allapot", "lokalis_csucsok_szama", "meres_jelleg"}):
            continue
        if isinstance(value, str) or value is None:
            result[key] = value
        else:
            result[key] = optional_number(value)
    return result


def axis_from_values(values, unit, speed):
    if unit == "index":
        return None
    factors = {"Hz": 1, "CPM": 1 / 60, "order": speed}
    factor = factors.get(unit)
    if factor is None or factor <= 0:
        raise ValueError("A kapott tengely nem váltható Hz-re")
    f = [number(value) * factor for value in values]
    if len(f) < 2 or f[0] < 0:
        raise ValueError("Érvénytelen előfeldolgozott tengely")
    step = (f[-1] - f[0]) / (len(f) - 1)
    if step <= 0 or any(abs(x - (f[0] + i * step)) > max(step * 1e-7, 1e-10) for i, x in enumerate(f)):
        raise ValueError("Az előfeldolgozott tengely nem egyenletesen növekvő")
    return Axis(f[0], step, len(f))


def extract_features(record, axis, speed, technical):
    """Az önálló útvonal leíró jellemzői. Nem számít időjel-RMS-t vagy kurtózist."""
    v = record.values
    if record.errors or not v:
        return {"jellemzok_allapot": "hibas_adat"}
    index = max(range(len(v)), key=v.__getitem__)
    result = {"jellemzok_allapot": "leiro_jellemzok_kiszamitva",
              "spektrum_max_amplitudo": v[index], "spektrum_max_index": index,
              "spektrum_max_frekvencia_hz": axis.frequency(index) if axis else None}
    candidates = [i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]
                  and v[i] >= technical["minimum_relativ_amplitudo"] * v[index]]
    result["lokalis_csucsok_szama"] = len(candidates)
    for rank, i in enumerate(sorted(candidates, key=lambda i: (-v[i], i))[:5], 1):
        result.update({f"csucs_{rank}_index": i, f"csucs_{rank}_amplitudo": v[i],
                       f"csucs_{rank}_frekvencia_hz": axis.frequency(i) if axis else None})
    if axis and speed:
        for k in (1, 2, 3):
            target, tol = k * speed, max(axis.step, .02 * k * speed)
            result.update({f"harm_{k}_cel_hz": target, f"harm_{k}_tures_hz": tol})
            if tol >= speed / 2 or target - tol < axis.start or target + tol > axis.end:
                result[f"harm_{k}_allapot"] = "nem_szamithato"
                continue
            lo = max(0, math.ceil((target - tol - axis.start) / axis.step - 1e-9))
            hi = min(len(v) - 1, math.floor((target + tol - axis.start) / axis.step + 1e-9))
            i = max(range(lo, hi + 1), key=v.__getitem__)
            result.update({f"harm_{k}_index": i, f"harm_{k}_amplitudo": v[i],
                           f"harm_{k}_frekvencia_hz": axis.frequency(i),
                           f"harm_{k}_allapot": "savmaximum_kiszamitva"})
        a = result.get("harm_1_amplitudo")
        if a and a > 0:
            for k in (2, 3):
                if result.get(f"harm_{k}_amplitudo") is not None:
                    result[f"harm_{k}_1_amplitudo_arany"] = result[f"harm_{k}_amplitudo"] / a
    return result


def automatic_context(record, settings, technical, *, axis=None, axis_supplied=False,
                      axis_origin="", features=None, feature_origin="onallo_jellemzokinyeres"):
    warnings, issue = [], ""
    conditional = False
    if axis_supplied:
        if axis is None:
            issue = "Az előfeldolgozás csak pontindexet tartalmaz; a frekvencia nincs kitalálva"
        conditional = "automatikus_standardprofil" in axis_origin or "ellenorizetlen" in axis_origin
    elif not record.errors:
        unit = settings.get("frekvenciaegyseg")
        start = optional_number(settings.get("kezdo_frekvencia_hz"))
        if unit in {"Hz", "CPM"} and start is not None:
            end = record.end * (1 if unit == "Hz" else 1 / 60)
            denominator = len(record.values) - 1 if settings.get("vegpontok") == "ket_vegpont" else len(record.values)
            if end <= start or denominator < 1:
                issue = "Érvénytelen megadott frekvenciatartomány"
            else:
                axis = Axis(start, (end - start) / denominator, len(record.values))
                axis_origin = "meglévő SKF-skálabeállítás"
                conditional = not bool(settings.get("skala_ellenorzott"))
        elif unit not in (None, "") or start is not None:
            issue = "Hiányos meglévő skálabeállítás; az automatikus profil nem írja felül"
        elif re.search(r"(?:^|[\\/_ -])(zoom|order)(?:$|[\\/_ -])", record.point, re.I):
            issue = "Zoom/order pontnál nincs egyértelmű frekvenciaskála; nincs automatikus 0 Hz feltételezés"
        elif record.end and len(record.values) >= 2:
            axis = Axis(0, record.end / (len(record.values) - 1), len(record.values))
            axis_origin, conditional = STANDARD_ORIGIN, True
    if conditional:
        warnings.append("A frekvenciaskála automatikus/ellenőrizetlen profilból származik; a hibairány feltételes")
    if axis and axis.count != len(record.values):
        raise ValueError("A kapott tengely és a spektrum pontszáma eltér")
    frequencies = [axis.frequency(i) for i in range(axis.count)] if axis else None
    estimate = fordulat_becsles(record.values, frequencies)
    if record.errors:
        estimate = fordulat_becsles([])
        estimate['fordulat_allapot'] = 'hibas_spektrum_vagy_tengely'
    speed, speed_origin = estimate['forgasi_frekvencia_hz'], estimate['fordulat_forrasa']
    speed_issue = '' if speed is not None else estimate['fordulat_allapot']
    if speed is not None:
        warnings.append('Vállalati 5–100 Hz sávmaximum alapján becsült fordulat; a Speed mezőt nem használjuk.')
        if not estimate['fordulat_teljes_sav']:
            warnings.append('A keresési sáv csak részben áll rendelkezésre; a meglévő pontok maximumát használtuk.')
        if estimate['fordulat_maximumok_szama'] > 1:
            warnings.append('Több azonos sávmaximum: a kisebb frekvenciát választottuk.')
    if settings.get('fordulati_frekvencia_hz') is not None or settings.get('fordulat_igazolt'):
        warnings.append('A korábbi kézi fordulatbeállítást a vállalati sávmaximumos eljárás felülírja.')
    feature_speed = optional_number((features or {}).get('forgasi_frekvencia_hz'))
    reusable = (features is not None and features.get('fordulat_modszer') in AZONOS_MODSZEREK and
                speed is not None and feature_speed is not None and
                math.isclose(feature_speed, speed, rel_tol=1e-10, abs_tol=1e-10))
    descriptor = clean_features(features) if reusable else extract_features(record, axis, speed, technical)
    descriptor.update(estimate)
    return {"axis": axis, "axis_origin": axis_origin, "axis_issue": issue,
            "axis_conditional": conditional, "speed": speed, "speed_origin": speed_origin,
            "speed_conditional": speed is not None, "speed_estimate": estimate,
            "speed_issue": speed_issue, "warnings": warnings, "features": descriptor,
            "feature_origin": feature_origin}


def from_main(meta, spectra, axes, features, source):
    """Nincs CSV-újraolvasás; a főprogram stabil rekordazonosítóját is megtartja."""
    rows = meta.to_dict("records") if hasattr(meta, "to_dict") else list(meta)
    feature_rows = features.to_dict("records") if hasattr(features, "to_dict") else list(features)
    if not len(rows) == len(spectra) == len(axes) == len(feature_rows):
        raise ValueError("Az átadott mérések, tengelyek és jellemzősorok száma eltér")
    prepared = []
    for m, values, x, feature in zip(rows, spectra, axes, feature_rows):
        v = [float(value) for value in values]
        errors = []
        if len(v) < 2 or any(not math.isfinite(a) or a < 0 for a in v) or not any(v):
            errors.append("Hibás/hiányzó/nem pozitív amplitúdóspektrum")
        speed = optional_number(m.get('export_speed_hz', m.get('forgasi_frekvencia_hz')))
        date = str(m.get("idopont", ""))
        metadata = {"Point Path": str(m.get("meresi_pont", "")), "DTS": date,
                    "Unit": str(m.get("mertekegyseg", "")), "Detection": str(m.get("detektalas", "")),
                    "Channel": str(m.get("csatorna", "")), "Lines": str(len(v)),
                    "End Frequency": str(m.get("vegfrekvencia_export", "")),
                    "Speed (Hz)": str(m.get('export_speed_nyers', speed if speed is not None else 'N/A'))}
        ident = str(m.get("rekord_id") or hashlib.sha256(json.dumps([metadata, v], sort_keys=True).encode()).hexdigest())
        record = Spectrum(ident, metadata, v, speed, optional_number(m.get("vegfrekvencia_export")), date,
                          [{"fajl": str(Path(source).resolve()), "sor": int(m.get("csv_sor", 0))}], errors)
        axis, axis_issue = None, ""
        try:
            unit = str(m.get('spektrum_tengely', 'index'))
            reference = (optional_number(feature.get('forgasi_frekvencia_hz'))
                         if feature.get('fordulat_modszer') in AZONOS_MODSZEREK and
                         m.get('skala_export_egyseg') in {'Hz', 'CPM'} else None)
            if unit != 'order' or reference is not None:
                axis = axis_from_values(x, unit, reference)
        except ValueError as exc:
            axis_issue = str(exc)
            record.errors.append(axis_issue)
        prepared.append((record, axis, clean_features(feature), str(m.get("skala_beallitas_forrasa", "fo_program_tengelye")), m))
    return prepared
