# A mentett modell alapján állapotbecslést készít a jellemzőtábla soraiból.

r"""Mentett MI-modell alkalmazása.

Indítás:
    python .\src\mi_becsles.py jellemzok.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from mi_modell import FORMATUM, verziok
except ModuleNotFoundError:
    if __name__ == '__main__':
        raise SystemExit('Hiányzó MI-csomag. Telepítés: python -m pip install -r requirements_mi.txt')
    raise
from tanito_adatok import JELLEMZOK
from fordulatszam import AZONOS_MODSZEREK, MODSZER


PROJEKT = Path(__file__).resolve().parent.parent


def modell_betoltes(ut: Path | None = None, modell_mappa: Path | None = None) -> tuple[object, dict]:
    if ut is None:
        statusz_ut = (modell_mappa or PROJEKT / 'outputs' / 'mi_modellek') / 'utolso_tanitas.json'
        if not statusz_ut.is_file():
            raise ValueError('Nincs mentett MI-modell. Először a mi_tanitas.py programot kell sikeresen lefuttatni.')
        statusz = json.loads(statusz_ut.read_text(encoding='utf-8'))
        if statusz.get('allapot') != 'sikeres':
            raise ValueError('Az utolsó tanítás nem volt sikeres: ' + statusz.get('hiba', statusz.get('allapot', 'ismeretlen')))
        ut = Path(statusz['modell_utvonal'])
    ut = Path(ut)
    meta = json.loads(ut.with_suffix('.json').read_text(encoding='utf-8'))
    if (meta.get('formatum') != FORMATUM or meta.get('jellemzo_oszlopok') != JELLEMZOK or
            meta.get('fordulat_modszer') not in AZONOS_MODSZEREK):
        raise ValueError('Eltérő modellformátum vagy jellemzőkészlet; a jelenlegi programmal újra kell tanítani.')
    most = verziok()
    for nev in ('scikit-learn', 'numpy', 'scipy', 'joblib'):
        if meta.get('verziok', {}).get(nev) != most[nev]:
            raise ValueError(f'Eltérő {nev}-verzió. Ugyanaz a tanítási környezet vagy újratanítás szükséges.')
    if meta.get('verziok', {}).get('python', '').split('.')[:2] != most['python'].split('.')[:2]:
        raise ValueError('Eltérő Python fő-/alverzió a mentett modellhez képest.')
    if hashlib.sha256(ut.read_bytes()).hexdigest() != meta.get('modell_sha256'):
        raise ValueError('A modellfájl és a leíró JSON nem tartozik össze vagy a modellfájl megváltozott.')
    csomag = joblib.load(ut)
    if (not isinstance(csomag, dict) or csomag.get('formatum') != FORMATUM or
            csomag.get('fordulat_modszer') not in AZONOS_MODSZEREK or not hasattr(csomag.get('pipeline'), 'predict')):
        raise ValueError('Nem támogatott modellcsomag.')
    meta['modell_utvonal'] = str(ut.resolve())
    return csomag['pipeline'], meta


def becsles_tabla(tabla: pd.DataFrame, modell, meta: dict) -> tuple[pd.DataFrame, dict]:
    kell = set(JELLEMZOK + ['rekord_id', 'mertekegyseg', 'detektalas', 'jellemzok_allapot',
                          'skala_allapot', 'spektrum_tengely', 'frekvenciapontkoz_hz', 'harmonikus_tures_rel',
                          'fordulat_modszer'])
    if kell - set(tabla):
        raise ValueError(f'Hiányos becslési jellemzőtábla: {sorted(kell - set(tabla))}. Futtasd újra a jellemzőkinyerést.')
    if meta.get('fordulat_modszer') not in AZONOS_MODSZEREK:
        raise ValueError('A modell fordulatmódszere nem egyezik az aktuális vállalati eljárással; újratanítás szükséges.')
    tabla = tabla.reset_index(drop=True)
    x = pd.DataFrame(index=tabla.index)
    rossz_szam = pd.Series(False, index=tabla.index)
    for k in JELLEMZOK:
        szoveg = tabla[k].fillna('').astype(str).str.strip()
        x[k] = pd.to_numeric(szoveg.str.replace(',', '.', regex=False), errors='coerce')
        rossz_szam |= szoveg.ne('') & (x[k].isna() | ~np.isfinite(x[k]))
    rossz_szam |= x.lt(0).any(axis=1) | x.isna().all(axis=1)
    df = pd.to_numeric(tabla.frekvenciapontkoz_hz.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    tures = pd.to_numeric(tabla.harmonikus_tures_rel.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    maszk, sorok = [], []
    tanitott = set(meta.get('tanitott_rekord_idk', []))
    tesztelt = set(meta.get('teszt_rekord_idk', []))
    gepek = set(meta.get('tanitott_gepek', []))
    for i, m in tabla.iterrows():
        okok = []
        if m.fordulat_modszer not in AZONOS_MODSZEREK:
            okok.append('eltero_vagy_elavult_fordulatmodszer')
        rid = '' if pd.isna(m.rekord_id) else str(m.rekord_id).strip()
        if not rid:
            okok.append('hianyzo_rekordazonosito')
        if m.mertekegyseg != meta['mertekegyseg'] or m.detektalas != meta['detektalas']:
            okok.append('a_modelltol_eltero_meresfajta')
        if m.jellemzok_allapot != 'leiro_jellemzok_kiszamitva' or rossz_szam.iloc[i]:
            okok.append('ervenytelen_modelljellemzok')
        if (m.skala_allapot != 'felhasznalo_altal_megadott' or m.spektrum_tengely not in {'Hz', 'CPM', 'order'}
                or not np.isfinite(df.iloc[i]) or df.iloc[i] <= 0):
            okok.append('nem_tisztazott_frekvenciaskala')
        if not np.isfinite(tures.iloc[i]) or not np.isclose(tures.iloc[i], meta['harmonikus_tures_rel'], rtol=0, atol=1e-12):
            okok.append('eltero_vagy_ismeretlen_harmonikussav_beallitas')
        maszk.append(not okok)
        sorok.append({
            **{k: m.get(k, '') for k in ['rekord_id', 'meresi_pont', 'gepegyseg_utvonal', 'idopont', 'mertekegyseg', 'detektalas',
                                       'forgasi_frekvencia_hz', 'fordulatszam_perc', 'fordulat_modszer', 'fordulat_allapot']},
            'mi_allapot': 'KIERTEKELVE' if not okok else 'NEM_ERTEKELHETO',
            'becsult_hibas': np.nan, 'becsult_allapot': '', 'kihagyas_oka': '; '.join(okok),
            'modell_tipus': meta['modell_tipus'],
            'tanitasban_szerepelt': rid in tanitott, 'tesztben_szerepelt': rid in tesztelt,
            'tanitasi_gephez_tartozik': m.get('gepegyseg_utvonal', '') in gepek,
        })
    oszlopok = ['rekord_id', 'meresi_pont', 'gepegyseg_utvonal', 'idopont', 'mertekegyseg', 'detektalas',
                'forgasi_frekvencia_hz', 'fordulatszam_perc', 'fordulat_modszer', 'fordulat_allapot',
                'mi_allapot', 'becsult_hibas', 'becsult_allapot', 'kihagyas_oka', 'modell_tipus',
                'tanitasban_szerepelt', 'tesztben_szerepelt', 'tanitasi_gephez_tartozik']
    eredmeny = pd.DataFrame(sorok, columns=oszlopok)
    maszk = np.asarray(maszk, dtype=bool)
    if maszk.any():
        becsult = np.asarray(modell.predict(x.loc[maszk, JELLEMZOK]), dtype=int)
        if not set(becsult) <= {0, 1}:
            raise ValueError('A mentett modell nem bináris állapotot adott vissza.')
        eredmeny.loc[maszk, 'becsult_hibas'] = becsult
        eredmeny.loc[maszk, 'becsult_allapot'] = np.where(becsult == 1, 'hibas', 'nem_hibas')
    jelent = {
        'formatum': FORMATUM, 'modell_tipus': meta['modell_tipus'],
        'fordulat_modszer': MODSZER,
        'modell_utvonal': meta.get('modell_utvonal', ''), 'modell_sha256': meta['modell_sha256'],
        'rekordok': len(eredmeny), 'ertekelt_rekordok': int(maszk.sum()),
        'becsult_hibas': int(eredmeny.becsult_hibas.eq(1).sum()),
        'becsult_nem_hibas': int(eredmeny.becsult_hibas.eq(0).sum()),
        'kihagyott_rekordok': int((~maszk).sum()),
        'tanitasban_szerepelt': int(eredmeny.tanitasban_szerepelt.sum()),
        'tesztben_szerepelt': int(eredmeny.tesztben_szerepelt.sum()),
        'megjegyzes': 'Modellbecslés, nem igazolt állapotcímke. Nem módosítja a tanítási címkéket. '
                      'Új gépek vagy eltérő üzemi feltételek teljesítményét ez a futás nem igazolja.',
    }
    return eredmeny, jelent


def mentes(tabla: pd.DataFrame, jelent: dict, mappa: Path) -> None:
    mappa.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(mappa / 'mi_eredmenyek.csv', sep=';', decimal=',', encoding='utf-8-sig', index=False, na_rep='')
    (mappa / 'mi_jelentes.json').write_text(
        json.dumps(jelent, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'MI-becslés ({jelent["modell_tipus"]}): {jelent["ertekelt_rekordok"]}/{jelent["rekordok"]} rekord; '
          f'hibásnak becsült: {jelent["becsult_hibas"]}; kihagyott: {jelent["kihagyott_rekordok"]}.')
    print(f'Részletes eredmények: {(mappa / "mi_eredmenyek.csv").resolve()}')


def jellemzok_betoltes(ut: Path) -> pd.DataFrame:
    with ut.open(encoding='utf-8-sig', newline='') as f:
        fejlec = next(csv.reader(f, delimiter=';'), [])
    if len(fejlec) != len(set(fejlec)):
        raise ValueError('Ismétlődő oszlopnév a jellemzőtáblában.')
    tabla = pd.read_csv(ut, sep=';', decimal=',', encoding='utf-8-sig', dtype={'rekord_id': str})
    if 'harmonikus_tures_rel' not in tabla:
        beallitas_ut = ut.parent / 'abra_beallitasok.json'
        b = json.loads(beallitas_ut.read_text(encoding='utf-8')) if beallitas_ut.is_file() else {}
        if not isinstance(b, dict) or not isinstance(b.get('jellemzok', {}), dict):
            raise ValueError('Hibás abra_beallitasok.json.')
        tabla['harmonikus_tures_rel'] = b.get('jellemzok', {}).get('harmonikus_tures_rel', np.nan)
    return tabla


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jellemzok_csv', type=Path)
    parser.add_argument('--modell', type=Path)
    parser.add_argument('--kimenet', type=Path)
    args = parser.parse_args(argv)
    try:
        modell, meta = modell_betoltes(args.modell)
        tabla = jellemzok_betoltes(args.jellemzok_csv)
        eredmeny, jelent = becsles_tabla(tabla, modell, meta)
        mappa = args.kimenet or args.jellemzok_csv.parent / 'mi_becsles' / meta['modell_sha256'][:10]
        mentes(eredmeny, jelent, mappa)
        return 0 if jelent['ertekelt_rekordok'] else 2
    except (ValueError, OSError, ImportError, TypeError, KeyError, EOFError, pd.errors.ParserError) as exc:
        print(f'MI-becslési hiba: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
