# Az állapotcímkéket és a hozzájuk tartozó igazolásokat a mérésekhez rendeli.

r"""Állapotcímkék és külön kezelt, opcionális SKF-összehasonlítás.

Indítás:
    python .\src\cimkezes.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fordulatszam import AZONOS_MODSZEREK


CIMKE_MEZOK = ['allapot', 'igazolas', 'rekordra_ervenyes', 'hibatipus',
               'hibatipus_igazolt', 'bizonyitek_forras', 'cimkezo', 'megjegyzes']
ELL_MEZOK = ['skf_max_amplitudo', 'skf_max_frekvencia_hz', 'azonos_meres',
             'beallitasok_egyeznek', 'ellenorzes_forrasa', 'ellenorzo', 'megjegyzes']


def rekord_azonositok(meta: pd.DataFrame, spektrumok: list[np.ndarray]) -> pd.DataFrame:
    """Forrásfájlnévtől/sorsorrendtől és megjelenítési skálától független kulcs.

    A pontosság része a kulcsnak: perces és másodperces rekordot nem egyesítünk
    találomra. Ugyanazon rekord eltérő exportpontosságú másolata kézi egyeztetést kér.
    """
    if len(meta) != len(spektrumok):
        raise ValueError('Eltérő számú rekord és spektrum.')
    kulcsok, csoportok = [], []
    for i, (_, m) in enumerate(meta.iterrows()):
        leiro = [str(m[k]) for k in ['meresi_pont', 'idopont', 'idopont_pontossag',
                                    'mertekegyseg', 'detektalas', 'csatorna', 'spektrumhossz']]
        # Az eltérő tizedesírással megadott azonos érték közös kulcsot kap.
        # A kézi címke a fordulat újraszámítása után is ugyanahhoz a méréshez tartozik.
        export_speed = m.get('export_speed_hz', m.get('forgasi_frekvencia_hz', np.nan))
        leiro += [float(m['vegfrekvencia_export']).hex(), float(export_speed).hex()]
        h = hashlib.sha256(json.dumps(leiro, ensure_ascii=False).encode('utf-8'))
        amplitudo = np.asarray(spektrumok[i], dtype='<f8').copy()
        amplitudo[amplitudo == 0] = 0  # A két előjeles nulla egy értéket jelöl.
        h.update(amplitudo.tobytes())
        kulcsok.append(h.hexdigest())
        gep = m['gepegyseg_utvonal'] if pd.notna(m['gepegyseg_utvonal']) else m['csoport_utvonal']
        csoportok.append(hashlib.sha256(json.dumps([str(gep), str(m['meresi_nap'])]).encode()).hexdigest())
    eredmeny = meta.copy()
    eredmeny['rekord_id'] = kulcsok
    eredmeny['felosztasi_csoport'] = csoportok
    return eredmeny


def _csv_iras(tabla: pd.DataFrame, ut: Path, csak_uj: bool = False) -> bool:
    try:
        tabla.to_csv(ut, sep=';', decimal=',', encoding='utf-8-sig', index=False,
                     date_format='%Y-%m-%d %H:%M:%S', na_rep='', mode='x' if csak_uj else 'w')
        return True
    except FileExistsError:
        return False


def sablonok(tabla: pd.DataFrame, mappa: Path) -> None:
    mappa.mkdir(parents=True, exist_ok=True)
    info = ['rekord_id', 'meres_sorszam', 'meresi_pont', 'gepegyseg', 'gepresz', 'oldal',
            'idopont', 'idopont_pontossag', 'mertekegyseg', 'detektalas', 'meres_jelleg',
            'skala_allapot', 'spektrum_max_amplitudo', 'spektrum_max_frekvencia_hz']
    alap = tabla[[k for k in info if k in tabla]].drop_duplicates('rekord_id').copy()
    cimkek = alap.copy()
    for k in CIMKE_MEZOK:
        cimkek[k] = ''
    cimkek['allapot'], cimkek['igazolas'] = 'ismeretlen', 'nem_igazolt'
    cimkek['rekordra_ervenyes'], cimkek['hibatipus_igazolt'] = 'nem', 'nem'
    ell = alap.copy()
    for k in ELL_MEZOK:
        ell[k] = ''
    ell['azonos_meres'], ell['beallitasok_egyeznek'] = 'nem', 'nem'
    # Újrafuttatáskor megmaradnak a kézzel kitöltött fájlok.
    _csv_iras(cimkek, mappa / 'cimkek.csv', csak_uj=True)
    _csv_iras(ell, mappa / 'skf_ellenorzes.csv', csak_uj=True)


def szerkesztett_csv(ut: Path, oszlopok: list[str], ismert: set[str]) -> pd.DataFrame:
    if not ut.is_file():
        raise ValueError(f'Nem található a kitöltött fájl: {ut}')
    b = ut.read_bytes()
    try:
        szoveg = b.decode('utf-8-sig')
    except UnicodeError:
        szoveg = b.decode('cp1250')
    fejlec = next(csv.reader(io.StringIO(szoveg), delimiter=';'), [])
    if len(fejlec) != len(set(fejlec)):
        raise ValueError(f'{ut.name}: ismétlődő oszlopnév.')
    tabla = pd.read_csv(io.StringIO(szoveg), sep=';', dtype=str, keep_default_na=False)
    hianyzik = set(['rekord_id'] + oszlopok) - set(tabla.columns)
    if hianyzik:
        raise ValueError(f'{ut.name}: hiányzó oszlopok: {sorted(hianyzik)}')
    for k in ['rekord_id'] + oszlopok:
        tabla[k] = tabla[k].str.strip()
    if tabla.rekord_id.eq('').any() or tabla.rekord_id.duplicated().any():
        raise ValueError(f'{ut.name}: üres vagy ismétlődő rekord_id. Nincs bizonytalan összekapcsolás.')
    idegen = set(tabla.rekord_id) - ismert
    if idegen:
        raise ValueError(f'{ut.name}: {len(idegen)} azonosító nem szerepel ebben a jellemzőtáblában. Ellenőrizd a fájlpárt.')
    return tabla[['rekord_id'] + oszlopok]


def _szam(szoveg: str) -> float:
    if not str(szoveg).strip():
        return np.nan
    try:
        x = float(str(szoveg).replace(',', '.'))
    except ValueError as exc:
        raise ValueError(f'Nem értelmezhető ellenőrző szám: {szoveg!r}') from exc
    if not math.isfinite(x) or x < 0:
        raise ValueError('Az SKF-referencia nem negatív, véges szám legyen.')
    return x


def _skala_statusz(meres: pd.Series, skf_statusz: str) -> tuple[str, bool]:
    """A megadott profilt megkülönbözteti az automatikus feltételezéstől.

    Nem igazolja a profil helyességét: a megadott beállításokért a felhasználó
    felel. A kézi összevetésnél a teljes mérési beállítás egyezése is feltétel.
    """
    tengely = str(meres.get('spektrum_tengely', ''))
    allapot = str(meres.get('skala_allapot', ''))
    if tengely == 'index' or allapot == 'nem_ismert':
        return 'nincs_frekvenciaskala', False
    if tengely not in {'Hz', 'CPM', 'order'}:
        return 'hianyzo_skalaadat', False
    if allapot == 'felhasznalo_altal_megadott':
        return 'megadott_meresi_profil', True
    if skf_statusz == 'egyezik_a_megadott_turesen_belul':
        return 'rekord_skf_beallitassal_megerositve', True
    if allapot == 'automatikus_vagy_ellenorizetlen_profil':
        return 'feltetelezett_profil_ellenorizendo', False
    return 'ismeretlen_skalaallapot', False


def osszekapcsolas(tabla: pd.DataFrame, cimkek: pd.DataFrame, ell: pd.DataFrame | None = None,
                   amp_rel: float = 0.01, amp_abs: float = 1e-8,
                   freq_pontkoz: float = 0.5,
                   egyseg: str = 'mm/s', detektalas: str = 'RMS',
                   ismetlesek_kizarasa: bool = True) -> pd.DataFrame:
    for nev, ertek in [('amp_rel', amp_rel), ('amp_abs', amp_abs), ('freq_pontkoz', freq_pontkoz)]:
        if not math.isfinite(ertek) or ertek < 0:
            raise ValueError(f'{nev}: nem negatív, véges tűrés kell.')
    if ell is None:
        ell = pd.DataFrame(columns=['rekord_id'] + ELL_MEZOK)
    for t in (cimkek, ell):
        if t.rekord_id.duplicated().any() or t.rekord_id.eq('').any():
            raise ValueError('Dupla vagy üres címke-/ellenőrzésazonosító.')
        if set(t.rekord_id) - set(tabla.rekord_id):
            raise ValueError('Idegen azonosító a címke-/ellenőrzéstáblában.')
    c = cimkek.rename(columns={k: 'cimke_' + k for k in CIMKE_MEZOK})
    e = ell.rename(columns={k: 'ell_' + k for k in ELL_MEZOK})
    out = tabla.merge(c, on='rekord_id', how='left', validate='many_to_one', sort=False)
    out = out.merge(e, on='rekord_id', how='left', validate='many_to_one', sort=False)
    ismetelt = out.rekord_id.duplicated(keep=False)
    eredmenyek = []
    for i, m in out.iterrows():
        get = lambda k: '' if pd.isna(m[k]) else str(m[k]).strip()
        allapot = get('cimke_allapot') or 'ismeretlen'
        igazolas = get('cimke_igazolas') or 'nem_igazolt'
        hatokor = get('cimke_rekordra_ervenyes') or 'nem'
        hiba_igaz = get('cimke_hibatipus_igazolt') or 'nem'
        for mezo, ertek, valasztek in [
            ('allapot', allapot, {'ismeretlen', 'megfelelo', 'hibas'}),
            ('igazolas', igazolas, {'igazolt', 'nem_igazolt'}),
            ('rekordra_ervenyes', hatokor, {'igen', 'nem'}),
            ('hibatipus_igazolt', hiba_igaz, {'igen', 'nem'}),
        ]:
            if ertek not in valasztek:
                raise ValueError(f'{m.rekord_id[:12]}: hibás {mezo}: {ertek!r}')
        hibatipus = get('cimke_hibatipus')
        if hiba_igaz == 'igen' and (allapot != 'hibas' or not hibatipus or igazolas != 'igazolt'):
            raise ValueError('Igazolt hibatípushoz hibas állapot, igazolt címke és kitöltött hibatipus kell.')
        if allapot == 'megfelelo' and hibatipus:
            raise ValueError('Megfelelő állapot mellett nem lehet hibatípus. Javítsd az ellentmondást.')
        if allapot == 'ismeretlen':
            cimke_statusz = 'ismeretlen_allapot'
        elif igazolas != 'igazolt' or hatokor != 'igen':
            cimke_statusz = 'nem_igazolt_vagy_nem_rekordra_ervenyes'
        elif not get('cimke_bizonyitek_forras') or not get('cimke_cimkezo'):
            cimke_statusz = 'hianyzo_bizonyitek_vagy_cimkezo'
        else:
            cimke_statusz = 'igazolt_rekordcimke'

        amp = _szam(get('ell_skf_max_amplitudo'))
        freq = _szam(get('ell_skf_max_frekvencia_hz'))
        for k in ('ell_azonos_meres', 'ell_beallitasok_egyeznek'):
            if get(k) not in ('', 'igen', 'nem'):
                raise ValueError(f'{k}: igen vagy nem szükséges.')
        df = float(m['frekvenciapontkoz_hz'])
        program_amp = float(m['spektrum_max_amplitudo'])
        program_freq = float(m['spektrum_max_frekvencia_hz'])
        amp_elteres = abs(amp - program_amp)
        freq_elteres = abs(freq - program_freq)
        amp_tures = max(amp_abs, amp_rel * abs(amp)) if math.isfinite(amp) else np.nan
        freq_tures = freq_pontkoz * df if math.isfinite(df) else np.nan
        referencia_megkezdve = any(get('ell_' + k) for k in (
            'skf_max_amplitudo', 'skf_max_frekvencia_hz', 'ellenorzes_forrasa', 'ellenorzo'))
        referencia_megkezdve |= any(get('ell_' + k) == 'igen' for k in (
            'azonos_meres', 'beallitasok_egyeznek'))
        numerikus_ok = (all(math.isfinite(v) for v in (df, program_amp, program_freq))
                        and df > 0 and program_amp >= 0 and program_freq >= 0)
        if not referencia_megkezdve:
            ell_statusz = 'nem_megadott_opcionalis'
        elif not math.isfinite(amp) or not math.isfinite(freq):
            ell_statusz = 'hianyos_skf_referencia'
        elif not numerikus_ok:
            ell_statusz = 'nincs_ervenyes_program_skala_vagy_jellemzo'
        elif (get('ell_azonos_meres') != 'igen' or get('ell_beallitasok_egyeznek') != 'igen'
              or not get('ell_ellenorzes_forrasa') or not get('ell_ellenorzo')):
            ell_statusz = 'azonossag_es_beallitasok_nem_igazoltak'
        elif amp_elteres <= amp_tures and freq_elteres <= freq_tures:
            ell_statusz = 'egyezik_a_megadott_turesen_belul'
        else:
            ell_statusz = 'elteres_ellenorizendo'

        skala_statusz, skala_ok = _skala_statusz(m, ell_statusz)
        adat_okok = []
        if 'fordulat_modszer' in m.index and m.fordulat_modszer not in AZONOS_MODSZEREK:
            adat_okok.append('elavult_fordulatmodszer_uj_jellemzokinyeres_szukseges')
        if not numerikus_ok:
            adat_okok.append('ervenytelen_numerikus_jellemzo_vagy_pontkoz')
        if not skala_ok:
            adat_okok.append(skala_statusz)
        # A megkezdett összevetés hiányos vagy eltérő eredményét külön jelölöm.
        if ell_statusz not in {'nem_megadott_opcionalis', 'egyezik_a_megadott_turesen_belul'}:
            adat_okok.append(ell_statusz)
        if m['jellemzok_allapot'] != 'leiro_jellemzok_kiszamitva':
            adat_okok.append('jellemzok_ellenorizendok')
        if pd.isna(m['gepegyseg_utvonal']) or not str(m['gepegyseg_utvonal']).strip():
            adat_okok.append('ismeretlen_gepegyseg')
        if ismetlesek_kizarasa and ismetelt.iloc[i]:
            adat_okok.append('ismetelt_rekord')
        okok = []
        if cimke_statusz != 'igazolt_rekordcimke':
            okok.append(cimke_statusz)
        okok.extend(adat_okok)
        if m['mertekegyseg'] != egyseg or m['detektalas'] != detektalas:
            okok.append('mas_meresfajta')
        eredmenyek.append({
            'cimke_allapot_ertekelt': allapot, 'cimke_statusz': cimke_statusz,
            'skf_ellenorzes_statusz': ell_statusz,
            'skala_felhasznalhatosag': skala_statusz,
            'adatminoseg_statusz': 'elokesziteshez_megfelelo' if not adat_okok else 'ellenorizendo',
            'adatminoseg_okai': '; '.join(adat_okok),
            'skf_amplitudo_abs_elteres': amp_elteres, 'skf_frekvencia_abs_elteres_hz': freq_elteres,
            'skf_amplitudo_tures': amp_tures, 'skf_frekvencia_tures_hz': freq_tures,
            'tanitasi_jelolt': not okok, 'kizaras_okai': '; '.join(okok),
            'cel_hibas': 1 if allapot == 'hibas' and cimke_statusz == 'igazolt_rekordcimke'
                         else 0 if allapot == 'megfelelo' and cimke_statusz == 'igazolt_rekordcimke' else np.nan,
            'cel_hibatipus': hibatipus if hiba_igaz == 'igen' and cimke_statusz == 'igazolt_rekordcimke' else '',
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(eredmenyek)], axis=1)


def feldolgozas(tabla: pd.DataFrame, mappa: Path, cimke_fajl: Path | None = None,
                ell_fajl: Path | None = None, amp_rel: float = 0.01, amp_abs: float = 1e-8,
                freq_pontkoz: float = 0.5, egyseg: str = 'mm/s', detektalas: str = 'RMS') -> pd.DataFrame:
    kell = {'rekord_id', 'felosztasi_csoport', 'gepegyseg_utvonal', 'mertekegyseg', 'detektalas',
            'jellemzok_allapot', 'frekvenciapontkoz_hz', 'spektrum_max_amplitudo', 'spektrum_max_frekvencia_hz'}
    if kell - set(tabla):
        raise ValueError('Régi vagy hiányos jellemzőtábla. Futtasd újra a 7. változat beolvasóját.')
    if tabla.empty or tabla.rekord_id.isna().any() or tabla.rekord_id.astype(str).str.strip().eq('').any():
        raise ValueError('Üres jellemzőtábla vagy hiányzó rekordazonosító.')
    sablonok(tabla, mappa)
    ismert = set(tabla.rekord_id)
    c = szerkesztett_csv(cimke_fajl or mappa / 'cimkek.csv', CIMKE_MEZOK, ismert)
    # A kézi ellenőrzést az állapotcímkétől külön tárolom.
    e = szerkesztett_csv(ell_fajl or mappa / 'skf_ellenorzes.csv', ELL_MEZOK, ismert)
    eredmeny = osszekapcsolas(tabla, c, e, amp_rel, amp_abs, freq_pontkoz, egyseg, detektalas)
    _csv_iras(eredmeny, mappa / 'cimkezett_jellemzok.csv')
    jeloltek = eredmeny[eredmeny.tanitasi_jelolt].copy()
    _csv_iras(jeloltek, mappa / 'tanitasi_jeloltek.csv')
    kizarasok = eredmeny.kizaras_okai.str.split('; ').explode()
    osztalyok_megvannak = {0, 1}.issubset(set(jeloltek.cel_hibas.dropna()))
    jelentes = {
        'cimkezes_valtozat': 8,
        'kulso_cimkefajl': str(cimke_fajl.resolve()) if cimke_fajl else None,
        'kulso_skf_ellenorzes_fajl': str(ell_fajl.resolve()) if ell_fajl else None,
        'rekordok': len(eredmeny), 'tanitasi_jeloltek': len(jeloltek),
        'jelolt_csoportok': int(jeloltek.felosztasi_csoport.nunique()),
        'allapotok_jeloltek': jeloltek.cimke_allapot_ertekelt.value_counts().to_dict(),
        'cimke_statuszok': eredmeny.cimke_statusz.value_counts().to_dict(),
        'skf_statuszok': eredmeny.skf_ellenorzes_statusz.value_counts().to_dict(),
        'skf_osszevetes_rekordonkent_kotelezo': False,
        'skala_statuszok': eredmeny.skala_felhasznalhatosag.value_counts().to_dict(),
        'adatminoseg_statuszok': eredmeny.adatminoseg_statusz.value_counts().to_dict(),
        'kizarasi_okok': kizarasok[kizarasok.ne('')].value_counts().to_dict(),
        'mindket_binaris_osztaly_a_jeloltek_kozott': osztalyok_megvannak,
        'kivalasztott_egyseg': egyseg, 'kivalasztott_detektalas': detektalas,
        'turesek': {'amplitudo_rel': amp_rel, 'amplitudo_abs': amp_abs, 'frekvencia_pontkoz_szorzo': freq_pontkoz},
        'megjegyzes': 'A kézi SKF-összevetés opcionális; a megadott eltérések ellenőrizendők. '
                      'A megadott profil nem automatikusan igazolt profil. '
                      'Nincs tanítás vagy tesztfelosztás. Azonos gép+napi csoport együtt tartandó. '
                      'Külön fájlok átfedése és eltérő időpontpontosság további ellenőrzést kér.',
    }
    (mappa / 'cimkezes_jelentes.json').write_text(json.dumps(jelentes, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Címkék: {mappa / "cimkek.csv"} (meglévő fájlt nem írunk felül)')
    print(f'Opcionális kézi SKF-összehasonlítás: {mappa / "skf_ellenorzes.csv"}')
    print(f'Tanítási jelöltek: {len(jeloltek)}/{len(eredmeny)}; külön csoportok: {jelentes["jelolt_csoportok"]}')
    if jelentes['kizarasi_okok']:
        print('Kizárási okok:', jelentes['kizarasi_okok'])
    if not osztalyok_megvannak:
        print('A bináris tanításhoz még nincs mindkét igazolt állapotból kiválasztott mérés.')
    print('Ez előkészítés, nem betanított modell és nem végleges tanító/tesztfelosztás.')
    return eredmeny


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jellemzok_csv', type=Path, nargs='?')
    parser.add_argument('--cimkek', type=Path)
    parser.add_argument('--skf-ellenorzes', type=Path)
    parser.add_argument('--egyseg', default='mm/s')
    parser.add_argument('--detektalas', default='RMS')
    parser.add_argument('--amplitudo-rel-tures', type=float, default=0.01)
    parser.add_argument('--amplitudo-abs-tures', type=float, default=1e-8)
    parser.add_argument('--frekvencia-pontkoz-tures', type=float, default=0.5)
    args = parser.parse_args()
    try:
        if args.jellemzok_csv is None:
            gyoker = Path(__file__).resolve().parent.parent / 'outputs'
            fajlok = []
            for ut in sorted(gyoker.rglob('jellemzok.csv')):
                with ut.open(encoding='utf-8-sig', newline='') as f:
                    if 'rekord_id' in next(csv.reader(f, delimiter=';'), []):
                        fajlok.append(ut)
            if not fajlok:
                raise ValueError('Nincs 7. változattal készült jellemzok.csv. Először futtasd a beolvasót.')
            for i, ut in enumerate(fajlok, 1):
                print(f'{i}. {ut.relative_to(gyoker)}')
            if len(fajlok) == 1:
                args.jellemzok_csv = fajlok[0]
            else:
                valasz = input('A jellemzőtábla sorszáma (q = kilépés): ').strip()
                if valasz.lower() == 'q':
                    return
                if not valasz.isdecimal() or not 1 <= int(valasz) <= len(fajlok):
                    raise ValueError('Érvénytelen sorszám.')
                args.jellemzok_csv = fajlok[int(valasz) - 1]
        tabla = pd.read_csv(args.jellemzok_csv, sep=';', decimal=',', encoding='utf-8-sig',
                            dtype={'rekord_id': str, 'felosztasi_csoport': str})
        feldolgozas(tabla, args.jellemzok_csv.parent, args.cimkek, args.skf_ellenorzes,
                     args.amplitudo_rel_tures, args.amplitudo_abs_tures,
                     args.frekvencia_pontkoz_tures, args.egyseg, args.detektalas)
    except (ValueError, OSError, UnicodeError, EOFError, pd.errors.ParserError) as exc:
        parser.exit(1, f'Címkézési hiba: {exc}\n')


if __name__ == '__main__':
    main()
