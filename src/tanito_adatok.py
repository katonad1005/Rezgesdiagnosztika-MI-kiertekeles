# A mérések jellemzőit az igazolt állapotcímkékkel közös tanítótáblába rendezi.

r"""Közös, ellenőrzött tanítóadatok összeállítása. Indítás: python .\src\tanito_adatok.py

A friss címkéket újra ellenőrzi; nem a korábbi tanitasi_jeloltek.csv fájlokat másolja.
Nem tanít modellt, nem pótolja a hiányzó jellemzőket és nem talál ki gépállapotot.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cimkezes import CIMKE_MEZOK, ELL_MEZOK, osszekapcsolas, szerkesztett_csv
from fordulatszam import AZONOS_MODSZEREK, MODSZER


PROJEKT = Path(__file__).resolve().parent.parent
JELLEMZOK = [
    'dominans_pozitiv_amplitudo', 'dominans_pozitiv_frekvencia_hz',
    'harm_1_amplitudo', 'harm_2_amplitudo', 'harm_3_amplitudo',
    'harm_2_1_amplitudo_arany', 'harm_3_1_amplitudo_arany',
]
OSSZEVETENDO = JELLEMZOK + [
    'frekvenciapontkoz_hz', 'spektrum_max_amplitudo', 'spektrum_max_frekvencia_hz',
    'harm_1_tures_hz', 'harm_2_tures_hz', 'harm_3_tures_hz',
]
KOTELEZO = set(OSSZEVETENDO + [
    'rekord_id', 'felosztasi_csoport', 'gepegyseg_utvonal', 'meresi_pont',
    'idopont', 'idopont_pontossag', 'csatorna', 'mertekegyseg', 'detektalas',
    'jellemzok_allapot', 'skala_allapot', 'spektrum_tengely',
])
META = ['rekord_id', 'felosztasi_csoport', 'gepegyseg_utvonal', 'meresi_nap']


def _szoveg(x) -> str:
    return '' if pd.isna(x) else str(x).strip()


def _hash(ut: Path) -> str:
    return hashlib.sha256(ut.read_bytes()).hexdigest()


def _kulso_forras(mappa: Path, jelent: dict, kulcs: str, alapnev: str) -> Path:
    ertek = jelent.get(kulcs)
    if ertek is None:
        return mappa / alapnev
    if not isinstance(ertek, str) or not ertek.strip():
        raise ValueError(f'Hibás külső forráshivatkozás: {kulcs}')
    ut = Path(ertek)
    ut = ut if ut.is_absolute() else mappa / ut
    if not ut.is_file():
        raise ValueError(f'Nem található a korábban megadott külső forrás: {ut}')
    return ut


def forras_betolt(ut: Path, egyseg: str, detektalas: str) -> tuple[pd.DataFrame, dict]:
    """Csak olvas: a felhasználó címkéit és korábbi eredményeit nem módosítja."""
    with ut.open(encoding='utf-8-sig', newline='') as f:
        fejlec = next(csv.reader(f, delimiter=';'), [])
    if len(fejlec) != len(set(fejlec)):
        raise ValueError('Ismétlődő oszlopnév a jellemzőtáblában.')
    tabla = pd.read_csv(ut, sep=';', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    if KOTELEZO - set(tabla):
        raise ValueError(f'Hiányos jellemzőtábla: {sorted(KOTELEZO - set(tabla))}. Futtasd újra a beolvasót.')
    if any(k.startswith(('cimke_', 'ell_')) or k in {'cel_hibas', 'cel_hibatipus', 'tanitasi_jelolt'}
           for k in tabla):
        raise ValueError('Nyers jellemzok.csv szükséges, nem egy korábbi címkézett eredmény.')
    if tabla.empty:
        raise ValueError('Üres jellemzőtábla.')
    for k in KOTELEZO - set(OSSZEVETENDO):
        tabla[k] = tabla[k].str.strip()
    if tabla.rekord_id.eq('').any():
        raise ValueError('Hiányzó rekordazonosító.')
    for k in OSSZEVETENDO:
        szoveg = tabla[k].str.strip()
        szam = pd.to_numeric(szoveg.str.replace(',', '.', regex=False), errors='coerce')
        if (szoveg.ne('') & (szam.isna() | ~np.isfinite(szam))).any():
            raise ValueError(f'Hibás vagy nem véges szám a(z) {k} oszlopban.')
        tabla[k] = szam
    # A mérés azonosítója alapján a korábbi kézi címke átvehető.
    # A régi Speed-alapú jellemzőket külön kezelem.
    if 'fordulat_modszer' not in tabla:
        tabla['fordulat_modszer'] = 'korabbi_vagy_ismeretlen'
    # A v7 tábláinál az eredeti feldolgozási beállítást használom.
    if 'harmonikus_tures_rel' not in tabla:
        beallitas_ut = ut.parent / 'abra_beallitasok.json'
        beallitas = json.loads(beallitas_ut.read_text(encoding='utf-8')) if beallitas_ut.is_file() else {}
        if not isinstance(beallitas, dict) or not isinstance(beallitas.get('jellemzok', {}), dict):
            raise ValueError('Hibás abra_beallitasok.json.')
        tabla['harmonikus_tures_rel'] = beallitas.get('jellemzok', {}).get('harmonikus_tures_rel', np.nan)
    tabla['harmonikus_tures_rel'] = pd.to_numeric(
        tabla.harmonikus_tures_rel.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    napok, idok = [], []
    for _, m in tabla.iterrows():
        ido = datetime.fromisoformat(m.idopont)
        if ido.tzinfo is not None or m.idopont_pontossag not in {'perc', 'masodperc'}:
            raise ValueError('Nem támogatott időpont vagy időpontpontosság.')
        if not m.meresi_pont or not m.csatorna:
            raise ValueError('Hiányzó mérési pont vagy csatorna.')
        napok.append(ido.date().isoformat())
        idok.append(ido.isoformat(sep=' '))
    tabla['idopont'], tabla['meresi_nap'] = idok, napok
    # Egy gép azonos napi mérései az exportfájltól függetlenül egy csoportba kerülnek.
    tabla['felosztasi_csoport'] = [hashlib.sha256(json.dumps([m, n]).encode()).hexdigest()
                                   for m, n in zip(tabla.gepegyseg_utvonal, napok)]
    jelent_ut = ut.parent / 'cimkezes_jelentes.json'
    jelent = json.loads(jelent_ut.read_text(encoding='utf-8')) if jelent_ut.is_file() else {}
    if not isinstance(jelent, dict):
        raise ValueError('Hibás cimkezes_jelentes.json.')
    c_ut = _kulso_forras(ut.parent, jelent, 'kulso_cimkefajl', 'cimkek.csv')
    e_ut = _kulso_forras(ut.parent, jelent, 'kulso_skf_ellenorzes_fajl', 'skf_ellenorzes.csv')
    ismert = set(tabla.rekord_id)
    c = (szerkesztett_csv(c_ut, CIMKE_MEZOK, ismert) if c_ut.is_file()
         else pd.DataFrame(columns=['rekord_id'] + CIMKE_MEZOK))
    e = szerkesztett_csv(e_ut, ELL_MEZOK, ismert) if e_ut.is_file() else None
    # Az ismétlődéseket az összes forrás alapján ellenőrzöm.
    eredmeny = osszekapcsolas(tabla, c, e, egyseg=egyseg, detektalas=detektalas,
                            ismetlesek_kizarasa=False)
    hibas = (eredmeny[JELLEMZOK].lt(0).any(axis=1) | eredmeny[JELLEMZOK].isna().all(axis=1))
    eredmeny.loc[hibas, 'adatminoseg_statusz'] = 'ellenorizendo'
    eredmeny.loc[hibas, 'adatminoseg_okai'] += '; hibas_vagy_hianyzo_modelljellemzok'
    eredmeny['forras_jellemzotabla'] = str(ut.resolve())
    forras = {
        'jellemzotabla': str(ut.resolve()), 'sha256': _hash(ut), 'rekordok': len(eredmeny),
        'cimkefajl': str(c_ut.resolve()) if c_ut.is_file() else None,
        'cimkefajl_sha256': _hash(c_ut) if c_ut.is_file() else None,
        'skf_ellenorzes': str(e_ut.resolve()) if e_ut.is_file() else None,
        'skf_ellenorzes_sha256': _hash(e_ut) if e_ut.is_file() else None,
    }
    return eredmeny, forras


def _egy_rekord(resz: pd.DataFrame, egyseg: str, detektalas: str) -> dict:
    jo_adat = resz[(resz.adatminoseg_statusz == 'elokesziteshez_megfelelo') &
                   (resz.mertekegyseg == egyseg) & (resz.detektalas == detektalas)]
    alap = (jo_adat if not jo_adat.empty else resz).iloc[0].to_dict()
    okok = []
    cimkek = resz[resz.cimke_statusz == 'igazolt_rekordcimke']
    allapotok = set(cimkek.cel_hibas.dropna())
    alap['cel_hibas'], alap['cel_hibatipus'] = np.nan, ''
    alap['cimke_forras_jellemzotabla'] = ''
    hibatipusok = sorted({_szoveg(t) for t in cimkek.cel_hibatipus if _szoveg(t)})
    alap['hibatipus_utkozes'] = len(hibatipusok) > 1
    alap['hibatipusok_forrasokban'] = json.dumps(hibatipusok, ensure_ascii=False)
    alap['allapotok_forrasokban'] = json.dumps(sorted(int(k) for k in allapotok))
    if any(resz[k].nunique(dropna=False) > 1 for k in [
        'meresi_pont', 'idopont', 'idopont_pontossag', 'csatorna', 'mertekegyseg', 'detektalas']):
        okok.append('utkozo_rekordazonosito')
    if len(allapotok) != 1:
        okok.append('utkozo_igazolt_allapotcimkek' if allapotok else 'nincs_igazolt_allapotcimke')
        alap['cimke_statusz'] = okok[-1]
        alap['cimke_allapot_ertekelt'] = 'ismeretlen'
        for k in CIMKE_MEZOK:
            alap['cimke_' + k] = ''
    else:
        # Az eredeti spektrum azonosítója skálaváltáskor is megőrzi a címke kapcsolatát.
        ismert_tipus = cimkek[cimkek.cel_hibatipus.fillna('').ne('')]
        c = (ismert_tipus if len(hibatipusok) == 1 else cimkek).iloc[0]
        for k in CIMKE_MEZOK:
            alap['cimke_' + k] = c['cimke_' + k]
        alap['cimke_statusz'] = 'igazolt_rekordcimke'
        alap['cimke_allapot_ertekelt'] = c.cimke_allapot_ertekelt
        alap['cel_hibas'] = int(next(iter(allapotok)))
        alap['cel_hibatipus'] = hibatipusok[0] if len(hibatipusok) == 1 else ''
        alap['cimke_forras_jellemzotabla'] = c.forras_jellemzotabla
        if len(hibatipusok) > 1:
            alap['cimke_hibatipus'], alap['cimke_hibatipus_igazolt'] = '', 'nem'
    if jo_adat.empty:
        okok.append('nincs_hasznalhato_meresi_valtozat')
        okok.extend(k for k in _szoveg(alap['adatminoseg_okai']).split('; ') if k)
        if alap['mertekegyseg'] != egyseg or alap['detektalas'] != detektalas:
            okok.append('mas_meresfajta')
    else:
        ertekek = jo_adat[OSSZEVETENDO].to_numpy(dtype=float)
        if not np.isclose(ertekek, ertekek[0], rtol=1e-8, atol=1e-12, equal_nan=True).all():
            okok.append('utkozo_ervenyes_feldolgozasok')
        if jo_adat.gepegyseg_utvonal.nunique() != 1:
            okok.append('utkozo_gepazonositas')
        if not jo_adat.fordulat_modszer.isin(AZONOS_MODSZEREK).all():
            okok.append('utkozo_vagy_elavult_fordulatmodszer')
    megkezdett = set(resz.skf_ellenorzes_statusz) - {
        'nem_megadott_opcionalis', 'egyezik_a_megadott_turesen_belul'}
    if megkezdett:
        okok.append('tisztazatlan_kezzel_megadott_skf_ellenorzes')
    alap['forras_valtozatok_szama'] = len(resz)
    alap['forras_jellemzotablak'] = json.dumps(sorted(set(resz.forras_jellemzotabla)), ensure_ascii=False)
    alap['adatkeszlet_kizaras_okai'] = '; '.join(dict.fromkeys(okok))
    alap['tanitasi_jelolt'] = not okok
    alap['kizaras_okai'] = alap['adatkeszlet_kizaras_okai']
    return alap


def _atfedesek(tabla: pd.DataFrame) -> None:
    """Az eltérő azonosítójú, de nem elkülöníthető mérési eseteket kizárja."""
    aktiv = tabla[tabla.tanitasi_jelolt].copy()
    if aktiv.empty:
        return
    aktiv['_perc'] = aktiv.idopont.map(lambda x: datetime.fromisoformat(x).replace(second=0, microsecond=0))
    kulcs = ['meresi_pont', 'csatorna', 'mertekegyseg', 'detektalas']
    jelzesek = []
    for _, resz in aktiv.groupby(kulcs + ['idopont'], sort=False):
        if len(resz) > 1:
            jelzesek.append((resz.index, 'azonos_meresi_kulcs_eltero_tartalommal'))
    for _, resz in aktiv.groupby(kulcs + ['_perc'], sort=False):
        if resz.idopont_pontossag.nunique() > 1:
            jelzesek.append((resz.index, 'eltero_idopontpontossagu_atfedes'))
    for indexek, ok in jelzesek:
        for i in indexek:
            okok = [k for k in tabla.at[i, 'adatkeszlet_kizaras_okai'].split('; ') if k]
            tabla.at[i, 'adatkeszlet_kizaras_okai'] = '; '.join(dict.fromkeys(okok + [ok]))
            tabla.at[i, 'kizaras_okai'] = tabla.at[i, 'adatkeszlet_kizaras_okai']
            tabla.at[i, 'tanitasi_jelolt'] = False


def osszeallitas(fajlok: list[Path], egyseg: str = 'mm/s', detektalas: str = 'RMS'
                ) -> tuple[pd.DataFrame, dict]:
    fajlok = sorted({ut.resolve() for ut in fajlok}, key=str)
    if not fajlok:
        raise ValueError('Nincs jellemzok.csv. Először futtasd a csv_beolvasas.py programot.')
    reszek, forrasok, hibak = [], [], []
    for ut in fajlok:
        try:
            tabla, forras = forras_betolt(ut, egyseg, detektalas)
            reszek.append(tabla)
            forrasok.append(forras)
        except (ValueError, OSError, UnicodeError, csv.Error, pd.errors.ParserError) as exc:
            hibak.append({'jellemzotabla': str(ut), 'hiba': str(exc)})
    if reszek:
        minden = pd.concat(reszek, ignore_index=True)
        egyedi = pd.DataFrame([_egy_rekord(resz, egyseg, detektalas)
                              for _, resz in minden.groupby('rekord_id', sort=True)])
        _atfedesek(egyedi)
    else:
        minden = pd.DataFrame()
        egyedi = pd.DataFrame(columns=META + JELLEMZOK + ['fordulat_modszer',
            'cel_hibas', 'cel_hibatipus', 'tanitasi_jelolt', 'adatkeszlet_kizaras_okai',
            'hibatipus_utkozes'])
    jeloltek = egyedi[egyedi.tanitasi_jelolt.astype(bool)]
    osztalyok = {str(k): int(jeloltek.cel_hibas.eq(k).sum()) for k in (0, 1)}
    csoportok = {str(k): int(jeloltek.loc[jeloltek.cel_hibas.eq(k), 'felosztasi_csoport'].nunique())
                 for k in (0, 1)}
    gepek = {str(k): int(jeloltek.loc[jeloltek.cel_hibas.eq(k), 'gepegyseg_utvonal'].nunique())
             for k in (0, 1)}
    if hibak:
        allapot = 'forrashibak_ellenorizendok'
    elif not len(jeloltek):
        allapot = 'nincs_felhasznalhato_cimkezett_meres'
    elif not all(osztalyok.values()):
        allapot = 'csak_egy_allapotosztaly'
    elif min(csoportok.values()) < 2:
        allapot = 'keves_csoport_a_csoportos_ertekeleshez'
    else:
        allapot = 'adatkeszlet_osszeallitva'
    okok = egyedi.adatkeszlet_kizaras_okai.str.split('; ').explode()
    jelent = {
        'valtozat': 11, 'fordulat_modszer': MODSZER, 'keszult_utc': datetime.now(timezone.utc).isoformat(),
        'allapot': allapot, 'kivalasztott_egyseg': egyseg, 'kivalasztott_detektalas': detektalas,
        'forrasok': forrasok, 'forrashibak': hibak, 'bemeneti_valtozatok': len(minden),
        'egyedi_rekordok': len(egyedi), 'ismetelt_valtozatok': len(minden) - len(egyedi),
        'tanitasi_rekordok': len(jeloltek), 'kizart_rekordok': len(egyedi) - len(jeloltek),
        'osztalyok': osztalyok, 'csoportok_osztalyonkent': csoportok, 'gepek_osztalyonkent': gepek,
        'felosztasi_csoportok': int(jeloltek.felosztasi_csoport.nunique()),
        'kizarasi_okok': okok[okok.notna() & okok.ne('')].value_counts().to_dict(),
        'hibatipus_utkozesek': int(egyedi.hibatipus_utkozes.astype(bool).sum()),
        'jellemzo_oszlopok': JELLEMZOK,
        'hianyzo_jellemzok': {k: int(jeloltek[k].isna().sum()) for k in JELLEMZOK},
        'rekord_sorrend_sha256': hashlib.sha256('\n'.join(jeloltek.rekord_id).encode()).hexdigest(),
        'megjegyzes': 'Nincs tanítás vagy tesztfelosztás. A csoportszám nem igazolja a mérési '
                      'esetek függetlenségét vagy az adatmennyiség elegendőségét. '
                      'A hiányzó jellemzők pótlása csak a későbbi tanítóadatokon illeszthető.',
    }
    return egyedi, jelent


def mentes(tabla: pd.DataFrame, jelent: dict, mappa: Path) -> None:
    mappa.mkdir(parents=True, exist_ok=True)
    jeloltek = tabla[tabla.tanitasi_jelolt.astype(bool)].copy()
    gepek = sorted({g for g in tabla.gepegyseg_utvonal.dropna() if str(g).strip()})
    pd.DataFrame({'gepegyseg_utvonal': gepek, 'szerep': ['kihagy'] * len(gepek)}).to_csv(
        mappa / 'gepterv_sablon.csv', sep=';', encoding='utf-8-sig', index=False)
    for fajl, t in [
        ('adatkeszlet_attekintes.csv', tabla), ('tanito_adatok.csv', jeloltek),
        ('X.csv', jeloltek[JELLEMZOK]), ('y.csv', jeloltek[['cel_hibas']]),
        ('csoportok.csv', jeloltek[META]),
    ]:
        t.to_csv(mappa / fajl, sep=';', decimal=',', encoding='utf-8-sig', index=False, na_rep='')
    (mappa / 'adatkeszlet_jelentes.json').write_text(
        json.dumps(jelent, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (mappa / 'jellemzo_oszlopok.json').write_text(
        json.dumps(JELLEMZOK, ensure_ascii=False, indent=2), encoding='utf-8')
    osszegzes = [
        'TANÍTÓADATOK ÖSSZEÁLLÍTÁSA – 11. VÁLTOZAT', f'Állapot: {jelent["allapot"]}',
        f'Beolvasott változatok: {jelent["bemeneti_valtozatok"]}',
        f'Egyedi rekordok: {jelent["egyedi_rekordok"]}',
        f'Tanításra kiválasztott rekordok: {jelent["tanitasi_rekordok"]}',
        f'Igazolt megfelelő: {jelent["osztalyok"]["0"]}; igazolt hibás: {jelent["osztalyok"]["1"]}',
        f'Gép+napi csoportok: {jelent["felosztasi_csoportok"]}',
        f'Forráshibák: {len(jelent["forrashibak"])}',
        'A részletes kizárási okok az adatkeszlet_attekintes.csv táblában láthatók.',
        'Az X.csv csak a hét leíró jellemzőt tartalmazza. A hiányzó értékek üresek.',
        'Az X.csv, y.csv és csoportok.csv sorai azonos sorrendben tartoznak össze.',
        'Csak az 5–100 Hz sávmaximumos eljárással újraszámolt jellemzők használhatók.',
        'Ez előkészített adatkészlet; betanított modell és pontosságmérés még nincs.',
    ]
    (mappa / 'OSSZEGZES.txt').write_text('\n'.join(osszegzes) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jellemzotablak', nargs='*', type=Path,
                        help='Konkrét jellemzok.csv fájlok; nélkülük automatikusan keres az outputs alatt.')
    parser.add_argument('--bemenet', type=Path, default=PROJEKT / 'outputs')
    parser.add_argument('--kimenet', type=Path, default=PROJEKT / 'outputs' / 'tanito_adatok')
    parser.add_argument('--egyseg', default='mm/s')
    parser.add_argument('--detektalas', default='RMS')
    args = parser.parse_args(argv)
    try:
        fajlok = args.jellemzotablak or list(args.bemenet.rglob('jellemzok.csv'))
        tabla, jelent = osszeallitas(fajlok, args.egyseg, args.detektalas)
        mentes(tabla, jelent, args.kimenet)
    except (ValueError, OSError, UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        parser.exit(1, f'Adatkészlet-összeállítási hiba: {exc}\n')
    print((args.kimenet / 'OSSZEGZES.txt').read_text(encoding='utf-8'))
    print(f'Eredmények: {args.kimenet.resolve()}')
    for hiba in jelent['forrashibak']:
        print(f'Forráshiba: {hiba["jellemzotabla"]}: {hiba["hiba"]}')
    return 2 if jelent['forrashibak'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
