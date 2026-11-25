# A spektrumból kiszámítja a csúcsok és a forgási harmonikusok jellemzőit.

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from fordulatszam import AZONOS_MODSZEREK, becsles as fordulat_becsles


@dataclass(frozen=True)
class Parameterek:
    csucsok_szama: int = 5
    prominencia_rel: float = 0.05
    harmonikus_tures_rel: float = 0.02

    def ellenorzes(self) -> None:
        if isinstance(self.csucsok_szama, bool) or not isinstance(self.csucsok_szama, int) or not 1 <= self.csucsok_szama <= 50:
            raise ValueError("A kiemelt csúcsok száma 1 és 50 közötti egész legyen.")
        if not math.isfinite(self.prominencia_rel) or not 0 <= self.prominencia_rel <= 1:
            raise ValueError("A relatív prominencia 0 és 1 közé essen.")
        if not math.isfinite(self.harmonikus_tures_rel) or not 0 <= self.harmonikus_tures_rel < 0.25:
            raise ValueError("A harmonikus relatív tűrése 0 és 0,25 közé essen, 0,25 nélkül.")


def hz_tengely(x: np.ndarray, egyseg: str, speed: float) -> np.ndarray | None:
    if egyseg == "Hz":
        return x.copy()
    if egyseg == "CPM":
        return x / 60.0
    if egyseg == "order" and math.isfinite(speed) and speed > 0:
        return x * speed
    return None


def meres_jelleg(egyseg: str) -> str:
    egyseg = egyseg.strip().replace('µ', 'u').replace('μ', 'u').replace(' ', '').lower()
    if egyseg in {'mm/s', 'mm/sec', 'm/s', 'm/sec', 'ips', 'in/s', 'in/sec'}:
        return 'sebessegspektrum'
    if egyseg in {'g', 'm/s2', 'm/s²', 'm/s^2', 'mm/s2', 'mm/s²', 'mm/s^2'}:
        return 'gyorsulasspektrum'
    if egyseg in {'ge'}:
        return 'burkologorbe_spektrum'
    if egyseg in {'um', 'micron', 'microns', 'mil', 'mils', 'mm'}:
        return 'elmozdulasspektrum'
    return 'ismeretlen_mertekegyseg'


def _uresek(p: Parameterek) -> dict:
    d = {
        "jellemzok_allapot": "meg_nem_szamolt", "frekvencia_jellemzok_allapot": "nincs_hz_skala",
        "nulla_spektrum": False, "spektrum_max_amplitudo": np.nan,
        "spektrum_max_index": np.nan, "spektrum_max_frekvencia_hz": np.nan,
        "dominans_pozitiv_amplitudo": np.nan, "dominans_pozitiv_frekvencia_hz": np.nan,
        "lokalis_csucsok_szama": np.nan, "prominencia_kuszob_amplitudo": np.nan,
        "harm_2_1_amplitudo_arany": np.nan, "harm_3_1_amplitudo_arany": np.nan,
        "harm_aranyok_allapot": "nem_szamithato",
    }
    for k in range(1, p.csucsok_szama + 1):
        for mezo in ("index", "amplitudo", "frekvencia_hz", "prominencia"):
            d[f"csucs_{k}_{mezo}"] = np.nan
    for k in (1, 2, 3):
        for mezo in ("cel_hz", "tures_hz", "amplitudo", "frekvencia_hz", "index"):
            d[f"harm_{k}_{mezo}"] = np.nan
        d[f"harm_{k}_allapot"] = "nem_szamithato"
    return d


def egy_spektrum(amplitudo: np.ndarray, x: np.ndarray, egyseg: str, speed: float,
                 p: Parameterek | None = None) -> tuple[dict, list[dict]]:
    """A harmonikusoknál a megadott sáv maximumát keressük, nem hibát állapítunk meg.

    A speed itt már a kiválasztott referencia Hz-ben, nem az export Speed mezője.
    A teljes munkamenetben a jellemzotabla adja át a vállalati eljárással becsült forgási frekvenciát.

    Tűrés = max(pontköz, relatív tűrés * célfrekvencia). A sávnak teljesen bele
    kell férnie a rendelkezésre álló tartományba. Durva rácsnál nem találgatunk.
    """
    p = p or Parameterek()
    p.ellenorzes()
    y = np.asarray(amplitudo, dtype=float)
    x = np.asarray(x, dtype=float)
    d = _uresek(p)
    if y.ndim != 1 or x.ndim != 1 or len(y) < 2 or len(y) != len(x):
        d['jellemzok_allapot'] = 'hibas_tombalak'
        return d, []
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        d['jellemzok_allapot'] = 'nem_veges_adat'
        return d, []
    lepesek = np.diff(x)
    if not (lepesek > 0).all() or not np.allclose(lepesek, lepesek[0], rtol=1e-7, atol=abs(lepesek[0]) * 1e-7):
        d['jellemzok_allapot'] = 'nem_egyenletes_novekvo_tengely'
        return d, []
    if np.any(y < 0):
        d['jellemzok_allapot'] = 'negativ_ertek_nem_amplitudospektrum'
        return d, []

    d['nulla_spektrum'] = bool(not np.any(y > 0))
    d['jellemzok_allapot'] = 'nulla_spektrum_ellenorizendo' if d['nulla_spektrum'] else 'leiro_jellemzok_kiszamitva'
    maxi = int(np.argmax(y))
    d['spektrum_max_amplitudo'], d['spektrum_max_index'] = float(y[maxi]), maxi
    f = hz_tengely(x, egyseg, speed)
    if f is not None:
        if f[0] < 0:
            d['jellemzok_allapot'] = 'negativ_frekvencia_nem_tamogatott'
            return d, []
        d['frekvencia_jellemzok_allapot'] = 'megadott_skala_alapjan'
        d['spektrum_max_frekvencia_hz'] = float(f[maxi])
        pozitiv = np.flatnonzero(f > 0)
        if len(pozitiv) and not d['nulla_spektrum']:
            index = int(pozitiv[np.argmax(y[pozitiv])])
            if y[index] > 0:
                d['dominans_pozitiv_amplitudo'] = float(y[index])
                d['dominans_pozitiv_frekvencia_hz'] = float(f[index])

    # A végpont csak a teljes spektrum maximumaként vehető figyelembe.
    kuszob = p.prominencia_rel * float(y.max())
    d['prominencia_kuszob_amplitudo'] = kuszob
    csucsok, tul = find_peaks(y, prominence=kuszob)
    if f is not None:
        maszk = f[csucsok] > 0
        csucsok = csucsok[maszk]
        prominenciak = tul['prominences'][maszk]
    else:
        prominenciak = tul['prominences']
    d['lokalis_csucsok_szama'] = len(csucsok)
    sorrend = sorted(range(len(csucsok)), key=lambda i: (-y[csucsok[i]], csucsok[i]))[:p.csucsok_szama]
    lista = []
    for rang, poz in enumerate(sorrend, 1):
        index = int(csucsok[poz])
        rekord = {'rang': rang, 'index': index, 'amplitudo': float(y[index]),
                  'frekvencia_hz': float(f[index]) if f is not None else np.nan,
                  'prominencia': float(prominenciak[poz])}
        lista.append(rekord)
        for mezo in ('index', 'amplitudo', 'frekvencia_hz', 'prominencia'):
            d[f'csucs_{rang}_{mezo}'] = rekord[mezo]

    for k in (1, 2, 3):
        cel = k * speed if math.isfinite(speed) and speed > 0 else np.nan
        d[f'harm_{k}_cel_hz'] = cel
        if f is None:
            ok = 'nincs_hz_skala'
        elif not math.isfinite(cel):
            ok = 'nincs_pozitiv_forgasi_frekvencia'
        else:
            df = float((f[-1] - f[0]) / (len(f) - 1))
            tures = max(df, p.harmonikus_tures_rel * cel)
            d[f'harm_{k}_tures_hz'] = tures
            eps = max(1.0, abs(float(f[-1]))) * 1e-12
            if df >= speed / 2 or tures >= speed / 2:
                ok = 'durva_pontkoz_vagy_atfedo_sav'
            elif cel < f[0] - eps or cel > f[-1] + eps:
                ok = 'cel_tartomanyon_kivul'
            elif cel - tures < f[0] - eps or cel + tures > f[-1] + eps:
                ok = 'reszleges_sav_nem_hasznaljuk'
            else:
                jeloltek = np.flatnonzero(np.abs(f - cel) <= tures + eps)
                if not len(jeloltek):
                    ok = 'nincs_pont_a_savban'
                else:
                    index = int(jeloltek[np.argmax(y[jeloltek])])
                    d[f'harm_{k}_amplitudo'] = float(y[index])
                    d[f'harm_{k}_frekvencia_hz'] = float(f[index])
                    d[f'harm_{k}_index'] = index
                    ok = 'savmaximum_kiszamitva'
        d[f'harm_{k}_allapot'] = ok

    a1 = d['harm_1_amplitudo']
    if math.isfinite(a1) and a1 > 0:
        for k in (2, 3):
            ak = d[f'harm_{k}_amplitudo']
            if math.isfinite(ak):
                arany = ak / a1
                if math.isfinite(arany):
                    d[f'harm_{k}_1_amplitudo_arany'] = arany
        aranyok = [d['harm_2_1_amplitudo_arany'], d['harm_3_1_amplitudo_arany']]
        d['harm_aranyok_allapot'] = ('kiszamitva' if all(math.isfinite(a) for a in aranyok)
                                      else 'reszben_vagy_nem_szamithato')
    elif math.isfinite(a1):
        d['harm_aranyok_allapot'] = 'nulla_1x_amplitudo'
    return d, lista


def jellemzotabla(meta: pd.DataFrame, spektrumok: list[np.ndarray],
                  tengelyek: list[np.ndarray], p: Parameterek | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    p = p or Parameterek()
    p.ellenorzes()
    if not len(meta) == len(spektrumok) == len(tengelyek):
        raise ValueError('A metaadatok, spektrumok és tengelyek száma eltér.')
    rekordok, csucslista, rajzok = [], [], []
    for i, (_, meres) in enumerate(meta.iterrows()):
        referencia = np.nan
        if (meres.get('fordulat_modszer') in AZONOS_MODSZEREK and
                meres.get('skala_export_egyseg') in {'Hz', 'CPM'}):
            referencia = float(meres.get('forgasi_frekvencia_hz') or np.nan)
        f = hz_tengely(np.asarray(tengelyek[i], dtype=float), meres['spektrum_tengely'], referencia)
        fordulat = fordulat_becsles(spektrumok[i], f)
        referencia = fordulat['forgasi_frekvencia_hz']
        jelleg = meres_jelleg(str(meres['mertekegyseg']))
        detektalas = str(meres['detektalas']).strip().lower().replace('-', ' ')
        if jelleg == 'ismeretlen_mertekegyseg' or detektalas not in {'rms', 'peak', 'peak to peak', 'pk', 'pk pk'}:
            d, lista = _uresek(p), []
            d['jellemzok_allapot'] = ('nem_tamogatott_mertekegyseg' if jelleg == 'ismeretlen_mertekegyseg'
                                      else 'nem_tamogatott_detektalas')
        else:
            d, lista = egy_spektrum(spektrumok[i], tengelyek[i], meres['spektrum_tengely'],
                                   referencia if referencia is not None else np.nan, p)
        d.update(fordulat)
        d['meres_jelleg'] = jelleg
        rekordok.append(d)
        for csucs in lista:
            csucslista.append({'meres_sorszam': meres['meres_sorszam'],
                               'meresi_pont': meres['meresi_pont'], 'idopont': meres['idopont'],
                               'mertekegyseg': meres['mertekegyseg'], 'detektalas': meres['detektalas'],
                               'skala_allapot': meres['skala_allapot'], **csucs})
        rajzok.append({'jellemzok': d, 'csucsok': lista})
    extra = pd.DataFrame(rekordok)
    tabla = pd.concat([meta.drop(columns=extra.columns, errors='ignore').reset_index(drop=True), extra], axis=1)
    tabla['harmonikus_tures_rel'] = p.harmonikus_tures_rel
    # A csoportkulcs mellett az üzemi feltételeket is ellenőrizni kell.
    tabla['meresfajta_kulcs'] = tabla['mertekegyseg'].astype(str) + ' | ' + tabla['detektalas'].astype(str)
    oszlopok = ['meres_sorszam', 'meresi_pont', 'idopont', 'mertekegyseg', 'detektalas',
                'skala_allapot', 'rang', 'index', 'amplitudo', 'frekvencia_hz', 'prominencia']
    return tabla, pd.DataFrame(csucslista, columns=oszlopok), rajzok
