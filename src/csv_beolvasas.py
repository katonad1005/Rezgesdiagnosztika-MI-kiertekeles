# Beolvassa az SKF-spektrumexportot, előkészíti a frekvenciatengelyt és elindítja a jellemzők számítását.

r"""SKF Analysis spektrumexport beolvasása és ellenőrzése.

Futtatás a projekt gyökeréből: python .\src\csv_beolvasas.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fordulatszam import MODSZER, becsles as fordulat_becsles


PROJEKT = Path(__file__).resolve().parent.parent
FEJLEC = [
    "Point Path", "DTS", "Unit", "Detection", "Channel", "Lines",
    "End Frequency", "Speed (Hz)", "Process Value", "Unit", "BOV / GAP",
    "DIGITAL", "Data",
]


def szam(szoveg: str, mezo: str, sor: int) -> float:
    try:
        ertek = float(szoveg.strip().replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"CSV {sor}. sor: hibás szám a(z) {mezo} mezőben: {szoveg!r}") from exc
    if not math.isfinite(ertek):
        raise ValueError(f"CSV {sor}. sor: nem véges szám a(z) {mezo} mezőben.")
    return ertek


def hierarchia(utvonal: str) -> dict[str, str | None]:
    """A mintán ellenőrzött névkonvenció; más struktúrát nem találunk ki."""
    reszek = [resz for resz in utvonal.split("\\") if resz]
    alap = {
        "oldal": "nincs_megadva",
        "helycsoport": reszek[-2] if len(reszek) >= 2 else None,
        "csoport_utvonal": "\\" + "\\".join(reszek[:-1]),
    }
    for idx, resz in enumerate(reszek[:-1]):
        klima = re.fullmatch(r"(.+)_(BEFUVO|ELSZIVO)", resz, flags=re.IGNORECASE)
        if klima:
            kulcs = "\\" + "\\".join(reszek[:idx] + [klima[1]])
            gepresz = re.search(r"(?:_|\b)(MOTOR|VENTI)(?:-|_|$)", reszek[-1], re.IGNORECASE)
            return {
                **alap,
                "gepegyseg_utvonal": kulcs,
                "csoport_utvonal": kulcs,
                "gepegyseg": klima[1],
                "gepresz": gepresz[1].upper() if gepresz else None,
                "oldal": "befuvo" if klima[2].upper() == "BEFUVO" else "elszivo",
                "hierarchia_allapot": "legkezelo_nevkonvencio_alapjan",
            }
    if len(reszek) >= 3 and reszek[-2].endswith(("_MOTOR", "_TENGELY")):
        return {
            **alap,
            "csoport_utvonal": "\\" + "\\".join(reszek[:-2]),
            "gepegyseg_utvonal": "\\" + "\\".join(reszek[:-2]),
            "gepegyseg": reszek[-3],
            "gepresz": reszek[-2],
            "hierarchia_allapot": "mintabeli_nevkonvencio_alapjan",
        }
    return {
        **alap,
        "gepegyseg_utvonal": None,
        "gepegyseg": None,
        "gepresz": None,
        "hierarchia_allapot": "nem_felismert_ellenorizendo",
    }


def napi_osszesites(meta: pd.DataFrame) -> pd.DataFrame:
    """Csak áttekintés: minden spektrum és időbélyeg külön megmarad."""
    return (
        meta.groupby(["csoport_utvonal", "gepegyseg", "oldal", "meresi_nap"],
                     dropna=False, sort=True)
        .agg(meresi_rekordok=("meres_sorszam", "size"),
             meresi_utvonalak=("meresi_pont", "nunique"),
             elso_meres=("idopont", "min"),
             utolso_meres=("idopont", "max"))
        .reset_index()
    )


def datum_beolvasas(szoveg: str, sor: int) -> tuple[datetime, str]:
    tiszta = re.sub(r"\s*\.\s*", ".", szoveg.strip())
    for forma in ("%Y.%m.%d.%H:%M:%S", "%Y.%m.%d.%H:%M",
                  "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M",
                  "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(tiszta, forma), ("masodperc" if "%S" in forma else "perc")
        except ValueError:
            pass
    raise ValueError(f"CSV {sor}. sor: nem támogatott dátumformátum: {szoveg!r}.")


def csv_szoveg(fajl: Path, kodolas: str | None) -> str:
    bajtok = fajl.read_bytes()
    if kodolas:
        return bajtok.decode(kodolas)
    if bajtok.startswith((b"\xff\xfe", b"\xfe\xff")):
        return bajtok.decode("utf-16")
    try:
        return bajtok.decode("utf-8-sig")
    except UnicodeError:
        return bajtok.decode("cp1250")


def fajlvalasztas(mappa: Path) -> Path:
    fajlok = sorted((p for p in mappa.iterdir() if p.is_file() and p.suffix.lower() == ".csv"),
                    key=lambda p: p.name.casefold()) if mappa.is_dir() else []
    if not fajlok:
        raise ValueError(f"Nincs CSV ebben a mappában: {mappa}. Másold ide az SKF-exportot.")
    print("Választható CSV-fájlok:")
    for idx, fajl in enumerate(fajlok, 1):
        print(f"  {idx}. {fajl.name}")
    if len(fajlok) == 1:
        print("Egy fájl található, ezt olvasom be.")
        return fajlok[0]
    while True:
        try:
            valasz = input("A beolvasandó fájl sorszáma (q = kilépés): ").strip()
        except EOFError as exc:
            raise ValueError("Nincs interaktív bevitel. Add meg a CSV útvonalát a parancsban.") from exc
        if valasz.lower() == "q":
            raise SystemExit(0)
        if valasz.isdecimal() and 1 <= int(valasz) <= len(fajlok):
            return fajlok[int(valasz) - 1]
        print(f"1 és {len(fajlok)} közötti sorszámot adj meg.")


def gepterkep_alkalmazas(meta: pd.DataFrame, fajl: Path) -> pd.DataFrame:
    """Új hierarchiák géphatárai kódmódosítás nélkül megadhatók."""
    szabalyok = json.loads(fajl.read_text(encoding="utf-8-sig"))
    if not isinstance(szabalyok, list):
        raise ValueError("A géptérkép JSON-lista legyen.")
    latott = set()
    for szabaly in szabalyok:
        if not isinstance(szabaly, dict) or any(
            not isinstance(szabaly.get(k), str) or not szabaly[k].strip()
            for k in ("utvonal_prefix", "gepegyseg")
        ):
            raise ValueError("Minden géptérkép-szabályhoz utvonal_prefix és gepegyseg szöveg kell.")
        prefix = szabaly["utvonal_prefix"].rstrip("\\")
        if not prefix.startswith("\\") or prefix in latott:
            raise ValueError("A géptérkép prefixei teljes, egyedi SKF-útvonalak legyenek.")
        if "oldal" in szabaly and not isinstance(szabaly["oldal"], str):
            raise ValueError("A géptérkép oldal mezője szöveg legyen.")
        latott.add(prefix)
    
    # A hosszabb útvonal pontosabban azonosítja a gépet.
    for idx, meres in meta.iterrows():
        for szabaly in sorted(szabalyok, key=lambda x: len(x["utvonal_prefix"].rstrip("\\")), reverse=True):
            prefix = szabaly["utvonal_prefix"].rstrip("\\")
            if meres["meresi_pont"].startswith(prefix + "\\"):
                meta.loc[idx, ["gepegyseg_utvonal", "csoport_utvonal", "gepegyseg", "hierarchia_allapot"]] = [
                    prefix, prefix, szabaly["gepegyseg"], "kezzel_megadott_gepterkep"]
                if "oldal" in szabaly:
                    meta.loc[idx, "oldal"] = szabaly["oldal"]
                break
    return meta


def biztonsagos_nev(szoveg: str) -> str:
    return (re.sub(r'[^\w.-]+', '_', szoveg).strip('._') or 'csoport')[:70]


def skala_bekeres() -> tuple[str, float | None, str]:
    print("\nA CSV nem ad külön kezdőfrekvenciát vagy frekvenciaegységet.")
    print("Csak a POINT-beállításoknak megfelelő lehetőséget válaszd:")
    print("  1. 0-tól induló, Hz-ben megadott export")
    print("  2. 0-tól induló, CPM-ben megadott export")
    print("  3. Nem ismert – pontindex, frekvenciafeltételezés nélkül")
    print("  4. Más kezdőérték vagy fordulati rend alapú export")
    while True:
        try:
            valasz = input("Skála sorszáma: ").strip()
            if valasz in ("1", "2"):
                return ("Hz" if valasz == "1" else "CPM"), 0.0, "interaktivan_megadott"
            if valasz == "3":
                return "index", None, "nem_ismert"
            if valasz == "4":
                egyseg = input("Az export egysége (Hz / CPM / order): ").strip()
                if egyseg not in ("Hz", "CPM", "order"):
                    print("Ismeretlen egység.")
                    continue
                kezdo = float(input("Kezdőérték az export egységében: ").replace(",", "."))
                if math.isfinite(kezdo) and kezdo >= 0:
                    return egyseg, kezdo, "interaktivan_megadott"
            print("Érvényes beállítást adj meg.")
        except ValueError:
            print("A kezdőérték nem negatív, véges szám legyen.")
        except EOFError as exc:
            raise ValueError("Nincs interaktív bevitel. Add meg: --export-egyseg Hz --kezdo-frekvencia 0, vagy --export-egyseg index.") from exc


def skalak_alkalmazas(meta: pd.DataFrame, egyseg: str, kezdo: float | None,
                      tengely: str, forras: str, terkep: Path | None = None,
                      rekord_skalak: dict | None = None,
                      spektrumok: list[np.ndarray] | None = None
                      ) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """A spektrumpontok egyenletesek, az első és utolsó pont a megadott sáv két vége.
    A frekvenciaskálát a megadott mérési beállításokból állítom elő. A CSV Lines mezőjét tényleges pontszámként használom.
    """
    szabalyok = [] if terkep is None else json.loads(terkep.read_text(encoding="utf-8-sig"))
    if not isinstance(szabalyok, list):
        raise ValueError("A skálatérkép JSON-lista legyen.")
    prefixek = set()
    for s in szabalyok:
        if not isinstance(s, dict) or not isinstance(s.get("utvonal_prefix"), str):
            raise ValueError("Minden skálaszabályban utvonal_prefix szükséges.")
        prefix = s["utvonal_prefix"].rstrip("\\")
        if not prefix.startswith("\\") or prefix in prefixek:
            raise ValueError("A skálatérkép prefixei teljes, egyedi SKF-útvonalak legyenek.")
        prefixek.add(prefix)
        if s.get("egyseg") not in ("Hz", "CPM", "order", "index"):
            raise ValueError("A skálaszabály egyseg értéke Hz, CPM, order vagy index legyen.")
        if s["egyseg"] != "index":
            k = s.get("kezdo")
            if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 0:
                raise ValueError("A skálaszabály kezdo értéke nem negatív, véges szám legyen.")
        if "forras" in s and not isinstance(s["forras"], str):
            raise ValueError("A skálaszabály forras mezője szöveg legyen.")
    szabalyok.sort(key=lambda s: len(s["utvonal_prefix"].rstrip("\\")), reverse=True)
    if spektrumok is not None and len(spektrumok) != len(meta):
        raise ValueError('Eltérő számú metaadatsor és spektrum.')
    axes, kiegeszitesek = [], []
    for i, (_, m) in enumerate(meta.iterrows()):
        amplitudo = spektrumok[i] if spektrumok is not None else []
        fordulat = fordulat_becsles(amplitudo)
        e, k, f = egyseg, kezdo, forras
        vegpontok = "ket_vegpont"
        if rekord_skalak and int(m["csv_sor"]) in rekord_skalak:
            s = rekord_skalak[int(m["csv_sor"])]
            e, k, f = s["egyseg"], s["kezdo"], s["forras"]
            vegpontok = s.get("vegpontok", "ket_vegpont")
        for s in szabalyok:
            prefix = s["utvonal_prefix"].rstrip("\\")
            if m["meresi_pont"] == prefix or m["meresi_pont"].startswith(prefix + "\\"):
                e, k, f = s["egyseg"], s.get("kezdo"), s.get("forras", "felhasznaloi_skalaterkep")
                vegpontok = "ket_vegpont"
                break
        if f.startswith("automatikus_standardprofil") and re.search(r"(?:^|[\\/_ -])(zoom|order)(?:$|[\\/_ -])", str(m["meresi_pont"]), re.I):
            e, k, f = "index", None, "automatikus_profil_nem_alkalmazhato_zoom_order"
        n = int(m["spektrumhossz"])
        allapot = "felhasznalo_altal_megadott"
        if f.startswith("automatikus_standardprofil") or "ellenorizetlen" in f:
            allapot = "automatikus_vagy_ellenorizetlen_profil"
        hz_lepes = np.nan
        if e == "index":
            x = np.arange(n)
            cimke, cel, allapot = "Spektrumpont indexe (0-tól)", "index", "nem_ismert"
        else:
            if k is None or not math.isfinite(k) or k < 0 or m["vegfrekvencia_export"] <= k:
                raise ValueError(f"CSV {m['csv_sor']}. sor: a kezdőérték kisebb legyen a végértéknél, és nem negatív, véges szám legyen.")
            x = np.linspace(k, m["vegfrekvencia_export"], n, endpoint=vegpontok != "felso_nelkul")
            if e == 'order':
                # Az önmagában megadott rendtengelyhez hiányzik a független Hz-skála.
                if tengely != 'order':
                    raise ValueError(f"CSV {m['csv_sor']}. sor: natív order-export nem váltható Hz/CPM-re a Speed mezőből. Hz- vagy CPM-alapú spektrumexport szükséges.")
            else:
                hz = x if e == 'Hz' else x / 60
                if spektrumok is not None:
                    fordulat = fordulat_becsles(amplitudo, hz)
                sebesseg = fordulat['forgasi_frekvencia_hz']
                if tengely == 'order' and sebesseg is None:
                    raise ValueError(f"CSV {m['csv_sor']}. sor: order-ábrához nincs 5–100 Hz-ből becsülhető fordulat. Használd a Hz vagy CPM tengelyt.")
                x = hz if tengely == 'Hz' else hz * 60 if tengely == 'CPM' else hz / sebesseg
            cel = tengely
            cimke = {"Hz": "Frekvencia [Hz]", "CPM": "Frekvencia [CPM]",
                     "order": "Fordulati rend [× forgási frekvencia]"}[cel]
            if cel == "Hz":
                hz_lepes = (x[-1] - x[0]) / (n - 1)
            elif cel == "CPM":
                hz_lepes = (x[-1] - x[0]) / (n - 1) / 60
            elif fordulat['forgasi_frekvencia_hz'] is not None:
                hz_lepes = (x[-1] - x[0]) / (n - 1) * fordulat['forgasi_frekvencia_hz']
        axes.append(x)
        kiegeszitesek.append({
            "skala_export_egyseg": e, "skala_kezdo_export": k if e != "index" else np.nan,
            "skala_beallitas_forrasa": f, "skala_allapot": allapot,
            "spektrum_tengely": cel, "tengely_felirat": cimke,
            "tengely_elso_pont": float(x[0]), "tengely_utolso_pont": float(x[-1]),
            "tengely_pontkoz": float((x[-1] - x[0]) / (n - 1)),
            "frekvenciapontkoz_hz": hz_lepes,
            "csucs_tengelyertek": float(x[int(m["maximum_index_0tol"])]), **fordulat})
    # Korábbi méréseknél is megőrzöm az eredeti Speed mezőt.
    meta = meta.copy()
    if 'export_speed_hz' not in meta:
        meta['export_speed_hz'] = meta.get('forgasi_frekvencia_hz', np.nan)
    extra = pd.DataFrame(kiegeszitesek)
    return pd.concat([meta.drop(columns=extra.columns, errors='ignore').reset_index(drop=True), extra], axis=1), axes


def spektrum_abra(meres: pd.Series, amplitudo: np.ndarray, kep: Path, ablak: bool,
                  x: np.ndarray, amplitudo_skala: str = "linear",
                  jellemzes: dict | None = None) -> None:
    """Minden exportált pontot ábrázol; nincs átlagolás, ritkítás vagy új FFT."""
    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        if amplitudo_skala == "log" and not np.any(amplitudo > 0):
            raise ValueError(f"CSV {meres['csv_sor']}. sor: log amplitúdóskálához nincs pozitív érték.")
        # A logaritmikus ábrán csak a pozitív amplitúdók jelennek meg.
        y = np.ma.masked_less_equal(amplitudo, 0) if amplitudo_skala == "log" else amplitudo
        ax.plot(x, y, linewidth=0.7)
        ax.set_yscale(amplitudo_skala)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_xlabel(meres['tengely_felirat'])
        ax.set_ylabel(f"Amplitúdó [{meres['mertekegyseg']}] – {meres['detektalas']}")
        if jellemzes:
            jelolt = [c['index'] for c in jellemzes['csucsok']
                      if amplitudo_skala != 'log' or amplitudo[c['index']] > 0]
            if jelolt:
                ax.scatter(x[jelolt], amplitudo[jelolt], s=24, c='crimson', zorder=4,
                           label='Kiemelt lokális csúcsok')
            d = jellemzes['jellemzok']
            becsult_index = d.get('fordulat_csucs_index')
            if becsult_index is not None and math.isfinite(becsult_index):
                i = int(becsult_index)
                ax.scatter([x[i]], [amplitudo[i]], s=40, marker='D', c='#226633', zorder=5,
                           label=f"Vállalati fordulatbecslés: {d['fordulatszam_perc']:.6g} 1/min")
            szinek = ['#338833', '#aa7733', '#6633aa']
            for k, szin in zip((1, 2, 3), szinek):
                if d[f'harm_{k}_allapot'] != 'savmaximum_kiszamitva':
                    continue
                cel, tol = d[f'harm_{k}_cel_hz'], d[f'harm_{k}_tures_hz']
                egyseg = meres['spektrum_tengely']
                szorzo = (1 if egyseg == 'Hz' else 60 if egyseg == 'CPM'
                          else 1 / d['forgasi_frekvencia_hz'])
                ax.axvspan((cel - tol) * szorzo, (cel + tol) * szorzo, color=szin, alpha=0.1)
                ax.axvline(cel * szorzo, color=szin, linewidth=0.8, linestyle='--',
                           label=f'{k}× becsült forgási frekvencia és keresési sáv')
            if becsult_index is not None or jelolt or any(d[f'harm_{k}_allapot'] == 'savmaximum_kiszamitva' for k in (1, 2, 3)):
                ax.legend(loc='upper right', fontsize=7)
        pontnev = meres['meresi_pont'].rsplit('\\', 1)[-1]
        forma = "%Y.%m.%d %H:%M:%S" if meres['idopont_pontossag'] == "masodperc" else "%Y.%m.%d %H:%M"
        ido = meres['idopont'].strftime(forma)
        gep = meres['gepegyseg'] if pd.notna(meres['gepegyseg']) else "Ellenőrizendő gépazonosítás"
        ax.set_title(f"{gep} | {meres['oldal']}\n{pontnev} | {ido} | {meres['idopont_pontossag']} pontosság", fontsize=10)
        ax.grid(True, alpha=0.3)
        egyseg = meres['spektrum_tengely']
        megjegyzes = ("A frekvenciaskála beállítása ismeretlen; pontindex." if egyseg == "index" else
                      f"Pontköz: {meres['tengely_pontkoz']:.6g} {egyseg} | Skála: felhasználói beállítás | N = {len(amplitudo)}")
        if amplitudo_skala == "log":
            megjegyzes += " | Nem pozitív amplitúdók a log ábrán nem láthatók."
        fig.text(0.5, 0.01, megjegyzes, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        kep.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(kep, dpi=180)
        if ablak:
            plt.show()
    finally:
        plt.close(fig)


def beolvasas(fajl: Path, kodolas: str | None = None) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Szigorú beolvasás; hibás rekordot nem hagyunk ki és nem javítunk találomra."""
    rekordok = []
    spektrumok = []
    with io.StringIO(csv_szoveg(fajl, kodolas), newline="") as forras:
        olvaso = csv.reader(forras, delimiter=";")
        fejlec = next(olvaso, None)
        if fejlec is None:
            raise ValueError("A CSV üres.")
        if [c.strip() for c in fejlec[:13]] != FEJLEC or any(c.strip() for c in fejlec[13:]):
            raise ValueError("Nem támogatott fejléc. Pontosvesszővel tagolt SKF spektrumexport szükséges, a Data mezőtől amplitúdóértékekkel.")

        for sor_szam, sor in enumerate(olvaso, start=2):
            if not sor or not any(c.strip() for c in sor):
                continue
            if len(sor) < 13:
                raise ValueError(f"CSV {sor_szam}. sor: hiányos metaadatok vagy spektrum.")
            try:
                hossz = int(sor[5])
            except ValueError as exc:
                raise ValueError(f"CSV {sor_szam}. sor: a Lines mező nem egész szám.") from exc
            if hossz < 2 or 12 + hossz > len(sor):
                raise ValueError(f"CSV {sor_szam}. sor: érvénytelen spektrumhossz: {hossz}.")
            adatok = sor[12:12 + hossz]
            if any(not c.strip() for c in adatok):
                raise ValueError(f"CSV {sor_szam}. sor: hiányzó amplitúdóérték.")
            if any(c.strip() for c in sor[12 + hossz:]):
                raise ValueError(f"CSV {sor_szam}. sor: a Lines mezőnél több spektrumérték.")
            if not sor[0].strip() or not sor[2].strip() or not sor[3].strip():
                raise ValueError(f"CSV {sor_szam}. sor: hiányzó mérési pont, egység vagy detektálási mód.")
            datum, pontossag = datum_beolvasas(sor[1], sor_szam)
            vegfrekvencia = szam(sor[6], "End Frequency", sor_szam)
            if vegfrekvencia <= 0:
                raise ValueError(f"CSV {sor_szam}. sor: a végfrekvencia nem pozitív.")
            try:
                sebesseg = szam(sor[7], 'Speed (Hz)', sor_szam)
            except ValueError:
                sebesseg = np.nan  # Az exportból átvett nyers érték.
            amplitudo = np.array([szam(c, "amplitúdó", sor_szam) for c in adatok])
            rekordok.append({
                "meres_sorszam": len(rekordok) + 1,
                "csv_sor": sor_szam,
                "meresi_pont": sor[0],
                **hierarchia(sor[0]),
                "idopont": datum,
                "idopont_pontossag": pontossag,
                "meresi_nap": datum.date().isoformat(),
                "adatellenorzes": "rendben",
                "diagnosztikai_kiertekeles": "meg_nem_tortent",
                "mertekegyseg": sor[2],
                "detektalas": sor[3],
                "csatorna": sor[4],
                "spektrumhossz": hossz,
                "vegfrekvencia_export": vegfrekvencia,
                "export_speed_hz": sebesseg,
                "export_speed_nyers": sor[7],
                "forgasi_frekvencia_hz": np.nan,
                "process_value_nyers": sor[8],
                "process_value_egyseg": sor[9],
                "bov_gap_nyers": sor[10],
                "digital_nyers": sor[11],
                "maximum_amplitudo": float(amplitudo.max()),
                "maximum_index_0tol": int(amplitudo.argmax()),
            })
            spektrumok.append(amplitudo)
    if not rekordok:
        raise ValueError("A fájl nem tartalmaz mérési rekordot.")
    return pd.DataFrame(rekordok), spektrumok


def skf_kiegeszites(fajl: Path, kimenet: Path,
                   meta, spektrumok, tengelyek, jellemzok,
                   konfig: Path | None = None, pontok: Path | None = None) -> int:
    """Csak kifejezett kérésre importáljuk és indítjuk az SKF-részt.

    Az alapjellemzők már elkészültek; az SKF-jelentés a címkézéstől függetlenül
    saját almappába kerül, és az MI címke-/jellemzőtábláihoz nem nyúl.
    """
    print("\nSKF-kiértékelés a főprogram már kiszámított jellemzőiből és tengelyeiből...")
    hiba = ""
    jelentes = ""
    try:
        from skf_diagnosztika.skf_feldolgozas import run_prepared
        destination = run_prepared(meta, spektrumok, tengelyek, jellemzok, fajl,
                                   kimenet / "skf_kiertekeles",
                                   config_path=konfig or PROJEKT / "skf_beallitasok.json",
                                   points_path=pontok or PROJEKT / "skf_pontbeallitasok.csv")
        kod, jelentes = 0, str((destination / "alkalmi_kiertekeles.csv").resolve())
    except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        kod, hiba = 2, str(exc)
    statusz = {"modul": "opcionalis_skf_kiertekeles", "kilepesi_kod": kod,
               "bemenet": str(fajl.resolve()), "kimeneti_mappa": str((kimenet / "skf_kiertekeles").resolve()),
               "hiba": hiba, "mi_cimkek_modositva": False}
    statusz.update(jelentes=jelentes, export_ujraolvasva=False,
                    adatfolyam="jellemzokinyeres_utan_kozvetlen_adatatadas")
    (kimenet / "skf_modul_statusz.json").write_text(
        json.dumps(statusz, ensure_ascii=False, indent=2), encoding="utf-8")
    if kod:
        print("Az opcionális SKF-futás nem fejeződött be sikeresen. "
              "Az alapfeldolgozás elkészült táblái megmaradtak.")
        if hiba:
            print("SKF-rész: " + hiba)
    return kod


def mi_kiegeszites(jellemzok: pd.DataFrame, kimenet: Path, modell_ut: Path | None = None) -> int:
    """A már kiszámított jellemzőkből becsül; a kézi címkékhez nem nyúl."""
    try:
        from mi_becsles import modell_betoltes, becsles_tabla, mentes
        modell, meta = modell_betoltes(modell_ut, PROJEKT / 'outputs' / 'mi_modellek')
        eredmeny, jelent = becsles_tabla(jellemzok, modell, meta)
        cel = kimenet / 'mi_becsles' / meta['modell_sha256'][:10]
        mentes(eredmeny, jelent, cel)
        kod = 0 if jelent['ertekelt_rekordok'] else 2
        hiba = '' if kod == 0 else 'Nincs a modellhez használható mérési rekord; a kihagyás okai a táblában láthatók.'
    except (ImportError, OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        kod, hiba = 2, str(exc)
        print('Az MI-becslés nem sikerült: ' + hiba)
    (kimenet / 'mi_modul_statusz.json').write_text(json.dumps({
        'kilepesi_kod': kod, 'hiba': hiba, 'cimkek_modositva': False,
        'adatfolyam': 'jellemzokinyeres_utan_kozvetlen_adatatadas',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return kod


def main() -> int:
    try:
        from jellemzok import Parameterek, jellemzotabla
        from cimkezes import rekord_azonositok, feldolgozas as cimke_feldolgozas
    except ModuleNotFoundError as exc:
        raise SystemExit('Hiányzó modul: másold a ZIP teljes src mappáját a fordulatszam.py fájllal és az SKF-almodulokkal együtt. SciPy telepítése: python -m pip install scipy') from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fajl", nargs="?", type=Path,
                        help="CSV elérési útja; elhagyva fájlválasztás a data mappából.")
    parser.add_argument("--data-mappa", type=Path, default=PROJEKT / "data")
    parser.add_argument("--kodolas", help="Kézi kódolás, például cp1250; alapból UTF-8/UTF-16 BOM/cp1250 felismerés.")
    parser.add_argument("--gep-terkep", type=Path, help="Opcionális JSON a saját géphierarchiákhoz.")
    parser.add_argument("--export-egyseg", choices=["Hz", "CPM", "order", "index"],
                        help="Az export végfrekvenciájának egysége. Elhagyva meglévő skála vagy jelölt standardprofil, kérdés nélkül.")
    parser.add_argument("--interaktiv-skala", action="store_true",
                        help="Opcionális kézi skálaválasztás; alapból nem kérdez rá.")
    parser.add_argument("--kezdo-frekvencia", type=float,
                        help="A spektrum kezdőértéke az export egységében; például 0.")
    parser.add_argument("--tengely", choices=["Hz", "CPM", "order"], default="Hz",
                        help="A kép vízszintes tengelye (alapból Hz).")
    parser.add_argument("--skala-terkep", type=Path,
                        help="Opcionális JSON útvonalanként eltérő kezdőértékekhez/egységekhez.")
    parser.add_argument("--amplitudo-skala", choices=["linear", "log"], default="linear")
    parser.add_argument("--csak-jellemzok", action="store_true", help="Minden rekord jellemzőit menti, képek készítése nélkül.")
    parser.add_argument("--csucsok-szama", type=int, default=5, help="Kiemelt csúcsok száma, 1–50 (alapból 5).")
    parser.add_argument("--prominencia-rel", type=float, default=0.05,
                        help="Csúcskeresés prominenciaküszöbe a spektrummaximum arányában (alapból 0.05).")
    parser.add_argument("--harmonikus-tures-rel", type=float, default=0.02,
                        help="Harmonikus keresési félsáv relatív része, legalább egy pontköz (alapból 0.02).")
    parser.add_argument("--meres", type=int,
                        help="Csak ennek a rekordnak az ábráját készíti el (1-től). Elhagyva minden rekordot ment.")
    parser.add_argument("--csak-mentes", action="store_true",
                        help="Ábra mentése ablak megnyitása nélkül.")
    parser.add_argument("--skf-kiertekeles", action="store_true",
                        help="Az alapfeldolgozás után a kiválasztott CSV-n külön SKF-tünetvizsgálatot is futtat.")
    parser.add_argument("--skf-konfig", type=Path,
                        help="Az opcionális SKF-rész saját konfigurációja; a --skf-kiertekeles kapcsolóval együtt.")
    parser.add_argument("--skf-pontok", type=Path,
                        help="Az opcionális SKF-rész pontbeállítási CSV-je; a --skf-kiertekeles kapcsolóval együtt.")
    parser.add_argument("--mi-becsles", action="store_true",
                        help="A jellemzőkinyerés után a mentett RF/SVM modellel is besorolja a méréseket.")
    parser.add_argument("--mi-modell", type=Path,
                        help="Korábban mentett modellfájl; a --mi-becsles kapcsolóval együtt.")
    args = parser.parse_args()
    if args.interaktiv_skala and args.export_egyseg is not None:
        parser.error("Vagy --interaktiv-skala, vagy --export-egyseg adható meg.")
    if not args.skf_kiertekeles and (args.skf_konfig is not None or args.skf_pontok is not None):
        parser.error("A --skf-konfig és --skf-pontok mellé --skf-kiertekeles szükséges.")
    if args.mi_modell is not None and not args.mi_becsles:
        parser.error("A --mi-modell mellé --mi-becsles szükséges.")
    jellemzo_param = Parameterek(args.csucsok_szama, args.prominencia_rel, args.harmonikus_tures_rel)
    try:
        jellemzo_param.ellenorzes()
    except ValueError as exc:
        parser.exit(1, f"Jellemzőbeállítási hiba: {exc}\n")
    if args.fajl is None:
        try:
            args.fajl = fajlvalasztas(args.data_mappa)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"Fájlválasztási hiba: {exc}\n")
    if not args.fajl.is_file():
        parser.exit(1, f"Nem található a CSV: {args.fajl}\nMásold az eredeti exportot a projekt data mappájába, vagy add meg az útvonalát idézőjelek között.\n")
    try:
        meta, spektrumok = beolvasas(args.fajl, args.kodolas)
        if args.gep_terkep:
            meta = gepterkep_alkalmazas(meta, args.gep_terkep)
    except (ValueError, OSError, UnicodeError, LookupError, csv.Error) as exc:
        parser.exit(1, f"Beolvasási hiba: {exc}\n")
    if args.meres is not None and not 1 <= args.meres <= len(meta):
        parser.exit(1, f"A --meres értéke 1 és {len(meta)} közé essen.\n")
    try:
        rekord_skalak = None
        if args.export_egyseg is None:
            if args.kezdo_frekvencia is not None:
                raise ValueError("A --kezdo-frekvencia mellé --export-egyseg is szükséges.")
            if args.interaktiv_skala:
                egyseg, kezdo, skala_forras = skala_bekeres()
            else:
                egyseg, kezdo = "Hz", 0.0
                skala_forras = "automatikus_standardprofil:0Hz-EndFrequency_Hz_ket_vegpont:feltetelezett"
                alap_terkep = PROJEKT / "skala_terkep.json"
                if args.skala_terkep is None and alap_terkep.is_file():
                    if json.loads(alap_terkep.read_text(encoding="utf-8-sig")):
                        args.skala_terkep = alap_terkep
                if args.skf_kiertekeles or args.mi_becsles:
                    try:
                        from skf_diagnosztika.skf_beallitasok import main_scale_overrides
                        rekord_skalak = main_scale_overrides(meta,
                            args.skf_konfig or PROJEKT / "skf_beallitasok.json",
                            args.skf_pontok or PROJEKT / "skf_pontbeallitasok.csv")
                    except (ImportError, ValueError, OSError) as exc:
                        print(f"Meglévő SKF-skálát nem lehetett átvenni: {exc}")
        else:
            egyseg, kezdo, skala_forras = args.export_egyseg, args.kezdo_frekvencia, "parancssori_beallitas"
            if egyseg != "index" and kezdo is None:
                raise ValueError("Add meg a --kezdo-frekvencia értékét is, például 0. A CSV nem tárolja külön.")
            if egyseg == "index" and kezdo is not None:
                raise ValueError("Pontindexhez ne adj kezdőfrekvenciát.")
        meta, tengelyek = skalak_alkalmazas(meta, egyseg, kezdo, args.tengely,
                                           skala_forras, args.skala_terkep, rekord_skalak,
                                           spektrumok=spektrumok)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(1, f"Skálabeállítási hiba: {exc}\n")

    meta = rekord_azonositok(meta, spektrumok)
    print("CSV-beolvasó és jellemzőkinyerő – 11. változat; vállalati 5–100 Hz fordulatbecslés")
    print(f"Kiválasztott fájl: {args.fajl.resolve()}")
    print(f"Mérési rekordok / spektrumok száma: {len(meta)}")
    print(f"Mérési útvonalak száma: {meta['meresi_pont'].nunique()}")
    print(f"Mérési napok száma: {meta['meresi_nap'].nunique()}")
    print(f"Névkonvenció alapján azonosított gépegységek: {meta['gepegyseg_utvonal'].nunique()}")
    print("A gépazonosítás az ellenőrzött szivattyús és légkezelős névkonvenciókból történik.")
    ismeretlen = int(meta['gepegyseg_utvonal'].isna().sum())
    print(f"Nem felismert géphierarchiájú rekordok: {ismeretlen}")
    if ismeretlen:
        print("FIGYELEM: ezek a rekordok is megmaradtak, géphez rendelésük ellenőrzést igényel.")
    napi = napi_osszesites(meta)
    if int(napi['meresi_rekordok'].sum()) != len(meta):
        raise RuntimeError("A napi összesítés nem fedi le az összes rekordot.")
    print("Napi áttekintés:")
    print(napi[["gepegyseg", "oldal", "meresi_nap", "meresi_rekordok", "meresi_utvonalak"]].to_string(index=False))
    print(f"Adatellenőrzésen átment rekordok: {len(meta)}/{len(meta)}")
    print("Spektrumtengelyek:", meta['spektrum_tengely'].value_counts().to_dict())
    print("A skála eredete a táblákban szerepel; az automatikus standardprofil feltételezés, nem külön exportált mérési adat.")
    print("Spektrumhosszak:", meta["spektrumhossz"].value_counts().to_dict())
    print("Mértékegységek és detektálási módok:")
    print(meta.groupby(["mertekegyseg", "detektalas"]).size().to_string())
    print("Fordulatbecslés (Speed nélkül):", meta['fordulat_allapot'].value_counts().to_dict())
    ismetlodes = meta.duplicated(["meresi_pont", "idopont", "csatorna"], keep=False)
    print("Ismétlődő pont–időpont–csatorna kulcsú sorok:", int(ismetlodes.sum()))
    jellemzok, csucsok, rajzjelolesek = jellemzotabla(meta, spektrumok, tengelyek, jellemzo_param)
    print("Jellemzőkinyerés:", jellemzok['jellemzok_allapot'].value_counts().to_dict())
    print("1× sávok:", jellemzok['harm_1_allapot'].value_counts().to_dict())

    # A fájl tartalma különbözteti meg az azonos nevű exportokat.
    ujjlenyomat = hashlib.sha256(args.fajl.read_bytes()).hexdigest()[:10]
    # Minden skálabeállítás külön kimenetet kap.
    beallitasok = {
        "verzio": 11, "fordulat_modszer": MODSZER,
        "tengely": args.tengely, "amplitudo_skala": args.amplitudo_skala,
        "jellemzok": vars(jellemzo_param),
        "skalak": meta[['meresi_pont', 'skala_export_egyseg', 'skala_kezdo_export',
                        'skala_beallitas_forrasa', 'csoport_utvonal']].fillna('').to_dict('records')}
    skala_hash = hashlib.sha256(json.dumps(beallitasok, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:8]
    kimenet = PROJEKT / "outputs" / f"{args.fajl.stem}_{ujjlenyomat}" / f"feldolgozas_v11_{skala_hash}"
    kimenet.mkdir(parents=True, exist_ok=True)
    (kimenet / 'abra_beallitasok.json').write_text(json.dumps(beallitasok, ensure_ascii=False, indent=2), encoding='utf-8')
    meta.to_csv(kimenet / "meresi_attekintes.csv", sep=";", decimal=",",
                encoding="utf-8-sig", index=False, date_format="%Y-%m-%d %H:%M:%S")
    napi.to_csv(kimenet / "napi_attekintes.csv", sep=";", decimal=",",
                encoding="utf-8-sig", index=False, date_format="%Y-%m-%d %H:%M:%S")
    jellemzok.to_csv(kimenet / "jellemzok.csv", sep=";", decimal=",", encoding="utf-8-sig",
                     index=False, date_format="%Y-%m-%d %H:%M:%S", na_rep="")
    csucsok.to_csv(kimenet / "csucsok.csv", sep=";", decimal=",", encoding="utf-8-sig",
                   index=False, date_format="%Y-%m-%d %H:%M:%S", na_rep="")
    print(f"Jellemzőtáblázat: {kimenet / 'jellemzok.csv'}")
    print(f"Kiemelt csúcsok: {kimenet / 'csucsok.csv'}")
    print("Az üres jellemző nem nulla: az állapotoszlop jelzi a számítás korlátját. Gépállapot-besorolás még nincs.")
    skf_kod = 0
    if args.skf_kiertekeles:
        skf_kod = skf_kiegeszites(args.fajl, kimenet, meta, spektrumok, tengelyek, jellemzok,
                                  args.skf_konfig, args.skf_pontok)
    if skf_kod == 130:
        return skf_kod
    mi_kod = mi_kiegeszites(jellemzok, kimenet, args.mi_modell) if args.mi_becsles else 0
    kilepesi_kod = max(skf_kod, mi_kod)
    try:
        cimke_feldolgozas(jellemzok, kimenet)
    except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
        plusz = " Az SKF-részvizsgálat külön jelentése is elkészült." if args.skf_kiertekeles and skf_kod == 0 else ""
        parser.exit(1, f"Címkézési hiba: {exc}\nA jellemzők elkészültek.{plusz} Javítsd a címke-/ellenőrző fájlt, és futtasd a cimkezes.py programot.\n")
    if args.csak_jellemzok:
        return kilepesi_kod
    indexek = [args.meres - 1] if args.meres is not None else list(range(len(meta)))
    if args.amplitudo_skala == 'log' and any(not np.any(spektrumok[i] > 0) for i in indexek):
        parser.exit(1, "Valamelyik spektrumhoz nincs pozitív amplitúdó; válaszd a --amplitudo-skala linear beállítást.\n")
    print(f"Spektrumképek mentése: {len(indexek)} darab...", flush=True)
    keplista = []
    for kesz, index in enumerate(indexek, 1):
        meres = meta.iloc[index]
        kulcs = meres['csoport_utvonal']
        ismert = pd.notna(meres['gepegyseg'])
        nev = str(meres['gepegyseg']) if ismert else 'ELLENORIZENDO'
        mappanev = biztonsagos_nev(nev) + '_' + hashlib.sha256(kulcs.encode('utf-8')).hexdigest()[:8]
        pontnev = biztonsagos_nev(meres['meresi_pont'].rsplit('\\', 1)[-1])
        idonev = meres['idopont'].strftime('%Y%m%d_%H%M%S' if meres['idopont_pontossag'] == 'masodperc' else '%Y%m%d_%H%M')
        kep = kimenet / 'spektrumok' / mappanev / biztonsagos_nev(meres['oldal']) / f"{index + 1:04d}_{pontnev}_{idonev}.png"
        spektrum_abra(meres, spektrumok[index], kep,
                      ablak=args.meres is not None and not args.csak_mentes,
                      x=tengelyek[index], amplitudo_skala=args.amplitudo_skala,
                      jellemzes=rajzjelolesek[index])
        keplista.append({"meres_sorszam": index + 1, "meresi_pont": meres['meresi_pont'],
                         "idopont": meres['idopont'], "idopont_pontossag": meres['idopont_pontossag'],
                         "tengely": meres['spektrum_tengely'],
                         "skala_allapot": meres['skala_allapot'],
                         "kep": kep.relative_to(kimenet).as_posix()})
        if kesz % 25 == 0 or kesz == len(indexek):
            print(f"Elmentett képek: {kesz}/{len(indexek)}", flush=True)
    pd.DataFrame(keplista).to_csv(kimenet / 'spektrum_keplista.csv', sep=';', index=False,
                                encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M:%S')
    print(f"Áttekintés: {kimenet / 'meresi_attekintes.csv'}")
    print(f"Napi áttekintés: {kimenet / 'napi_attekintes.csv'}")
    print(f"Spektrumképek: {kimenet / 'spektrumok'}")
    print("A spektrum pontköze az exportált pontok távolsága, nem az időbélyeg pontossága.")
    print("Az MI-modell tanítása a mi_tanitas.py programmal indítható; alkalmazása: --mi-becsles.")
    if args.skf_kiertekeles:
        print("Az opcionális SKF-tünetvizsgálat eredménye a külön SKF-jelentésben található.")
    return kilepesi_kod


if __name__ == "__main__":
    raise SystemExit(main())
