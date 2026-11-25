# A címkézett spektrumjellemzőkből modellt tanít, majd elmenti a kiválasztott modellt.

r"""RF/SVM tanítása és mentése.

Indítás:
    python .\src\mi_tanitas.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

try:
    import joblib
    from mi_modell import FORMATUM, tanitas, verziok
except ModuleNotFoundError:
    if __name__ == '__main__':
        raise SystemExit('Hiányzó MI-csomag. Telepítés: python -m pip install -r requirements_mi.txt')
    raise
from tanito_adatok import JELLEMZOK, osszeallitas, mentes as adatok_mentes
from fordulatszam import MODSZER


PROJEKT = Path(__file__).resolve().parent.parent


def json_mentes(ut: Path, adat: dict) -> None:
    ut.parent.mkdir(parents=True, exist_ok=True)
    tmp = ut.with_name(ut.name + '.tmp')
    tmp.write_text(json.dumps(adat, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(ut)


def _csv(t: pd.DataFrame, ut: Path) -> None:
    t.to_csv(ut, sep=';', decimal=',', encoding='utf-8-sig', index=False, na_rep='')


def eredmenyek_mentese(tabla: pd.DataFrame, adatjelent: dict, eredmeny: dict,
                      mappa: Path, csoportositas: str, teszt_arany: float, seed: int) -> dict:
    mappa.mkdir(parents=True, exist_ok=True)
    fejlesztes, teszt = eredmeny['fejlesztes_index'], eredmeny['teszt_index']
    felosztas = tabla[['rekord_id', 'felosztasi_csoport', 'gepegyseg_utvonal', 'cel_hibas']].copy()
    felosztas['idopont'] = tabla['idopont']
    felosztas['ertekelesi_csoport'] = eredmeny['csoportok']
    felosztas['halmaz'] = 'fejlesztes'
    felosztas.loc[teszt, 'halmaz'] = 'teszt'
    felosztas['belso_validacios_kor'] = pd.Series(pd.NA, index=tabla.index, dtype='Int64')
    for kor, (_, validacio) in enumerate(eredmeny['belso_cv'], 1):
        felosztas.loc[fejlesztes[validacio], 'belso_validacios_kor'] = kor
    _csv(felosztas, mappa / 'felosztas.csv')
    reszletes = felosztas.iloc[teszt].copy()
    for nev, becsult in eredmeny['becslesek'].items():
        reszletes['becsult_' + nev] = becsult
    _csv(reszletes, mappa / 'teszt_becslesek.csv')
    if csoportositas == 'idorend':
        by_machine = []
        for machine, part in reszletes.groupby('gepegyseg_utvonal'):
            for name in eredmeny['becslesek']:
                pred, actual = part['becsult_' + name], part.cel_hibas
                by_machine.append(dict(gep=machine, modell=name, rekordok=len(part),
                                       TP=int(((actual==1)&(pred==1)).sum()), TN=int(((actual==0)&(pred==0)).sum()),
                                       FP=int(((actual==0)&(pred==1)).sum()), FN=int(((actual==1)&(pred==0)).sum())))
        _csv(pd.DataFrame(by_machine), mappa / 'celgepek_eredmenyei.csv')
    sorok = []
    for r in eredmeny['eredmenyek']:
        sorok.append({'modell': r['modell'], **{'teszt_' + k: v for k, v in r['teszt'].items()},
                      **{'csoportsulyozott_' + k: v for k, v in r['csoportsulyozott_teszt'].items()}})
    _csv(pd.DataFrame(sorok), mappa / 'modell_osszehasonlitas.csv')
    for nev, kereso in eredmeny['keresok'].items():
        kereses = pd.DataFrame(kereso.cv_results_)
        oszlopok = ['params', 'mean_test_balanced_accuracy', 'std_test_balanced_accuracy',
                    'mean_test_f1', 'mean_test_recall', 'rank_test_balanced_accuracy']
        kereses['params'] = kereses.params.map(lambda p: json.dumps(p, ensure_ascii=False, sort_keys=True))
        _csv(kereses[oszlopok], mappa / f'parameterkereses_{nev}.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for nev, becsult in eredmeny['becslesek'].items():
        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            ConfusionMatrixDisplay.from_predictions(
                tabla.iloc[teszt].cel_hibas, becsult, labels=[0, 1],
                display_labels=['Nem hibás', 'Hibás'], colorbar=False, ax=ax, cmap='Blues')
            ax.set_xlabel('Becsült állapot')
            ax.set_ylabel('Igazolt állapot')
            ax.set_title(nev + ' – elkülönített teszt')
            fig.tight_layout()
            fig.savefig(mappa / f'konfuzios_matrix_{nev}.png', dpi=170)
        finally:
            plt.close(fig)
    modell_ut = mappa / 'valasztott_modell.joblib'
    joblib.dump({'formatum': FORMATUM, 'fordulat_modszer': MODSZER, 'pipeline': eredmeny['modell']}, modell_ut, compress=3)
    metadata = {
        'formatum': FORMATUM, 'modell_tipus': eredmeny['valasztott'], 'verziok': verziok(),
        'fordulat_modszer': MODSZER,
        'keszult_utc': datetime.now(timezone.utc).isoformat(),
        'jellemzo_oszlopok': JELLEMZOK, 'mertekegyseg': adatjelent['kivalasztott_egyseg'],
        'detektalas': adatjelent['kivalasztott_detektalas'],
        'harmonikus_tures_rel': eredmeny['harmonikus_tures_rel'],
        'csoportositas': csoportositas, 'illesztesi_halmaz': 'fejlesztes',
        'tanitott_rekord_idk': tabla.iloc[fejlesztes].rekord_id.tolist(),
        'teszt_rekord_idk': tabla.iloc[teszt].rekord_id.tolist(),
        'tanitott_gepek': sorted(set(tabla.iloc[fejlesztes].gepegyseg_utvonal)),
        'modell_sha256': hashlib.sha256(modell_ut.read_bytes()).hexdigest(),
    }
    json_mentes(modell_ut.with_suffix('.json'), metadata)
    valasztott_teszt = next(r['teszt'] for r in eredmeny['eredmenyek'] if r['modell'] == eredmeny['valasztott'])
    referencia = next(r['teszt'] for r in eredmeny['eredmenyek'] if r['modell'] == 'Referencia')
    jelent = {
        'formatum': FORMATUM, 'allapot': 'sikeres', 'valasztott_modell': eredmeny['valasztott'],
        'modellvalasztas': 'csak a fejlesztési rész belső CV balanced_accuracy pontszáma alapján',
        'csoportositas': csoportositas, 'seed': seed, 'kert_teszt_arany': teszt_arany,
        'fejlesztesi_rekordok': len(fejlesztes), 'teszt_rekordok': len(teszt),
        'fejlesztesi_csoportok': len(set(eredmeny['csoportok'][fejlesztes])),
        'teszt_csoportok': len(set(eredmeny['csoportok'][teszt])), 'belso_cv_korok': len(eredmeny['belso_cv']),
        'belso_cv_ranglista': eredmeny['ranglista'], 'teszteredmenyek': eredmeny['eredmenyek'],
        'jobb_a_referencianal_a_teszten': valasztott_teszt['balanced_accuracy'] > referencia['balanced_accuracy'],
        'adatkeszlet': adatjelent, 'modell': metadata,
        'forrasprogram_sha256': {n: hashlib.sha256((PROJEKT / 'src' / n).read_bytes()).hexdigest()
                                for n in ['mi_modell.py', 'mi_tanitas.py', 'mi_idorend.py', 'jellemzok.py', 'fordulatszam.py', 'cimkezes.py', 'tanito_adatok.py']},
        'korlat': 'A mutatók spektrumrekordokra vonatkoznak. A csoportosítás önmagában nem igazolja '
                  'a gépállapotok függetlenségét. A teszt nem vett részt az előfeldolgozás, '
                  'a paraméterek vagy az algoritmus kiválasztásában; a mentett modell csak a fejlesztési részen tanult.',
    }
    json_mentes(mappa / 'tanitasi_jelentes.json', jelent)
    osszeg = [
        'MI-MODELL TANÍTÁSA', f'Választott modell: {eredmeny["valasztott"]}',
        f'Fejlesztés: {len(fejlesztes)} rekord; teszt: {len(teszt)} rekord.',
        f'Belső csoportos CV: {len(eredmeny["belso_cv"])} kör; csoportosítás: {csoportositas}.',
        'A választás a belső CV alapján történt. A teszteredményt a választás után számítottuk.',
        f'Teszt balanced accuracy: {valasztott_teszt["balanced_accuracy"]:.4f}',
        f'Teszt hibafelismerési arány (recall): {valasztott_teszt["recall"]:.4f}',
        f'Téves riasztás (FP): {int(valasztott_teszt["FP"])}; fel nem ismert hiba (FN): {int(valasztott_teszt["FN"])}.',
        'A mentett modell nem tanult a tesztrészből. A mutatók a jelenlegi adatokra és felosztásra érvényesek.',
        'A gép+napi csoportos értékelés nem igazolja az új gépekre való általánosítást.',
    ]
    if not jelent['jobb_a_referencianal_a_teszten']:
        osszeg.append('A kiválasztott modell teszt balanced accuracy értéke nem jobb a többségi referenciaosztályozóénál.')
    (mappa / 'OSSZEGZES.txt').write_text('\n'.join(osszeg) + '\n', encoding='utf-8')
    return jelent


def futtatas(fajlok: list[Path], kimenet: Path, adat_mappa: Path, egyseg='mm/s', detektalas='RMS',
             csoportositas='gep_nap', teszt_arany=0.2, cv_max=5, seed=42, gepterv=None, teszt_kezdete=None) -> dict:
    futas = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    mappa = kimenet / futas
    statusz = {'allapot': 'folyamatban', 'futas': futas, 'kimenet': str(mappa.resolve())}
    json_mentes(kimenet / 'utolso_tanitas.json', statusz)
    try:
        attekintes, adatjelent = osszeallitas(fajlok, egyseg, detektalas)
        adatok_mentes(attekintes, adatjelent, adat_mappa)
        if adatjelent['forrashibak']:
            raise ValueError('Az adat-összeállítás forráshibákat jelzett; előbb ezeket kell tisztázni.')
        tabla = attekintes[attekintes.tanitasi_jelolt.astype(bool)].reset_index(drop=True)
        if tabla.empty:
            raise ValueError("Nincs igazolt, felhasználható tanítási rekord. Ellenőrizd az adatkeszlet_attekintes.csv kizárási okait.")
        if gepterv is not None or teszt_kezdete is not None:
            if gepterv is None or not teszt_kezdete or csoportositas != "gep_nap":
                raise ValueError("Az időrendi tanításhoz a --gepterv és --teszt-kezdete együtt szükséges, külön csoportosítás nélkül.")
            from mi_idorend import terv_beolvasas, idorendi_minta
            roles = terv_beolvasas(gepterv)
            tabla, excluded = idorendi_minta(tabla, roles, teszt_kezdete)
            adatjelent = dict(adatjelent, gepterv=roles, teszt_kezdete=teszt_kezdete,
                              terv_miatt_kihagyott_jeloltek=excluded)
            csoportositas, teszt_arany = "idorend", None
        eredmeny = tanitas(tabla, csoportositas, teszt_arany, cv_max, seed, teszt_kezdete=teszt_kezdete)
        jelent = eredmenyek_mentese(tabla, adatjelent, eredmeny, mappa, csoportositas, teszt_arany, seed)
        statusz.update(allapot='sikeres', modell_utvonal=str((mappa / 'valasztott_modell.joblib').resolve()))
        json_mentes(kimenet / 'utolso_tanitas.json', statusz)
        return jelent
    except (ValueError, OSError, ImportError, TypeError, KeyError) as exc:
        statusz.update(allapot='nem_sikerult', hiba=str(exc))
        json_mentes(mappa / 'tanitas_statusz.json', statusz)
        json_mentes(kimenet / 'utolso_tanitas.json', statusz)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jellemzotablak', nargs='*', type=Path)
    parser.add_argument('--bemenet', type=Path, default=PROJEKT / 'outputs')
    parser.add_argument('--kimenet', type=Path, default=PROJEKT / 'outputs' / 'mi_modellek')
    parser.add_argument('--egyseg', default='mm/s')
    parser.add_argument('--detektalas', default='RMS')
    parser.add_argument('--csoportositas', choices=['gep_nap', 'gep'], default='gep_nap')
    parser.add_argument('--gepterv', type=Path, help='Gépútvonal és szerep szerinti CSV')
    parser.add_argument('--teszt-kezdete', help='A teszt első napja, ÉÉÉÉ-HH-NN')
    parser.add_argument('--teszt-arany', type=float, default=0.2)
    parser.add_argument('--cv-max', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    fajlok = args.jellemzotablak or list(args.bemenet.rglob('jellemzok.csv'))
    try:
        futtatas(fajlok, args.kimenet, args.kimenet / 'adatkeszlet', args.egyseg,
                 args.detektalas, args.csoportositas, args.teszt_arany, args.cv_max, args.seed, args.gepterv, args.teszt_kezdete)
    except ValueError as exc:
        print(f'A tanítás nem indítható vagy nem fejezhető be: {exc}')
        print(f'Állapotjelentés: {args.kimenet / "utolso_tanitas.json"}')
        return 2
    except (OSError, ImportError, TypeError, KeyError) as exc:
        print(f'MI-tanítási hiba: {exc}')
        return 1
    statusz = json.loads((args.kimenet / 'utolso_tanitas.json').read_text(encoding='utf-8'))
    print((Path(statusz['kimenet']) / 'OSSZEGZES.txt').read_text(encoding='utf-8'))
    print(f'Modell és részletes eredmények: {statusz["kimenet"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
