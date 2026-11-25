# Az 5–100 Hz közötti legnagyobb amplitúdó frekvenciájából becsüli a tengely fordulatszámát.

from __future__ import annotations

import math

MODSZER = "vallalati_savmaximum_5_100_hz_v1"
# A korábbi, verziószám nélküli név is ugyanezt a számítást jelölte.
AZONOS_MODSZEREK = frozenset((MODSZER, "vallalati_savmaximum_5_100_hz"))
FORRAS = "Vállalati eljárás: az 5–100 Hz közötti legnagyobb amplitúdójú pont frekvenciája"
ALSO_HZ, FELSO_HZ = 5.0, 100.0


def becsles(amplitudok, frekvenciak_hz=None) -> dict:
    """Egyetlen spektrum becslése. Azonos maximumoknál a kisebb frekvencia dönt.

    Részleges sávon is a rendelkezésre álló pontok maximumát adja, külön jelölve
    a lefedést. Üres/nullás sávból nem talál ki fordulatot, Speed-visszaesés nincs.
    Az index az eredeti amplitúdótömbben értendő. A bemeneteket nem módosítja.
    """
    d = dict(fordulat_modszer=MODSZER, fordulat_forrasa=FORRAS,
             fordulat_allapot="nincs_hz_skala", forgasi_frekvencia_hz=None,
             fordulatszam_perc=None, fordulat_csucs_amplitudo=None,
             fordulat_csucs_index=None, fordulat_sav_pontszam=0,
             fordulat_teljes_sav=False, fordulat_maximumok_szama=0)
    if frekvenciak_hz is None:
        return d
    try:
        y, f = list(map(float, amplitudok)), list(map(float, frekvenciak_hz))
    except (ValueError, TypeError, OverflowError):
        return dict(d, fordulat_allapot="hibas_spektrum_vagy_tengely")
    if (len(y) < 2 or len(y) != len(f) or
            any(not math.isfinite(a) or a < 0 for a in y) or
            any(not math.isfinite(a) or a < 0 for a in f) or
            any(b <= a for a, b in zip(f, f[1:]))):
        return dict(d, fordulat_allapot="hibas_spektrum_vagy_tengely")
    eps = 1e-10  # A tűrés a Hz és CPM közötti átváltás kerekítését kezeli.
    d["fordulat_teljes_sav"] = f[0] <= ALSO_HZ + eps and f[-1] >= FELSO_HZ - eps
    indices = [i for i, hz in enumerate(f) if ALSO_HZ - eps <= hz <= FELSO_HZ + eps]
    d["fordulat_sav_pontszam"] = len(indices)
    if not indices:
        return dict(d, fordulat_allapot="nincs_pont_az_5_100_hz_savban")
    index = max(indices, key=y.__getitem__)
    if y[index] <= 0:
        return dict(d, fordulat_allapot="nincs_pozitiv_amplitudo_az_5_100_hz_savban")
    d.update(fordulat_allapot="becsult", forgasi_frekvencia_hz=f[index],
             fordulatszam_perc=60 * f[index], fordulat_csucs_amplitudo=y[index],
             fordulat_csucs_index=index,
             fordulat_maximumok_szama=sum(y[i] == y[index] for i in indices))
    return d
