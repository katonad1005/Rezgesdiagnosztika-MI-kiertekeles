# Meghatározza a frekvenciatengelyt, és a keresett frekvenciák környezetében csúcsokat keres.

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .skf_fajlkezeles import number
from fordulatszam import becsles as fordulat_becsles


@dataclass
class Axis:
    start: float
    step: float
    count: int

    @property
    def end(self):
        return self.start + self.step * (self.count - 1)

    def frequency(self, index):
        return self.start + self.step * index


def make_axis(record, settings):
    if settings.get("skala_ellenorzott") is not True:
        raise ValueError("A frekvenciaskála nincs ellenőrizve")
    factor = {"Hz": 1.0, "CPM": 1.0 / 60}.get(settings.get("frekvenciaegyseg"))
    if factor is None or settings.get("kezdo_frekvencia_hz") is None:
        raise ValueError("Hiányzó kezdőfrekvencia vagy export-frekvenciaegység")
    start, end = number(settings["kezdo_frekvencia_hz"]), record.end * factor
    n = len(record.values)
    if start < 0 or end <= start or n < 2:
        raise ValueError("Érvénytelen frekvenciatartomány")
    denominator = n - 1 if settings.get("vegpontok") == "ket_vegpont" else n
    return Axis(start, (end - start) / denominator, n)


def rotational_speed(record, settings):
    """Kompatibilis belépési pont, kizárólag a vállalati sávmaximumot használja."""
    axis = make_axis(record, settings)
    estimate = fordulat_becsles(record.values, [axis.frequency(i) for i in range(axis.count)])
    if estimate['forgasi_frekvencia_hz'] is None or record.errors:
        raise ValueError(estimate['fordulat_allapot'])
    return estimate['forgasi_frekvencia_hz'], estimate['fordulat_forrasa']


def modality(unit):
    unit = unit.casefold().replace(" ", "").replace("²", "2").replace("^", "")
    if unit in {"mm/s", "mm/sec", "m/s", "m/sec", "in/s", "in/sec", "ips"}:
        return "sebesseg"
    if unit in {"g", "m/s2", "mm/s2"}:
        return "gyorsulas"
    if unit in {"ge", "g(e)"}:
        return "burkologorbe"
    if unit in {"um", "µm", "μm", "micron", "microns", "mil", "mils", "mm"}:
        return "elmozdulas"
    return "ismeretlen"


class Peaks:
    def __init__(self, record, axis, technical, features=None):
        self.values, self.axis = record.values, axis
        self.pct = number(technical["frekvencia_tures_szazalek"]) / 100
        self.ratio = number(technical["minimum_hatter_arany"])
        self.relative = number(technical["minimum_relativ_amplitudo"])
        self.maximum = max(self.values, default=0)
        self.cache = {}
        self.features = features or {}
        self.reused = set()

    def tolerance(self, target):
        return max(self.axis.step, self.pct * target)

    def find(self, target, separation):
        key = (target, separation)
        if key not in self.cache:
            self.cache[key] = self._find(target, separation)
        return dict(self.cache[key])

    def _find(self, target, separation):
        a, v = self.axis, self.values
        tol = self.tolerance(target)
        result = {"cel_hz": target, "tures_hz": tol, "allapot": "NINCS_KIEMELKEDO_CSUCS"}
        if separation <= 0 or 2 * tol >= separation:
            return dict(result, allapot="NEM_FELBONTHATO")
        if target <= 0 or target - tol < a.start or target + tol > a.end:
            return dict(result, allapot="TARTOMANYON_KIVUL")
        lo = max(1, math.ceil((target - tol - a.start) / a.step - 1e-9))
        hi = min(len(v) - 2, math.floor((target + tol - a.start) / a.step + 1e-9))
        candidates = [i for i in range(lo, hi + 1)
                      if v[i] > v[i - 1] and v[i] >= v[i + 1]]
        for index in sorted(candidates, key=lambda i: v[i], reverse=True):
            width = max(1, math.ceil(tol / a.step))
            radius = max(8, 3 * width)
            background = [v[j] for j in range(max(0, index - radius), min(len(v), index + radius + 1))
                          if abs(j - index) > width]
            if len(background) < 4:
                continue
            floor = statistics.median(background)
            amplitude = v[index]
            if amplitude > 0 and amplitude >= self.relative * self.maximum and amplitude >= self.ratio * floor:
                return dict(result, allapot="TALALAT", index=index,
                            csucs_hz=a.frequency(index), amplitudo=amplitude,
                            helyi_hatter=floor,
                            hatter_arany=amplitude / floor if floor > 0 else None)
        return result

    def band_max(self, target):
        a = self.axis
        tol = self.tolerance(target)
        if target - tol < a.start or target + tol > a.end:
            return None
        # A korábbi sávmaximum akkor használható fel, ha a szűkebb sávba is beleesik.
        # Az indexet és az amplitúdót az eredeti spektrumon ellenőrzöm.
        for k in (1, 2, 3):
            f = self.features
            try:
                center = float(f[f"harm_{k}_cel_hz"])
                previous_tol = float(f[f"harm_{k}_tures_hz"])
                index = int(f[f"harm_{k}_index"])
                amplitude = float(f[f"harm_{k}_amplitudo"])
                if (f.get(f"harm_{k}_allapot") == "savmaximum_kiszamitva" and
                        abs(center - target) < max(a.step * 1e-9, 1e-12) and previous_tol >= tol and
                        0 <= index < len(self.values) and abs(a.frequency(index) - target) <= tol + a.step * 1e-9 and
                        math.isfinite(amplitude) and abs(self.values[index] - amplitude) <= max(abs(amplitude) * 1e-12, 1e-15)):
                    self.reused.add(k)
                    return amplitude
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        lo = max(0, math.ceil((target - tol - a.start) / a.step - 1e-9))
        hi = min(len(self.values) - 1, math.floor((target + tol - a.start) / a.step + 1e-9))
        return max(self.values[lo:hi + 1], default=None)


def is_hit(peak):
    return peak.get("allapot") == "TALALAT"


def distinct_hits(peaks):
    return list({p["index"]: p for p in peaks if is_hit(p)}.values())


def band_description(record, axis, low, high):
    if axis.start > low or axis.end < high:
        return {"allapot": "TARTOMANYON_KIVUL", "sav_hz": [low, high]}
    values = [x for i, x in enumerate(record.values) if low <= axis.frequency(i) <= high]
    if not values:
        return {"allapot": "NEM_FELBONTHATO", "sav_hz": [low, high]}
    return {"allapot": "LEIRO_ADAT", "sav_hz": [low, high], "median_amplitudo": statistics.median(values),
            "maximum_amplitudo": max(values), "pontszam": len(values)}
