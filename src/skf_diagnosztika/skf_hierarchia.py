# A Hierarchy export pontleírásaiból azonosítja a mérés irányát.

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .skf_fajlkezeles import decode, table


def folded(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def direction(description):
    words = set(re.findall(r"[a-z]+", folded(description)))
    # A mérési irányt a pont leírásából veszem.
    if words & {"pump", "motor", "szivattyu", "shaft", "tengely"} and not words & {
            "meres", "meresi", "measurement", "direction", "irany", "sensor", "ho", "ve", "hm", "hi", "ax"}:
        return None
    if words & {"nem", "not", "no"}:
        return None
    aliases = {"ho": {"ho", "horizontal", "vizszintes"}, "ve": {"ve", "vertical", "fuggoleges"},
               "hi": {"hi"}, "hm": {"hm"}, "ax": {"ax", "axial", "axialis", "tengelyiranyu"}}
    found = [code for code, variants in aliases.items() if words & variants]
    radial = bool(words & {"radial", "radialis"})
    if len(found) == 1 and not (radial and found == ["ax"]):
        return found[0]
    return "radialis" if radial and not found else None


def point_key(path):
    return path.strip().strip("\\").casefold()


def load_folder(folder):
    points = defaultdict(list)
    for path in sorted(folder.iterdir()):
        if not (path.is_file() and path.suffix.lower() == ".csv"):
            continue
        # Az átnevezett hierarchiaexportot a fejléc azonosítja.
        with path.open("rb") as handle:
            bom = handle.read(3)
        encodings = ("utf-16",) if bom.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "cp1250")
        header = []
        for encoding in encodings:
            try:
                with path.open(encoding=encoding, errors="strict") as handle:
                    first = next((line for line in handle if line.strip()), "")
                if first:
                    header = [folded(s.strip()) for s in table(first)[0]]
                break
            except (UnicodeError, csv.Error):
                continue
        def col(options):
            return next((i for i, h in enumerate(header) if h in options), None)
        route = col({"full path", "path", "point path", "node path", "hierarchy path"})
        desc = col({"description", "point description", "node description"})
        kind = col({"type", "node type", "item type"})
        if route is None or desc is None:
            continue
        rows = table(decode(path)[0])
        for row in rows[1:]:
            if len(row) <= max(route, desc, kind or 0):
                continue
            if kind is not None and row[kind].strip().casefold() != "point":
                continue
            if row[route].strip() and row[desc].strip():
                points[point_key(row[route])].append((path.name, row[desc].strip()))
    return points


def lookup(record, cache):
    entries = set()
    for source in record.sources:
        folder = Path(source["fajl"]).resolve().parent
        if folder not in cache:
            cache[folder] = load_folder(folder) if folder.is_dir() else {}
        entries.update(cache[folder].get(point_key(record.point), []))
    directions = {direction(desc) for _, desc in entries}
    selected = next(iter(directions)) if len(directions) == 1 else None
    return selected, sorted(entries)
