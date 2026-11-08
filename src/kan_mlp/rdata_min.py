"""Minimal reader for R .rda/.rds files (XDR serialization v2/v3).

Supports the object types that appear in plain data frames: atomic vectors,
lists, pairlists, symbols, attributes and the compact/wrapper ALTREP forms.
Written because R and pyreadr are unavailable in the build environment.
"""
import bz2, gzip, lzma, struct

NA_INT = -2147483648

class RObj:
    def __init__(self, kind, value, attrs=None):
        self.kind, self.value, self.attrs = kind, value, attrs or {}
    def attr(self, name, default=None):
        return self.attrs.get(name, default)
    def __repr__(self):
        return f"RObj({self.kind}, attrs={list(self.attrs)})"

def _decompress(raw):
    if raw[:2] == b"\x1f\x8b": return gzip.decompress(raw)
    if raw[:3] == b"BZh": return bz2.decompress(raw)
    if raw[:5] == b"\xfd7zXZ": return lzma.decompress(raw)
    return raw

class _Reader:
    def __init__(self, buf):
        self.b, self.p, self.refs = buf, 0, []
    def i(self):
        v = struct.unpack(">i", self.b[self.p:self.p+4])[0]; self.p += 4; return v
    def n(self, k, fmt, size):
        v = struct.unpack(f">{k}{fmt}", self.b[self.p:self.p+k*size]); self.p += k*size; return list(v)
    def length(self):
        n = self.i()
        if n == -1:
            hi, lo = self.i(), self.i(); n = (hi << 32) + lo
        return n
    def pairlist_attrs(self):
        out = {}
        node = self.item()
        while isinstance(node, RObj) and node.kind == "pairlist":
            for tag, val in node.value: out[tag] = val
            break
        return out
    def item(self):
        flags = self.i()
        t = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == 254: return None                      # NILVALUE
        if t in (253, 242, 241, 250, 252, 251): return RObj("env", None)
        if t == 255:                                  # REFSXP
            idx = flags >> 8
            if idx == 0: idx = self.i()
            return self.refs[idx-1]
        if t in (247, 249, 248):                      # persist/namespace/package
            self.i(); n = self.i(); vals = [self.item() for _ in range(n)]
            obj = RObj("ns", vals); self.refs.append(obj); return obj
        if t == 1:                                    # SYMSXP
            name = self.item(); self.refs.append(name); return name
        if t == 9:                                    # CHARSXP
            n = self.i()
            if n == -1: return None
            s = self.b[self.p:self.p+n].decode("utf-8", "replace"); self.p += n; return s
        if t in (2, 6, 3, 5, 17, 239, 240):           # pairlist-like
            items = []
            while True:
                attrs = self.pairlist_attrs() if has_attr else {}
                tag = self.item() if has_tag else None
                car = self.item()
                items.append((tag, car))
                flags = self.i(); t2 = flags & 0xFF
                if t2 == 254: break
                if t2 not in (2, 6, 3, 5, 17, 239, 240):
                    self.p -= 4; items.append((None, self.item())); break
                has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
            return RObj("pairlist", items)
        if t == 4:                                    # ENVSXP
            obj = RObj("env", None); self.refs.append(obj)
            self.i(); self.item(); self.item(); self.item(); self.item(); return obj
        if t in (10, 13):
            n = self.length(); v = self.n(n, "i", 4)
            v = [None if x == NA_INT else x for x in v]; kind = "logical" if t == 10 else "integer"
        elif t == 14:
            n = self.length(); v = self.n(n, "d", 8); kind = "double"
        elif t == 15:
            n = self.length(); d = self.n(2*n, "d", 8); v = [complex(d[2*k], d[2*k+1]) for k in range(n)]; kind = "complex"
        elif t == 16:
            n = self.length(); v = [self.item() for _ in range(n)]; kind = "character"
        elif t in (19, 20):
            n = self.length(); v = [self.item() for _ in range(n)]; kind = "list"
        elif t == 24:
            n = self.length(); v = self.b[self.p:self.p+n]; self.p += n; kind = "raw"
        elif t == 238:                                # ALTREP
            info = self.item(); state = self.item(); attr = self.item()
            cls = info.value[0][1] if isinstance(info, RObj) else None
            if cls == "compact_intseq":
                n, start, step = (int(x) for x in state.value)
                obj = RObj("integer", list(range(start, start+n*step, step)))
            elif cls == "compact_realseq":
                n, start, step = state.value
                obj = RObj("double", [start+k*step for k in range(int(n))])
            elif cls and cls.startswith("wrap_"):
                obj = state.value[0]
            elif cls == "deferred_string":
                base = state.value[0][1] if state.kind == "pairlist" else state
                obj = RObj("character", [str(x) for x in base.value])
            else:
                raise NotImplementedError(f"ALTREP class {cls}")
            if isinstance(attr, RObj) and attr.kind == "pairlist":
                obj.attrs.update({k: v for k, v in attr.value})
            return obj
        elif t == 25:                                 # S4
            obj = RObj("S4", None)
            if has_attr: obj.attrs = self.pairlist_attrs()
            return obj
        else:
            raise NotImplementedError(f"SEXP type {t} at byte {self.p}")
        obj = RObj(kind, v)
        if has_attr: obj.attrs = self.pairlist_attrs()
        return obj

def read(path):
    raw = _decompress(open(path, "rb").read())
    if raw[:5] in (b"RDX2\n", b"RDX3\n"): raw = raw[5:]
    assert raw[:2] == b"X\n", "only XDR format supported"
    r = _Reader(raw[2:])
    version = r.i(); r.i(); r.i()
    if version == 3:
        n = r.i(); r.p += n
    top = r.item()
    if top is not None and top.kind == "pairlist":
        return {tag: val for tag, val in top.value}
    return top

def as_frame(obj):
    """Convert an R data.frame RObj into a pandas DataFrame."""
    import pandas as pd
    names = obj.attr("names").value
    cols = {}
    for name, col in zip(names, obj.value):
        vals = col.value
        levels = col.attr("levels")
        if levels is not None:
            lv = levels.value; vals = [None if v is None else lv[v-1] for v in vals]
        cols[name] = vals
    return pd.DataFrame(cols)
