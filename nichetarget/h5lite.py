"""Minimal pure-Python (numpy + zlib) HDF5 reader for 10x Genomics feature-barcode matrices.

Implements only the subset of the HDF5 file format needed to read
``filtered_feature_bc_matrix.h5`` / ``raw_feature_bc_matrix.h5`` files:

* superblock v0/v1 and v2/v3
* object headers v1 and v2 (with continuation blocks)
* old-style groups (symbol table message, v1 B-tree type 0, SNOD, local heap)
  and new-style compact groups (link messages); dense link storage
  (fractal heap) is not supported and raises ``NotImplementedError``
* dataspace (simple/scalar), datatype (fixed-point, float, fixed-length string)
* data layout v1-v3 (compact, contiguous, chunked with v1 B-tree index),
  v4 compact/contiguous/single-chunk
* filters: deflate, shuffle, fletcher32 (checksum stripped, not verified)

The file is parsed as untrusted data: every read is bounds-checked and
nothing from the file is ever executed.

Usage::

    from nichetarget.h5lite import read_10x_h5
    m = read_10x_h5("sample_filtered_feature_bc_matrix.h5")
    m["data"], m["indices"], m["indptr"], m["shape"], m["barcodes"], ...
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

__all__ = ["H5File", "read_10x_h5"]

_SIG = b"\x89HDF\r\n\x1a\n"

# message types
_MSG_DATASPACE = 0x0001
_MSG_LINKINFO = 0x0002
_MSG_DATATYPE = 0x0003
_MSG_FILLVALUE = 0x0005
_MSG_LINK = 0x0006
_MSG_LAYOUT = 0x0008
_MSG_FILTER = 0x000B
_MSG_CONT = 0x0010
_MSG_SYMTAB = 0x0011

_FILTER_DEFLATE = 1
_FILTER_SHUFFLE = 2
_FILTER_FLETCHER32 = 3

_MAX_CHUNK_BYTES = 1 << 31  # sanity cap for a single decompressed chunk


class H5Error(ValueError):
    pass


class _Dataset:
    def __init__(self, f, shape, dtype, layout, filters):
        self._f = f
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.filters = filters

    def read(self):
        return self._f._read_dataset(self)


class H5File:
    """Read-only, minimal HDF5 file. Use as ``H5File(path)['matrix/data'][()]``-like via ``read``."""

    def __init__(self, path):
        with open(path, "rb") as fh:
            self.buf = memoryview(fh.read())
        self.features = set()  # names of HDF5 features encountered (for diagnostics)
        self._parse_superblock()

    # ------------------------------------------------------------------ helpers
    def _bytes(self, off, n):
        if off < 0 or n < 0 or off + n > len(self.buf):
            raise H5Error(f"read out of bounds: offset {off} len {n} (file {len(self.buf)})")
        return self.buf[off:off + n]

    def _uint(self, off, n):
        return int.from_bytes(self._bytes(off, n), "little")

    def _undef(self, addr, n=None):
        n = n or self.so
        return addr == (1 << (8 * n)) - 1

    def _addr(self, a):
        return a + self.base

    # --------------------------------------------------------------- superblock
    def _parse_superblock(self):
        # signature may be at 0, 512, 1024, ...
        off = 0
        while True:
            if off + 8 > len(self.buf):
                raise H5Error("HDF5 signature not found")
            if bytes(self.buf[off:off + 8]) == _SIG:
                break
            off = 512 if off == 0 else off * 2
        ver = self._uint(off + 8, 1)
        self.sb_version = ver
        self.features.add(f"superblock v{ver}")
        if ver in (0, 1):
            self.so = self._uint(off + 13, 1)
            self.sl = self._uint(off + 14, 1)
            p = off + 24 + (4 if ver == 1 else 0)
            so = self.so
            self.base = self._uint(p, so)
            p += 4 * so  # base, free-space, EOF, driver info
            # root group symbol table entry
            root_oh = self._uint(p + so, so)
        elif ver in (2, 3):
            self.so = self._uint(off + 9, 1)
            self.sl = self._uint(off + 10, 1)
            p = off + 12
            so = self.so
            self.base = self._uint(p, so)
            root_oh = self._uint(p + 3 * so, so)
        else:
            raise NotImplementedError(f"superblock version {ver}")
        if self.so not in (2, 4, 8) or self.sl not in (2, 4, 8):
            raise H5Error("bad sizes of offsets/lengths")
        self.root = root_oh

    # ------------------------------------------------------------ object header
    def _messages(self, addr):
        """Return list of (type, flags, memoryview data) for object header at addr."""
        a = self._addr(addr)
        msgs = []
        if bytes(self._bytes(a, 4)) == b"OHDR":
            self.features.add("object header v2")
            ver = self._uint(a + 4, 1)
            if ver != 2:
                raise H5Error(f"bad OHDR version {ver}")
            flags = self._uint(a + 5, 1)
            p = a + 6
            if flags & 0x20:
                p += 16
            if flags & 0x10:
                p += 4
            csize_len = 1 << (flags & 3)
            size0 = self._uint(p, csize_len)
            p += csize_len
            blocks = [(p, size0 - 4)]  # minus checksum (gap handled by bound)
            track_order = bool(flags & 0x04)
            i = 0
            while i < len(blocks):
                start, length = blocks[i]
                i += 1
                q, end = start, start + length
                hdr = 4 + (2 if track_order else 0)
                while q + hdr <= end:
                    mtype = self._uint(q, 1)
                    msize = self._uint(q + 1, 2)
                    mflags = self._uint(q + 3, 1)
                    q += hdr
                    data = self._bytes(q, msize)
                    q += msize
                    if mtype == _MSG_CONT:
                        caddr = int.from_bytes(data[:self.so], "little")
                        clen = int.from_bytes(data[self.so:self.so + self.sl], "little")
                        ca = self._addr(caddr)
                        if bytes(self._bytes(ca, 4)) != b"OCHK":
                            raise H5Error("bad OCHK signature")
                        blocks.append((ca + 4, clen - 8))
                    else:
                        msgs.append((mtype, mflags, data))
                if len(blocks) > 10000:
                    raise H5Error("too many continuation blocks")
        else:
            ver = self._uint(a, 1)
            if ver != 1:
                raise H5Error(f"unsupported object header version {ver} at {addr}")
            self.features.add("object header v1")
            nmsgs = self._uint(a + 2, 2)
            hsize = self._uint(a + 8, 4)
            blocks = [(a + 16, hsize)]
            i = 0
            count = 0
            while i < len(blocks) and count < nmsgs:
                start, length = blocks[i]
                i += 1
                q, end = start, start + length
                while q + 8 <= end and count < nmsgs:
                    mtype = self._uint(q, 2)
                    msize = self._uint(q + 2, 2)
                    mflags = self._uint(q + 4, 1)
                    q += 8
                    data = self._bytes(q, msize)
                    q += msize
                    count += 1
                    if mtype == _MSG_CONT:
                        caddr = int.from_bytes(data[:self.so], "little")
                        clen = int.from_bytes(data[self.so:self.so + self.sl], "little")
                        blocks.append((self._addr(caddr), clen))
                    else:
                        msgs.append((mtype, mflags, data))
                if len(blocks) > 10000:
                    raise H5Error("too many continuation blocks")
        return msgs

    # ------------------------------------------------------------------- groups
    def _group_links(self, addr):
        """Return dict name -> object header address for a group object header."""
        msgs = self._messages(addr)
        links = {}
        for mtype, mflags, data in msgs:
            if mtype == _MSG_SYMTAB:
                self.features.add("symbol-table group (v1 B-tree + local heap)")
                btree = int.from_bytes(data[:self.so], "little")
                heap = int.from_bytes(data[self.so:2 * self.so], "little")
                links.update(self._symtab_links(btree, heap))
            elif mtype == _MSG_LINK:
                self.features.add("link messages (compact group)")
                name, target = self._parse_link(data)
                if target is not None:
                    links[name] = target
            elif mtype == _MSG_LINKINFO:
                fheap = int.from_bytes(data[2 + (8 if data[1] & 1 else 0):][:self.so], "little")
                if not self._undef(fheap):
                    raise NotImplementedError("dense link storage (fractal heap) not supported")
        return links

    def _parse_link(self, d):
        ver, flags = d[0], d[1]
        if ver != 1:
            raise H5Error(f"link message version {ver}")
        p = 2
        ltype = 0
        if flags & 0x08:
            ltype = d[p]; p += 1
        if flags & 0x04:
            p += 8
        if flags & 0x10:
            p += 1
        nlen_size = 1 << (flags & 3)
        nlen = int.from_bytes(d[p:p + nlen_size], "little"); p += nlen_size
        name = bytes(d[p:p + nlen]).decode("utf-8", "replace"); p += nlen
        if ltype == 0:
            return name, int.from_bytes(d[p:p + self.so], "little")
        return name, None  # soft/external links ignored

    def _local_heap(self, heap_addr):
        a = self._addr(heap_addr)
        if bytes(self._bytes(a, 4)) != b"HEAP":
            raise H5Error("bad local heap signature")
        size = self._uint(a + 8, self.sl)
        data_addr = self._uint(a + 8 + 2 * self.sl, self.so)
        return self._bytes(self._addr(data_addr), size)

    @staticmethod
    def _cstr(heap, off):
        if off >= len(heap):
            raise H5Error("heap offset out of range")
        b = bytes(heap[off:])
        return b[:b.find(b"\0") if b"\0" in b else len(b)].decode("utf-8", "replace")

    def _symtab_links(self, btree, heap_addr):
        heap = self._local_heap(heap_addr)
        links = {}
        so = self.so
        stack, seen = [btree], set()
        while stack:
            node = stack.pop()
            if node in seen:
                raise H5Error("B-tree cycle")
            seen.add(node)
            a = self._addr(node)
            sig = bytes(self._bytes(a, 4))
            if sig == b"TREE":
                ntype, level, used = self._uint(a + 4, 1), self._uint(a + 5, 1), self._uint(a + 6, 2)
                if ntype != 0:
                    raise H5Error("expected group B-tree node")
                p = a + 8 + 2 * so + self.sl  # skip siblings, key0
                for _ in range(used):
                    stack.append(self._uint(p, so))
                    p += so + self.sl
            elif sig == b"SNOD":
                n = self._uint(a + 6, 2)
                p = a + 8
                for _ in range(n):
                    name_off = self._uint(p, so)
                    oh = self._uint(p + so, so)
                    links[self._cstr(heap, name_off)] = oh
                    p += 2 * so + 24
            else:
                raise H5Error(f"unexpected node signature {sig!r}")
        return links

    def _resolve(self, path):
        addr = self.root
        for part in [p for p in path.split("/") if p]:
            links = self._group_links(addr)
            if part not in links:
                raise KeyError(f"{part!r} not found while resolving {path!r}")
            addr = links[part]
        return addr

    def keys(self, path="/"):
        return sorted(self._group_links(self._resolve(path)))

    # ------------------------------------------------------------------ datasets
    def dataset(self, path):
        msgs = self._messages(self._resolve(path))
        shape = dtype = layout = None
        filters = []
        for mtype, mflags, d in msgs:
            if mtype in (_MSG_DATATYPE,) and mflags & 0x02:
                raise NotImplementedError("shared/committed datatype")
            if mtype == _MSG_DATASPACE:
                shape = self._parse_dataspace(d)
            elif mtype == _MSG_DATATYPE:
                dtype = self._parse_datatype(d)
            elif mtype == _MSG_LAYOUT:
                layout = self._parse_layout(d)
            elif mtype == _MSG_FILTER:
                filters = self._parse_filters(d)
        if shape is None or dtype is None or layout is None:
            raise H5Error(f"{path!r} is not a dataset")
        return _Dataset(self, shape, dtype, layout, filters)

    def read(self, path):
        return self.dataset(path).read()

    def _parse_dataspace(self, d):
        ver = d[0]
        rank = d[1]
        flags = d[2]
        if ver == 1:
            p = 8
            self.features.add("dataspace v1")
        elif ver == 2:
            p = 4
            self.features.add("dataspace v2")
            if d[3] == 2:
                return None  # null dataspace
        else:
            raise H5Error(f"dataspace version {ver}")
        dims = tuple(int.from_bytes(d[p + i * self.sl:p + (i + 1) * self.sl], "little") for i in range(rank))
        return dims

    def _parse_datatype(self, d):
        cls = d[0] & 0x0F
        bits = d[1] | (d[2] << 8) | (d[3] << 16)
        size = int.from_bytes(d[4:8], "little")
        if cls == 0:
            order = ">" if bits & 1 else "<"
            signed = bool(bits & 0x08)
            if size not in (1, 2, 4, 8):
                raise NotImplementedError(f"integer size {size}")
            self.features.add(f"datatype int{'' if signed else 'u'}{8 * size}")
            return np.dtype(f"{order}{'i' if signed else 'u'}{size}")
        if cls == 1:
            order = ">" if bits & 1 else "<"
            if size not in (2, 4, 8):
                raise NotImplementedError(f"float size {size}")
            self.features.add(f"datatype float{8 * size}")
            return np.dtype(f"{order}f{size}")
        if cls == 3:
            self.features.add(f"datatype fixed-length string (pad={bits & 0xF}, cset={(bits >> 4) & 0xF})")
            return np.dtype(f"S{size}")
        if cls == 9:
            raise NotImplementedError("variable-length datatype")
        raise NotImplementedError(f"datatype class {cls}")

    def _parse_layout(self, d):
        ver = d[0]
        so = self.so
        if ver in (1, 2):
            ndim, cls = d[1], d[2]
            p = 8
            addr = None
            if cls != 0:
                addr = int.from_bytes(d[p:p + so], "little"); p += so
            dims = [int.from_bytes(d[p + 4 * i:p + 4 * i + 4], "little") for i in range(ndim)]
            p += 4 * ndim
            if cls == 2:
                p += 4  # element size
            name = {0: "compact", 1: "contiguous", 2: "chunked"}[cls]
            self.features.add(f"layout v{ver} {name}")
            if cls == 0:
                n = int.from_bytes(d[p:p + 4], "little")
                return ("compact", bytes(d[p + 4:p + 4 + n]))
            if cls == 1:
                return ("contiguous", addr, None)
            return ("chunked_btree", addr, tuple(dims[:-1]) if len(dims) > 1 else tuple(dims))
        if ver == 3:
            cls = d[1]
            if cls == 0:
                self.features.add("layout v3 compact")
                n = int.from_bytes(d[2:4], "little")
                return ("compact", bytes(d[4:4 + n]))
            if cls == 1:
                self.features.add("layout v3 contiguous")
                addr = int.from_bytes(d[2:2 + so], "little")
                size = int.from_bytes(d[2 + so:2 + so + self.sl], "little")
                return ("contiguous", addr, size)
            if cls == 2:
                self.features.add("layout v3 chunked (v1 B-tree index)")
                ndim = d[2]
                addr = int.from_bytes(d[3:3 + so], "little")
                p = 3 + so
                dims = [int.from_bytes(d[p + 4 * i:p + 4 * i + 4], "little") for i in range(ndim)]
                return ("chunked_btree", addr, tuple(dims[:-1]))
            raise NotImplementedError(f"layout class {cls}")
        if ver == 4:
            cls = d[1]
            if cls == 0:
                self.features.add("layout v4 compact")
                n = int.from_bytes(d[2:4], "little")
                return ("compact", bytes(d[4:4 + n]))
            if cls == 1:
                self.features.add("layout v4 contiguous")
                addr = int.from_bytes(d[2:2 + so], "little")
                size = int.from_bytes(d[2 + so:2 + so + self.sl], "little")
                return ("contiguous", addr, size)
            if cls == 2:
                flags, ndim, enc = d[2], d[3], d[4]
                p = 5
                dims = [int.from_bytes(d[p + enc * i:p + enc * (i + 1)], "little") for i in range(ndim)]
                p += enc * ndim
                itype = d[p]; p += 1
                if itype == 1:
                    self.features.add("layout v4 chunked (single chunk)")
                    fsize = mask = None
                    if flags & 0x02:
                        fsize = int.from_bytes(d[p:p + self.sl], "little"); p += self.sl
                        mask = int.from_bytes(d[p:p + 4], "little"); p += 4
                    addr = int.from_bytes(d[p:p + so], "little")
                    return ("single_chunk", addr, tuple(dims[:-1]), fsize, mask)
                raise NotImplementedError(f"layout v4 chunk index type {itype}")
            raise NotImplementedError(f"layout v4 class {cls}")
        raise NotImplementedError(f"layout version {ver}")

    def _parse_filters(self, d):
        ver, n = d[0], d[1]
        filters = []
        p = 8 if ver == 1 else 2
        for _ in range(n):
            fid = int.from_bytes(d[p:p + 2], "little"); p += 2
            nlen = 0
            if ver == 1 or fid >= 256:
                nlen = int.from_bytes(d[p:p + 2], "little"); p += 2
            p += 2  # flags
            nvals = int.from_bytes(d[p:p + 2], "little"); p += 2
            if ver == 1:
                nlen = (nlen + 7) & ~7
            p += nlen
            vals = [int.from_bytes(d[p + 4 * i:p + 4 * i + 4], "little") for i in range(nvals)]
            p += 4 * nvals
            if ver == 1 and nvals % 2:
                p += 4
            name = {1: "deflate", 2: "shuffle", 3: "fletcher32"}.get(fid, f"filter{fid}")
            self.features.add(f"filter {name}" + (f"(level {vals[0]})" if fid == 1 and vals else ""))
            filters.append((fid, vals))
        return filters

    # ------------------------------------------------------------- data reading
    def _unfilter(self, raw, filters, mask, itemsize):
        b = bytes(raw)
        for i in range(len(filters) - 1, -1, -1):
            if mask & (1 << i):
                continue
            fid, vals = filters[i]
            if fid == _FILTER_DEFLATE:
                dec = zlib.decompressobj()
                b = dec.decompress(b, _MAX_CHUNK_BYTES)
                if dec.unconsumed_tail:
                    raise H5Error("decompressed chunk exceeds sanity limit")
            elif fid == _FILTER_SHUFFLE:
                es = vals[0] if vals else itemsize
                if es > 1:
                    n = len(b) // es
                    head = np.frombuffer(b, np.uint8, n * es).reshape(es, n).T.tobytes()
                    b = head + b[n * es:]
            elif fid == _FILTER_FLETCHER32:
                b = b[:-4]
            else:
                raise NotImplementedError(f"filter id {fid}")
        return b

    def _chunk_records(self, btree, ndim):
        """Yield (size, filter_mask, offsets tuple, addr) from a v1 chunk B-tree."""
        so = self.so
        ksize = 8 + 8 * (ndim + 1)
        stack, seen = [btree], set()
        while stack:
            node = stack.pop()
            if node in seen:
                raise H5Error("B-tree cycle")
            seen.add(node)
            a = self._addr(node)
            if bytes(self._bytes(a, 4)) != b"TREE":
                raise H5Error("bad chunk B-tree signature")
            ntype, level, used = self._uint(a + 4, 1), self._uint(a + 5, 1), self._uint(a + 6, 2)
            if ntype != 1:
                raise H5Error("expected chunk B-tree node")
            p = a + 8 + 2 * so
            for _ in range(used):
                size = self._uint(p, 4)
                mask = self._uint(p + 4, 4)
                offs = tuple(self._uint(p + 8 + 8 * k, 8) for k in range(ndim))
                child = self._uint(p + ksize, so)
                if level == 0:
                    yield size, mask, offs, child
                else:
                    stack.append(child)
                p += ksize + so

    def _read_dataset(self, ds):
        shape, dt, lay = ds.shape, ds.dtype, ds.layout
        if shape is None:
            return np.empty(0, dt)
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = n * dt.itemsize
        if nbytes > len(self.buf) * 1000 + (1 << 20):
            raise H5Error("dataset size implausibly large")
        kind = lay[0]
        if kind == "compact":
            raw = lay[1]
            out = np.frombuffer(raw[:nbytes], dt).copy()
        elif kind == "contiguous":
            addr = lay[1]
            if self._undef(addr):
                out = np.zeros(n, dt)
            else:
                out = np.frombuffer(bytes(self._bytes(self._addr(addr), nbytes)), dt).copy()
        elif kind == "single_chunk":
            _, addr, cdims, fsize, mask = lay
            raw = self._bytes(self._addr(addr), fsize if fsize is not None else nbytes)
            b = self._unfilter(raw, ds.filters, mask or 0, dt.itemsize) if ds.filters else bytes(raw)
            out = np.frombuffer(b[:nbytes], dt).copy()
        elif kind == "chunked_btree":
            _, btree, cdims = lay
            ndim = len(shape) if shape else 1
            shp = shape if shape else (1,)
            if len(cdims) != ndim:
                raise H5Error("chunk rank mismatch")
            out = np.zeros(shp, dt)
            if not self._undef(btree):
                csz = int(np.prod(cdims)) * dt.itemsize
                for size, mask, offs, addr in self._chunk_records(btree, ndim):
                    raw = self._bytes(self._addr(addr), size)
                    b = self._unfilter(raw, ds.filters, mask, dt.itemsize) if ds.filters else bytes(raw)
                    if len(b) < csz:
                        raise H5Error("decoded chunk too short")
                    chunk = np.frombuffer(b[:csz], dt).reshape(cdims)
                    dst, src = [], []
                    for o, c, s in zip(offs, cdims, shp):
                        if o >= s:
                            break
                        e = min(o + c, s)
                        dst.append(slice(o, e))
                        src.append(slice(0, e - o))
                    else:
                        out[tuple(dst)] = chunk[tuple(src)]
            out = out.reshape(-1)
        else:
            raise NotImplementedError(kind)
        return out.reshape(shape) if shape else out.reshape(())


def _strings(arr):
    return [x.rstrip(b"\0").decode("utf-8", "replace") for x in np.asarray(arr).ravel()]


def read_10x_h5(path, return_features_used=False):
    """Read a 10x Genomics (Cell/Space Ranger v3+) feature-barcode matrix .h5 file.

    Returns a dict with the CSC components (genes x barcodes):
    data, indices, indptr (numpy int64 arrays), shape (n_features, n_barcodes),
    barcodes, feature_id, feature_name, feature_type (lists of str).
    """
    f = H5File(path)
    out = {
        "data": f.read("matrix/data"),
        "indices": f.read("matrix/indices").astype(np.int64),
        "indptr": f.read("matrix/indptr").astype(np.int64),
    }
    shape = f.read("matrix/shape")
    out["shape"] = tuple(int(x) for x in np.asarray(shape).ravel()[:2])
    out["barcodes"] = _strings(f.read("matrix/barcodes"))
    out["feature_id"] = _strings(f.read("matrix/features/id"))
    out["feature_name"] = _strings(f.read("matrix/features/name"))
    try:
        out["feature_type"] = _strings(f.read("matrix/features/feature_type"))
    except KeyError:
        out["feature_type"] = ["Gene Expression"] * len(out["feature_id"])
    if out["data"].dtype.kind in "iu":
        out["data"] = out["data"].astype(np.int64)
    nf, nb = out["shape"]
    if len(out["indptr"]) != nb + 1 or len(out["barcodes"]) != nb or len(out["feature_id"]) != nf:
        raise H5Error("inconsistent matrix dimensions")
    if return_features_used:
        out["_features_used"] = sorted(f.features)
    return out
