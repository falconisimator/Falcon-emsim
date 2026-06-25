// gif.js — tiny self-contained animated GIF89a encoder (no deps, works offline).
// window.GIFENC.encode(frames, opts) -> Uint8Array, where `frames` is an array of
// RGBA pixel buffers (Uint8ClampedArray, all width*height*4). A single global
// 256-colour palette is built (median-cut) from a sample of every frame, then
// each frame is LZW-compressed. Good enough for the field/thermal views, whose
// colours are a smooth colormap that barely changes frame to frame.

(function () {
  function buildPalette(frames, w, h, maxColors) {
    // sample ~16k pixels across all frames for the colour cube
    const total = frames.length * w * h;
    const stride = Math.max(1, Math.floor(total / 16384));
    const samp = [];
    let g = 0;
    for (const f of frames) {
      for (let p = 0; p < w * h; p++, g++) {
        if (g % stride) continue;
        const i = p * 4;
        samp.push([f[i], f[i + 1], f[i + 2]]);
      }
    }
    if (!samp.length) samp.push([0, 0, 0]);
    // median cut
    let boxes = [samp];
    while (boxes.length < maxColors) {
      let bi = -1, brange = -1, bch = 0;
      for (let i = 0; i < boxes.length; i++) {
        const b = boxes[i];
        if (b.length < 2) continue;
        const mn = [255, 255, 255], mx = [0, 0, 0];
        for (const px of b) for (let k = 0; k < 3; k++) { if (px[k] < mn[k]) mn[k] = px[k]; if (px[k] > mx[k]) mx[k] = px[k]; }
        const r = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
        const ch = r[0] >= r[1] && r[0] >= r[2] ? 0 : (r[1] >= r[2] ? 1 : 2);
        if (r[ch] > brange) { brange = r[ch]; bi = i; bch = ch; }
      }
      if (bi < 0) break;                      // nothing left to split
      const box = boxes[bi];
      box.sort((a, b) => a[bch] - b[bch]);
      const mid = box.length >> 1;
      boxes.splice(bi, 1, box.slice(0, mid), box.slice(mid));
    }
    const pal = boxes.map((b) => {
      let r = 0, gg = 0, bl = 0;
      for (const px of b) { r += px[0]; gg += px[1]; bl += px[2]; }
      const n = b.length || 1;
      return [Math.round(r / n), Math.round(gg / n), Math.round(bl / n)];
    });
    while (pal.length < 256) pal.push([0, 0, 0]);
    return pal.slice(0, 256);
  }

  // map a frame's RGBA to palette indices, with a 15-bit colour cache for speed
  function mapFrame(f, w, h, pal, cache) {
    const out = new Uint8Array(w * h);
    for (let p = 0; p < w * h; p++) {
      const i = p * 4, r = f[i], gg = f[i + 1], b = f[i + 2];
      const key = ((r >> 3) << 10) | ((gg >> 3) << 5) | (b >> 3);
      let idx = cache[key];
      if (idx < 0) {
        let best = 0, bd = 1e9;
        for (let c = 0; c < pal.length; c++) {
          const dr = r - pal[c][0], dg = gg - pal[c][1], db = b - pal[c][2];
          const d = dr * dr + dg * dg + db * db;
          if (d < bd) { bd = d; best = c; }
        }
        idx = cache[key] = best;
      }
      out[p] = idx;
    }
    return out;
  }

  // GIF LZW (variable code width, LSB-first), returns a byte array (no sub-blocking)
  function lzw(indices, minCode) {
    const clear = 1 << minCode, eoi = clear + 1;
    const out = [];
    let cur = 0, bits = 0, codeSize = minCode + 1, next, dict;
    const emit = (code) => { cur |= code << bits; bits += codeSize; while (bits >= 8) { out.push(cur & 255); cur >>= 8; bits -= 8; } };
    const reset = () => { dict = new Map(); for (let i = 0; i < clear; i++) dict.set("" + i, i); next = eoi + 1; codeSize = minCode + 1; };
    reset();
    emit(clear);
    let prefix = "" + indices[0];
    for (let i = 1; i < indices.length; i++) {
      const k = indices[i], key = prefix + "," + k;
      if (dict.has(key)) { prefix = key; continue; }
      emit(dict.get(prefix));
      dict.set(key, next++);
      if (next === (1 << codeSize) && codeSize < 12) codeSize++;
      else if (next === 4096) { emit(clear); reset(); }
      prefix = "" + k;
    }
    emit(dict.get(prefix));
    emit(eoi);
    if (bits > 0) out.push(cur & 255);
    return out;
  }

  function GifWriter() {
    const b = [];
    return {
      byte: (v) => b.push(v & 255),
      word: (v) => { b.push(v & 255); b.push((v >> 8) & 255); },
      str: (s) => { for (let i = 0; i < s.length; i++) b.push(s.charCodeAt(i)); },
      // LZW byte stream -> 255-byte sub-blocks, 0x00 terminated
      blocks: (data) => {
        for (let i = 0; i < data.length; i += 255) {
          const chunk = data.slice(i, i + 255);
          b.push(chunk.length);
          for (const v of chunk) b.push(v);
        }
        b.push(0);
      },
      bytes: () => Uint8Array.from(b),
    };
  }

  function encode(frames, opts) {
    const w = opts.width, h = opts.height, delay = opts.delay || 5, loop = opts.loop || 0;
    const pal = buildPalette(frames, w, h, 256);
    const cache = new Int16Array(32768).fill(-1);
    const g = GifWriter();
    g.str("GIF89a");
    g.word(w); g.word(h);
    g.byte(0xF7);            // GCT present, 8-bit colour, 256-entry table
    g.byte(0); g.byte(0);    // bg colour index, pixel aspect ratio
    for (const c of pal) { g.byte(c[0]); g.byte(c[1]); g.byte(c[2]); }
    // Netscape loop extension
    g.byte(0x21); g.byte(0xFF); g.byte(0x0B); g.str("NETSCAPE2.0");
    g.byte(0x03); g.byte(0x01); g.word(loop); g.byte(0x00);
    for (const f of frames) {
      g.byte(0x21); g.byte(0xF9); g.byte(0x04);
      g.byte(0x04);          // disposal = do not dispose, no transparency
      g.word(delay); g.byte(0x00); g.byte(0x00);
      g.byte(0x2C);          // image descriptor
      g.word(0); g.word(0); g.word(w); g.word(h); g.byte(0x00);
      g.byte(8);             // LZW minimum code size
      g.blocks(lzw(mapFrame(f, w, h, pal, cache), 8));
    }
    g.byte(0x3B);            // trailer
    return g.bytes();
  }

  function download(bytes, name) {
    const url = URL.createObjectURL(new Blob([bytes], { type: "image/gif" }));
    const a = document.createElement("a");
    a.href = url; a.download = name || "export.gif"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  window.GIFENC = { encode, download };
})();
