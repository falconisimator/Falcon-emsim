# emsim — web build (GitHub Pages)

This folder is a **static, client-side** build of the busbar simulator. The
entire 2D magnetoquasistatic FEM solve runs in the browser via
[Pyodide](https://pyodide.org) (CPython + NumPy + SciPy compiled to
WebAssembly). There is **no backend** — all compute happens on the user's
machine, so hosting is just static file serving.

## Files
- `index.html` — page + UI.
- `main.js` — boots Pyodide, installs the `emsim` wheel, runs `Scene.solve()`
  with the gmsh-free mesher (`mesh_backend="py"`), and renders / animates the
  field on a `<canvas>` (instantaneous fields computed in JS as `Re(F·e^{jφ})`).
- `wheels/emsim-*.whl` — the pure-Python package, loaded with `micropip`
  (`deps=False`; only `numpy`/`scipy` are needed and are preloaded by Pyodide).

## Deploy to GitHub Pages
1. Push the repo to GitHub.
2. Repo **Settings → Pages → Build and deployment**: Source = *Deploy from a
   branch*, Branch = `main`, Folder = `/docs`.
3. The app will be live at `https://<user>.github.io/<repo>/`.

## Rebuild after changing Python source
```
python scripts/build_web.py      # rebuilds docs/wheels/emsim-*.whl
```
Then commit the new wheel and push.

## Local preview
```
python -m http.server -d docs 8123    # then open http://localhost:8123
```

## Notes / limitations
- WASM SciPy is ~2–5× slower than native; the gmsh-free mesher and the
  Dirichlet open-boundary keep meshes browser-friendly. Kelvin open boundary and
  P2 elements remain desktop-only for now.
- First load downloads Pyodide + SciPy (~20 MB, then cached).
