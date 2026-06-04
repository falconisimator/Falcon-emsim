# emsim — 2D Magnetoquasistatic Busbar EM Simulator

From-scratch 2D MQS finite-element solver (complex phasor, single frequency,
linear materials) for busbar systems. Computes current sharing, AC losses
(R_AC/R_DC), EMI/shielding and Maxwell-stress forces.

All quantities are SI and **per unit length** in z (Ω/m, W/m, N/m); a group's
prescribed current is the total current through its cross section.

## Physics

Complex magnetic vector potential `A_z(x,y)`, phasor convention `e^{+jωt}`:

    −∇·((1/µ)∇A_z) + jωσ A_z = σ (V̇/L)     (conductors)

Each parallel group (electrical terminal) adds a bordered unknown `u = V̇/L`
and a prescribed total current, solved as the arrowhead system

    [ S   −B ] [a]   [0    ]        S = K + jωM   (complex symmetric)
    [ C    I ] [u] = [I/g_g]        C = −(jω/g_g) Bᵀ

## Layout

```
emsim/
  config.py        SimulationConfig (frequency, ω, skin depth)
  materials.py     Material(σ, µ_r), MaterialTable, COPPER/ALUMINIUM/AIR/STEEL
  mesh/            Mesh (pure arrays); gmsh_backend (only place importing gmsh)
  fem/             elements (P1 kernels + quadrature), constraints (ParallelGroup),
                   assembly (bordered system + Dirichlet gauge pin)
  solve/           linear (sparse LU), solver (orchestration → Solution)
  post/            fields (A_z/B/J), losses (R_AC/R_DC, ohmic loss)
  analytic/        round_wire (Bessel + Kelvin skin-effect oracle)
  results.py       Solution dataclass
  plotting.py      |A|,|B|,|J| maps; sweep plots
validation/        per-milestone validation tests + figures
```

## Status

- **M1 ✓** Round-wire skin effect vs analytic Bessel: R_AC/R_DC error
  0.00 %→1.33 % across δ/a ∈ [10, 0.1] on a skin-graded mesh.
- **M2 ✓** Current sharing: symmetric 50/50 split; composite Cu/Al single
  terminal DC split 0.481 vs 0.482 analytic; independent go/return terminals.
- **M3 ✓** Kelvin open boundary: balanced-pair loop inductance within 0.3 % of
  analytic, and ~3× more accurate than a Dirichlet box at the same radius.
  *Note:* in 2D the Kelvin inversion is conformal, so the exterior maps to a
  standard constant-ν disk problem — no radius-dependent coefficient is needed
  (that term only arises in 3D/axisymmetric). See `emsim/mesh/kelvin.py`.

- **M4 ✓** Enclosure/EMI/forces: enclosure eddy loss exact by energy balance
  (input power = total ohmic loss, 0.000 %); two-wire Maxwell-stress force
  within 1.9 % of μ₀I₁I₂/(4πD); shielding effectiveness 35→175 dB rising with
  frequency (magnetostatic → eddy shielding).

- **M5 ✓** Adaptive ZZ refinement: from a coarse mesh the estimator drives the
  strong-skin (δ/a=0.1) R_AC error 9.1 %→0.35 %, global error estimate
  decreasing monotonically, refinement concentrated in the skin layer.

- **M6 ✓** Second-order (P2) elements through the same quadrature kernel:
  more accurate than P1 at equal element size (1.2 % vs 7.3 % at lc=a/2) and
  higher convergence order (1.65 vs 1.40); P1 results unchanged (regression).
- **M7 ✓** PySide6 GUI: interactive canvas to place/move/edit rectangular and
  round conductors (+enclosure), material/group/frequency editors, threaded
  Solve, and result views (|A|/|B|/|J| maps, per-bar current/loss/force table,
  R_AC summary). Headless smoke-tested.

**All 7 milestones complete — 18/18 validation tests passing.**

## Run

```
pip install -e .            # numpy, scipy, matplotlib, gmsh
pip install -e .[gui]       # + PySide6 for the GUI
python -m pytest validation/                  # all gates (18 tests)
python -m validation.test_round_wire          # M1 sweep + figure
python -m emsim.gui                            # launch the interactive GUI
```
