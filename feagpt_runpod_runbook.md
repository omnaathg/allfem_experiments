# FeaGPT on RunPod — runbook

Companion to `ALLFEM_RUNPOD_RUNBOOK.md`, same repo. Where that runbook tested
a locally-hosted, fine-tuned Ollama model generating raw FEniCS code, this
one tests **FeaGPT** (arXiv:2510.21993, [naividh/FeaGPT](https://github.com/naividh/FeaGPT)) —
a fundamentally different architecture, so read the callout below before you
provision anything.

## How this differs from the ALL-FEM run

| | ALL-FEM | FeaGPT |
|---|---|---|
| LLM | Local, via Ollama (`qwen3-short_think-fenics-local`, etc.) | **Cloud** — Gemini 2.5 Pro, called over the network via `GEMINI_API_KEY` |
| GPU needed? | Yes — inference runs on the pod | **No** — the pod does CAD/mesh/solve (CPU-bound); the LLM call leaves the pod entirely |
| Toolchain | conda `fenics` (legacy dolfin) + `mshr` | FreeCAD (geometry) → Gmsh ≥4.11 (meshing) → CalculiX ≥2.20 (solver) |
| What the model outputs | A full FEniCS Python script, executed as-is | A structured JSON spec (materials/loads/BCs/mesh density) that FeaGPT's own modules turn into a FreeCAD script, a Gmsh mesh, and a CalculiX `.inp` deck |
| Validated by the paper on | — | A cantilever-beam-style prompt (README's own example) and a 432-case parametric wing study; **no compliant-mechanism/self-contact case exists anywhere in the paper or repo** |

Practical consequence: **pick a CPU pod, not a GPU pod, for this run** — you're
paying for GPU-hours you won't use. Only reuse a GPU pod if you're keeping it
warm for the ALL-FEM tests too.

One more difference worth flagging up front: ALL-FEM gave you a real choice
of single- vs. multi-agent execution to test against each other. FeaGPT
doesn't expose that choice — `feagpt run "<prompt>"` is the one documented
execution path (planner → geometry → mesh → solve happens internally, with
no supported way to invoke the stages independently). So both test cases
below are just single runs of that one path, not a single-vs-multi-agent
comparison.

The second test case below is a **compliant forceps-scissors mechanism**,
using the exact geometry you supplied from Libu George & Bharanidaran
(2020), "Design of multifunctional compliant forceps for medical
application" (based on Aguirre & Frecker 2008's parametrization). Good news
relative to a plain pin-jointed scissors: this is a **single monolithic
flexible body** — flexure hinges, not a rigid pin joint — which is much
closer to what FeaGPT's single-solid geometry pipeline is actually built
for (like the wing's Boolean-composed spars/ribs) than a true multi-body
assembly would be. The catch: the source paper's ANSYS model still defines
three **frictionless self-contact regions** (sheath against the flexure
beams) that have no equivalent field in FeaGPT's documented JSON schema
(materials/loads/BCs/mesh/analysis, §II.B) — so treat the geometry/topology
as in-scope but the contact behavior as unvalidated. Section 5 is still an
experiment, just a better-grounded one, with real published numbers to
check the result against instead of a guess.

---

## 0. One-time RunPod console setup

1. Deploy a **CPU pod** (no GPU needed — see table above). Any pod with
   ~4 vCPU / 8GB RAM comfortably handles FreeCAD + Gmsh + CalculiX for
   beam-scale problems.
2. A generic Ubuntu 22.04 template is fine — you don't need the PyTorch
   template this time since there's no local model to load.
3. No need to expose any HTTP ports (no local inference server to reach).
4. Open the pod's Web Terminal.

## 1. Start a persistent tmux session

Same reasoning as the ALL-FEM runbook — installs + a 432-case parametric
study can run long; don't let a dropped connection kill it.

```bash
apt-get update -qq && apt-get install -y tmux
tmux new -s feagpt
```

Check on it later from another shell: `tmux attach -t feagpt`, or
non-interactively: `tmux capture-pane -t feagpt:0 -p | tail -40`.

## 2. Get a Gemini API key

Grab one from [Google AI Studio](https://aistudio.google.com/apikey) before
you start (you said you'll attach it — this is where it plugs in, step 4
below). FeaGPT calls Gemini 2.5 Pro for both analysis planning and
FreeCAD-script generation, so every run needs this key and a live network
path out of the pod.

## 3. Bootstrap: conda env + FreeCAD/Gmsh/CalculiX + FeaGPT

Paste this whole block into the tmux session. It's idempotent — each step
checks whether it's already done, so re-running after a pod restart just
skips ahead.

```bash
cat > ~/bootstrap_feagpt.sh << 'BOOTSTRAP'
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="feagpt"
REPO_DIR="${HOME}/FeaGPT"

echo "=== [1/4] Miniconda ==="
if [ ! -d /opt/conda ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
else
    echo "Miniconda already installed."
fi
source /opt/conda/bin/activate

echo "Accepting Anaconda default-channel ToS (needed on a fresh pod)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo
echo "=== [2/4] FreeCAD + Gmsh + CalculiX (conda-forge, one solve) ==="
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" -c conda-forge \
        python=3.10 freecad "gmsh>=4.11" calculix
else
    echo "Conda env '$ENV_NAME' already exists."
fi
conda activate "$ENV_NAME"

echo
echo "=== [3/4] FeaGPT ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/naividh/FeaGPT.git "$REPO_DIR"
else
    echo "Repo already cloned, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only || true
fi
cd "$REPO_DIR"
pip install -r requirements.txt
pip install -e .

echo
echo "=== [4/4] Sanity checks ==="
echo -n "ccx (CalculiX):   "; command -v ccx || echo "NOT ON PATH"
echo -n "gmsh:             "; python3 -c "import gmsh; print(gmsh.__file__)" 2>/dev/null || echo "NOT IMPORTABLE"
echo -n "FreeCAD:          "; python3 -c "import FreeCAD; print(FreeCAD.__file__)" 2>/dev/null || echo "NOT IMPORTABLE"
echo -n "feagpt CLI:       "; command -v feagpt || echo "NOT ON PATH"

echo
echo "Bootstrap done. Next: export GEMINI_API_KEY, then run a case (see runbook)."
BOOTSTRAP

chmod +x ~/bootstrap_feagpt.sh
~/bootstrap_feagpt.sh 2>&1 | tee ~/feagpt_install.log
```

If any of the four sanity checks in step [4/4] fail, stop and fix it before
moving on — see Troubleshooting (§6). The most common miss is `ccx` not on
`PATH` because it landed in the conda env's `bin/` but you're in a shell
that hasn't activated the env; always `conda activate feagpt` in new shells
before running anything below.

Then, every new shell/tmux window:

```bash
source /opt/conda/bin/activate feagpt
export GEMINI_API_KEY="paste-your-key-here"
cd ~/FeaGPT
```

---

## 4. Test case 1 — cantilever beam

This is the direct analog of the ALL-FEM beam test, and conveniently it's
**also the README's own worked example** — closest thing FeaGPT has to a
known-good smoke test.

Prompt (steel, 500mm square-section cantilever, 1000N point load at the free
end — matches the README verbatim):

```
Analyze a cantilever beam, 500mm long, 50mm square cross-section, steel,
with 1000N downward force at the free end
```

Analytical check (Euler-Bernoulli point-load cantilever, so you can
PASS/FAIL it the same way the ALL-FEM test does):

```
I = a^4 / 12 = 50^4 / 12 = 520,833.3 mm^4
delta = P*L^3 / (3*E*I) = 1000 * 500^3 / (3 * 200000 * 520833.3)
      = 0.400 mm  (downward)
```

Run it one-shot via the CLI — this is the only execution mode FeaGPT
actually documents (there's no single-agent/multi-agent toggle to choose
between the way there was in ALL-FEM; see the callout at the top):

```bash
feagpt run "Analyze a cantilever beam, 500mm long, 50mm square cross-section, steel, with 1000N downward force at the free end" 2>&1 | tee ~/feagpt_beam.log
```

Check the printed max von Mises stress / tip deflection / safety factor
against the analytical deflection above (0.400mm; expect the FEM answer to
be somewhat stiffer/softer than beam theory depending on mesh — same 15%
tolerance convention as the ALL-FEM test is a reasonable bar).

---

## 5. Test case 2 — compliant forceps-scissors mechanism

Source: Libu George B & R. Bharanidaran (2020), *Design of multifunctional
compliant forceps for medical application*, Aust. J. Mech. Eng.,
doi:10.1080/14484846.2020.1747151 — geometry per Figure 1, using the
"initial guess" values from Aguirre & Frecker (2008) that you supplied:

```
L1 = 3 mm      L5 = 2 mm
L2 = 4 mm      L6 = 2 mm
L3 = 8 mm      L7 = 0.85 mm
L4 = 9 mm      θ = 6°   (L3 incline to centerline)
               δ = 2°   (L5/L6 jaw-tip incline)
T1 = 1 mm      extruded thickness of L1, L2
T2 = 0.5 mm    extruded thickness of L4, L5, L6
```

Overall instrument: 5mm diameter, 15mm length. Material: 316L stainless
steel (biocompatible) — E ≈ 193 GPa, ν ≈ 0.3, ρ ≈ 8000 kg/m³ (standard 316L
values; the paper doesn't restate them, so give them explicitly in the
prompt rather than relying on FeaGPT's material knowledge base to have a
"316L" entry).

**Gap to flag before you run anything:** the paper never states the beam
cross-section width/height (the `w`/`h` in its Figure 2 inset) for *this*
geometry — only the lengths, two angles, and the two extrusion thicknesses
T1/T2. Aguirre & Frecker (2008), the cited source of the initial-guess
values, likely has that dimension; if you have access to it, use it. Absent
that, the prompt below assumes a **0.4mm beam width** as a placeholder —
call this out as an assumption, not a paper-sourced value, when you report
results.

The paper runs three separate static analyses on this one geometry —
useful because each gives you a real published number to check FeaGPT's
output against, the same way the beam case checks against Euler-Bernoulli:

| Case | Boundary condition | Load | Paper's result |
|---|---|---|---|
| 5a. Jaw opening (forceps) | Jaw tip fixed | 10mm axial displacement applied to sheath, opening direction | **1.049 mm** jaw displacement normal to opening direction (Fig. 4) |
| 5b. Cutting force (scissor) | Sheath: cylindrical support (fully fixed); jaw tip fixed to read reaction | 20 N·mm torsional moment applied at the beam | **0.654 N** reaction force at the tip = scissor cutting force |
| 5c. Additional opening | Jaw tip fixed | 2mm axial displacement applied to sheath, reverse direction | **0.29 mm** additional jaw-tip displacement beyond neutral (Fig. 6) |

### 5a. Jaw opening (forceps function)

```bash
feagpt run "Analyze a monolithic compliant forceps-scissors surgical instrument, 5mm overall diameter, 15mm overall length, made of 316L stainless steel with Young's modulus 193000 MPa, Poisson's ratio 0.3, density 8000 kg/m^3. The jaw mechanism is a chain of straight flexure beam links, mirrored about the centerline: L1=3mm rear actuation link, L2=4mm connector link, L3=8mm main flexure beam inclined at 6 degrees to the centerline passing through the sheath, L4=9mm jaw beam, L5=2mm and L6=2mm jaw-tip segments inclined at 2 degrees forming a sharp cutting tip, and a 0.85mm lever link L7 perpendicular to L3 near the sheath entrance. Beam cross-section width 0.4mm. Extrude L1 and L2 with thickness 1mm; extrude L4, L5, and L6 with thickness 0.5mm. A rigid cylindrical sheath of 5mm outer diameter surrounds the L3/L4 region. Fix the jaw tip end. Apply a 10mm displacement to the sheath in the jaw-opening axial direction. Perform static structural analysis and report the jaw displacement normal to the opening direction." 2>&1 | tee ~/feagpt_forceps_opening.log
```

Compare the reported jaw-normal displacement to **1.049mm**.

### 5b. Cutting force (scissor function)

Same geometry block, different BC/load — reuse the geometry description
verbatim and swap only the boundary-condition sentence:

```bash
feagpt run "Analyze a monolithic compliant forceps-scissors surgical instrument, 5mm overall diameter, 15mm overall length, made of 316L stainless steel with Young's modulus 193000 MPa, Poisson's ratio 0.3, density 8000 kg/m^3. The jaw mechanism is a chain of straight flexure beam links, mirrored about the centerline: L1=3mm rear actuation link, L2=4mm connector link, L3=8mm main flexure beam inclined at 6 degrees to the centerline passing through the sheath, L4=9mm jaw beam, L5=2mm and L6=2mm jaw-tip segments inclined at 2 degrees forming a sharp cutting tip, and a 0.85mm lever link L7 perpendicular to L3 near the sheath entrance. Beam cross-section width 0.4mm. Extrude L1 and L2 with thickness 1mm; extrude L4, L5, and L6 with thickness 0.5mm. A rigid cylindrical sheath of 5mm outer diameter surrounds the L3/L4 region. Apply a fully fixed cylindrical support to the sheath. Apply a 20 N-mm torsional moment at the L2-L3 junction. Fix the jaw tip and report the reaction force there as the cutting force. Perform static structural analysis." 2>&1 | tee ~/feagpt_scissor_force.log
```

Compare the reported tip reaction force to **0.654 N**.

### 5c. Additional jaw opening (reverse sheath travel)

```bash
feagpt run "Analyze a monolithic compliant forceps-scissors surgical instrument, 5mm overall diameter, 15mm overall length, made of 316L stainless steel with Young's modulus 193000 MPa, Poisson's ratio 0.3, density 8000 kg/m^3. The jaw mechanism is a chain of straight flexure beam links, mirrored about the centerline: L1=3mm rear actuation link, L2=4mm connector link, L3=8mm main flexure beam inclined at 6 degrees to the centerline passing through the sheath, L4=9mm jaw beam, L5=2mm and L6=2mm jaw-tip segments inclined at 2 degrees forming a sharp cutting tip, and a 0.85mm lever link L7 perpendicular to L3 near the sheath entrance. Beam cross-section width 0.4mm. Extrude L1 and L2 with thickness 1mm; extrude L4, L5, and L6 with thickness 0.5mm. A rigid cylindrical sheath of 5mm outer diameter surrounds the L3/L4 region. Fix the jaw tip end. Apply a 2mm displacement to the sheath in the reverse axial direction, opposite the jaw-opening direction. Perform static structural analysis and report the additional jaw-tip displacement beyond the neutral position." 2>&1 | tee ~/feagpt_forceps_additional_opening.log
```

Compare the reported additional displacement to **0.29mm**.

### Expected failure points

Based on what the GMSA pipeline is actually built to do (§II of the
FeaGPT paper):

- **Geometry complexity** — this is a single solid, but a much more
  intricate one than the wing (multiple angled/tapered segments, a
  sub-mm lever link, a surrounding sheath) built entirely from prose
  rather than the wing's more standard "airfoil + spars + ribs" pattern.
  Expect Novel Synthesis Mode (no pre-seeded knowledge-base pattern for
  this shape) to carry more risk of failing the AST/security/topology
  validator (§II.C) than the beam case.
- **Self-contact is invisible to the schema.** The paper's own model
  defines three frictionless contact regions between the sheath and the
  flexure beams (Node-Normal-to-Target detection) — FeaGPT's JSON spec
  has no field for contact/interaction pairs, so if a run "succeeds" it's
  almost certainly solving the flexure without that contact constraint,
  which will bias results away from the paper's numbers independent of
  any meshing/material differences.
- **Sub-mm features at "ultra fine" mesh density** — T2=0.5mm and
  L7=0.85mm are close to the paper's own hmin tier boundaries (§II.D);
  expect either an automatically-triggered ultra-fine mesh (small,
  slow-to-converge elements) or, if the mesher under-resolves the flexure
  hinge, an unrealistically stiff result.

Log whatever actually happens against each of the three published numbers
above — a clean run that's off by some percentage, an outright validator
rejection, or a silently-wrong result, are all useful outcomes to report.

---

## 6. Troubleshooting

- **`ccx: command not found`** — you're in a shell where `conda activate
  feagpt` wasn't run. `source /opt/conda/bin/activate feagpt` first.
- **`ModuleNotFoundError: No module named 'FreeCAD'`** — same cause, or
  conda-forge's `freecad` didn't put its Python bindings on `sys.path`
  correctly for your Python version; confirm `python3 -c "import FreeCAD"`
  *inside* the activated `feagpt` env before debugging further downstream.
- **Gemini API errors (401/403/quota)** — confirm `echo $GEMINI_API_KEY` is
  non-empty in the *current* shell (env vars don't persist across new tmux
  windows/panes — re-export in each one), and that the pod has outbound
  network access to `generativelanguage.googleapis.com`.
- **Generated FreeCAD script rejected by the validator** — the paper
  describes a 3-layer check (AST syntax, 18-op security blacklist incl.
  `os.system`/`eval`, FreeCAD topology feasibility, §II.C). If a legitimate
  geometry gets rejected, the error should tell you which layer — there's
  no documented override/bypass flag, so the practical fix is rewording the
  prompt to steer the LLM toward simpler, more standard geometry.
- **Mesh generation hangs or produces a degenerate mesh** — check the
  `hmin`/`hmax` implied by your prompt's density keyword ("ultra fine" =
  0.2mm minimum feature size per §II.D) against your actual part size; an
  "ultra fine" mesh on a 500mm beam is a much bigger job than on a small
  bracket and can blow past a small pod's RAM.
- **CalculiX doesn't converge** — check `~/FeaGPT`'s working/output
  directory for the `.inp`/`.dat`/`.sta` files CalculiX writes and inspect
  the `.sta` file's iteration log directly; this is standard CalculiX
  debugging, unrelated to FeaGPT's LLM layer.

## 7. Trying a different Gemini model version

Unlike ALL-FEM (where swapping models is the whole point), FeaGPT hardcodes
Gemini 2.5 Pro in the paper's description. Check `~/FeaGPT/feagpt/config.py`
and `config.yaml` for a model-name field before assuming it's swappable:

```bash
grep -rn "gemini" ~/FeaGPT/feagpt/config.py ~/FeaGPT/config.yaml 2>/dev/null
```

---

## 8. Reporting back (for comparison against the ALL-FEM run)

For an apples-to-apples writeup against `ALLFEM_RUNPOD_RUNBOOK.md`, capture
for each test case: PASS/MARGINAL/FAIL against its reference value (beam:
0.400mm analytical; forceps-scissors: 1.049mm / 0.654N / 0.29mm from the
George & Bharanidaran paper — note the self-contact caveat from §5 when
judging any mismatch there, since the paper's benchmark includes contact
behavior FeaGPT's schema can't express), wall-clock time per stage, and —
the more interesting axis given the architectural difference — *where*
each framework's failures come from: ALL-FEM's failures were in the LLM's
generated code (e.g. scalar vs. vector `FunctionSpace`), while FeaGPT's
failures, if any, are more likely to be architectural (schema/validator
can't express the problem at all, or silently drops the contact physics)
rather than a wrong-but-runnable answer.
