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

**Updated after a live run on 2026-07-15** — the naive version of this
script (conda env + `pip install -r requirements.txt` + `pip install -e .`)
passes conda's own solve but fails two of the four sanity checks and the
"install the CLI" step, all for reasons specific to this repo/toolchain
combination rather than anything pod-specific. The three fixes are baked
into the script below; the *why* for each is in the inline comments and
repeated in Troubleshooting (§6) in case they resurface:

1. `import gmsh` fails with `OSError: libXft.so.2: cannot open shared
   object file` — `requirements.txt` pins `gmsh>=4.11.0` via **pip**, which
   silently shadows the conda-forge `gmsh` installed in step 2 with a PyPI
   wheel that dynamically links a system X11 font library not present on a
   stock Ubuntu pod. Fixed with one `apt-get install libxft2`.
2. `import FreeCAD` fails with `ModuleNotFoundError` even though the conda
   package installed cleanly — conda-forge's FreeCAD 1.1.0 build puts its
   compiled bindings (`FreeCAD.so`, `FreeCADGui.so`) in `$CONDA_PREFIX/lib`,
   not `site-packages`, so they're never on `sys.path` by default. Fixed by
   writing a conda `activate.d`/`deactivate.d` hook so `PYTHONPATH` is set
   automatically on every future `conda activate feagpt` — no per-shell
   manual export needed.
3. `pip install -e .` fails outright with `AssertionError: Exactly one
   .egg-info should have been produced, but found 0` — as of the commit
   tested, this repo's `setup.py` is an **empty file** and there's no
   `pyproject.toml`. Even if packaging were fixed, there's no
   `console_scripts`/`entry_points` defined anywhere in the repo, so a
   `feagpt` command would never exist either way. **The actual working
   entry point is the repo's own `main.py`**, a click CLI with the same
   `run` / `interactive` / `batch` subcommands, runnable straight from the
   repo root with no install step (the `feagpt/` package is just plain
   importable relative to cwd). Every `feagpt run "..."` command elsewhere
   in this runbook is really `python main.py run "..."` — that substitution
   is made throughout §4/§5 below. The bootstrap still attempts
   `pip install -e .` (non-fatal) in case upstream fixes packaging later;
   check the sanity checks in step [5/5] rather than assuming either way.

```bash
cat > ~/bootstrap_feagpt.sh << 'BOOTSTRAP'
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="feagpt"
REPO_DIR="${HOME}/FeaGPT"

echo "=== [1/5] Miniconda ==="
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
echo "=== [2/5] FreeCAD + Gmsh + CalculiX (conda-forge, one solve) ==="
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" -c conda-forge \
        python=3.10 freecad "gmsh>=4.11" calculix
else
    echo "Conda env '$ENV_NAME' already exists."
fi
conda activate "$ENV_NAME"

# Known fix #2: conda-forge's FreeCAD bindings live in $CONDA_PREFIX/lib,
# not site-packages. Bake PYTHONPATH into the env's own activate hook so
# every future `conda activate feagpt` (this shell and any new one) picks
# it up automatically.
ACTIVATE_D="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_D="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"
if [ ! -f "${ACTIVATE_D}/freecad_pythonpath.sh" ]; then
    cat > "${ACTIVATE_D}/freecad_pythonpath.sh" << 'HOOK'
export _FEAGPT_OLD_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${CONDA_PREFIX}/lib:${PYTHONPATH:-}"
HOOK
    cat > "${DEACTIVATE_D}/freecad_pythonpath.sh" << 'HOOK'
export PYTHONPATH="${_FEAGPT_OLD_PYTHONPATH:-}"
unset _FEAGPT_OLD_PYTHONPATH
HOOK
fi
# activate.d only fires on `conda activate`, and we already ran that above
# before writing the hook — so apply it to *this* script's shell too.
export PYTHONPATH="${CONDA_PREFIX}/lib:${PYTHONPATH:-}"

echo
echo "=== [3/5] System dependency for gmsh (known fix #1) ==="
if ! ldconfig -p | grep -q libXft.so.2; then
    apt-get update -qq && apt-get install -y -qq libxft2
else
    echo "libXft.so.2 already present."
fi

echo
echo "=== [4/5] FeaGPT ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/naividh/FeaGPT.git "$REPO_DIR"
else
    echo "Repo already cloned, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only || true
fi
cd "$REPO_DIR"

# Known fixes #4-#10 (§3.5): the repo as published never actually runs its
# own README example end-to-end -- config loading, geometry export, BC/load
# application, and material units are all broken. Patch file lives
# alongside this runbook; see §3.5 for what each hunk fixes and why.
echo "Applying known FeaGPT code fixes (see runbook §3.5)..."
curl -fsSL https://raw.githubusercontent.com/omnaathg/allfem_experiments/main/feagpt_bugfixes.patch -o /tmp/feagpt_bugfixes.patch
if git apply --check /tmp/feagpt_bugfixes.patch 2>/dev/null; then
    git apply /tmp/feagpt_bugfixes.patch
    echo "Bug fixes applied."
else
    echo "Patch did not apply cleanly (already applied, or upstream changed) -- check manually against §3.5."
fi

pip install -r requirements.txt
# Known fix #3: expected to fail on the repo state described above.
# Non-fatal; use `python main.py run ...` regardless of outcome.
pip install -e . 2>&1 | tail -5 || echo "pip install -e . failed as expected (see §3 known fix #3) -- use 'python main.py run ...' instead."

echo
echo "=== [5/5] Sanity checks ==="
echo -n "ccx (CalculiX):   "; command -v ccx || echo "NOT ON PATH"
echo -n "gmsh:             "; python3 -c "import gmsh; print(gmsh.__file__)" 2>/dev/null || echo "NOT IMPORTABLE"
echo -n "FreeCAD:          "; python3 -c "import FreeCAD; print(FreeCAD.__file__)" 2>/dev/null || echo "NOT IMPORTABLE"
echo -n "feagpt CLI:       "; command -v feagpt || echo "NOT ON PATH (expected -- see known fix #3; use 'python main.py')"
echo -n "main.py CLI:      "; (cd "$REPO_DIR" && python3 main.py --help >/dev/null 2>&1 && echo "OK") || echo "FAILED"

echo
echo "Bootstrap done. Next: export GEMINI_API_KEY, then run a case (see runbook)."
BOOTSTRAP

chmod +x ~/bootstrap_feagpt.sh
~/bootstrap_feagpt.sh 2>&1 | tee ~/feagpt_install.log
```

If `ccx`, `gmsh`, or `FreeCAD` in step [5/5] still fail after the fixes
above, stop and check Troubleshooting (§6) — don't proceed to test cases on
a broken sanity check. `feagpt CLI: NOT ON PATH` is *expected* given known
fix #3 above; what matters is `main.py CLI: OK`. The most common residual
miss is being in a shell that hasn't run `conda activate feagpt` at all —
always do that in new shells before running anything below.

Then, every new shell/tmux window:

```bash
source /opt/conda/bin/activate feagpt
cd ~/FeaGPT
read -s -p "Gemini key: " GEMINI_API_KEY && export GEMINI_API_KEY
```

That last line reads the key straight into the env var without ever typing
it as a plain command-line argument, so it doesn't land in shell history or
any terminal-session logging (including AI coding assistants sharing this
shell, if you're using one) — same effect as
`export GEMINI_API_KEY="..."` but without the exposure. If a key ever does
end up somewhere it shouldn't (pasted into a chat, committed, etc.), treat
it as compromised and rotate it at
[Google AI Studio](https://aistudio.google.com/apikey) immediately rather
than trying to scrub the exposure after the fact.

## 3.5 Known FeaGPT code bugs (patched automatically above)

**Updated after actually running test case 1 to completion on 2026-07-15.**
The bootstrap script's patch step (§3) fixes all seven of these
automatically by applying `feagpt_bugfixes.patch` (checked into this repo
alongside this runbook) to the freshly-cloned `~/FeaGPT`. This section
exists so you know *why* each hunk is there and can diagnose a failure if
the patch stops applying cleanly against a newer upstream commit. None of
this is environment-specific — it reproduces on any pod. Before these
fixes, **FeaGPT's own README worked example never ran to completion**, at
any stage past config loading.

1. **Config never loaded, API key silently ignored.** `main.py` called
   `FeaGPTConfig(ctx.obj["config_path"])` — the dataclass's positional
   constructor — instead of the `FeaGPTConfig.from_yaml(...)` classmethod
   that actually parses YAML and applies `GEMINI_API_KEY`. Since `llm` is
   the first dataclass field, this silently set `config.llm` to the literal
   string `"config.yaml"`, and every other field stayed at its hardcoded
   default. Every subcommand (`run`, `interactive`, `batch`) had this bug.
2. **`config.yaml` itself wasn't valid YAML.** Every line was indented two
   spaces more than the line before it (a "staircase" reaching ~150 spaces
   by the end of a 99-line file) — a `yaml.ParserError`. This was never
   caught before because bug #1 meant `from_yaml()` was never actually
   called on it.
3. **`feagpt` CLI doesn't exist.** `setup.py` is empty, no `pyproject.toml`,
   no `entry_points` anywhere in the repo. Not fixed by the patch (nothing
   to patch) — use `python main.py run "..."` throughout, as already
   reflected in §4/§5 below.
4. **Geometry export used a module that doesn't exist.** All three
   geometry generators (`naca_wing`, `cantilever_beam`, `plate_with_hole`)
   in `feagpt/geometry/generator.py` did `import FreeCAD, Part,
   importerStep` and called `importerStep.export(...)` — there is no
   `importerStep` module in any FreeCAD version. The correct module is
   `Import` (`Import.export(...)`). Every geometry generation call failed.
5. **Simulation stage never actually built the CalculiX deck.**
   `pipeline.py`'s `_run_simulation` called `self._simulator.run(mesh_path,
   spec, output_dir)` directly — but `FEASimulator.run()` only takes
   `(input_file, output_dir)`, and there's a separate
   `generate_input_deck(spec, mesh_file, output_path)` method that actually
   writes material/BC/load cards into a real `.inp` file, which was never
   called at all. Fixed by generating the deck first, then running it.
6. **CalculiX invoked with the wrong argument format.** `simulator.py`
   passed `-i <full_path>.inp` to `ccx`; CalculiX's `-i` flag wants the job
   name **without** the `.inp` extension (it appends `.inp` itself), so it
   looked for `job.inp.inp` and failed instantly with an empty stderr
   (`result.stderr` was blank because the error went to stdout). Fixed by
   passing `job_name` (the path stem) instead.
7. **BC/load node sets never existed, three compounding ways.** This was
   the deepest one:
   - `simulator.py`'s `generate_input_deck()` hardcodes literal `NFIX`/
     `NLOAD` node-set names in its `*BOUNDARY`/`*CLOAD` cards, but never
     reads `bc.get("location")` / `load.get("location")` from `spec` at
     all — so nothing it emits was ever going to match whatever the mesher
     actually named things.
   - `mesher.py`'s `_create_physical_groups()` created **element sets**
     (Gmsh 2D physical groups export as `*ELSET`, not `*NSET`) named after
     those same free-text `location` strings (e.g. `left_edge`,
     `top_edge`) — but CalculiX's `*BOUNDARY`/`*CLOAD` require **node
     sets**. Even with matching names, the type was wrong.
   - Underneath both: which face got which physical group was chosen by
     **raw index order** from `gmsh.model.occ.getEntities(2)`
     (`surfaces[i]`) — not by any actual geometric check that a face is
     the fixed end vs. the free end.
   - Fixed by adding `_write_bc_load_nsets()` to `mesher.py`: identifies the
     fixed/free faces by their bounding-box X-extent against the model's
     global bounding box (min-X face = fixed end, max-X face = free end),
     writes `NFIX` as *all* nodes on the fixed face (a proper root
     constraint) and `NLOAD` as the *single* node nearest the free face's
     centroid — deliberately a single node, not the whole face, because
     `*CLOAD` applies its magnitude once per node in the set, and the free
     end's node count would otherwise multiply a "1000N total" load into
     "1000N × node count." The old positional 2D physical-group creation in
     `_create_physical_groups()` was removed entirely (nothing downstream
     ever read it by name, and its exported surface elements are what
     caused bug worth calling out separately below).
   - Side effect of removing those 2D physical groups: gmsh's Abaqus/
     CalculiX `.inp` writer had also been exporting the boundary surface
     mesh as `CPS6` (plane-stress) elements, which CalculiX rejects outright
     for a 3D solid (`should lie in the z=0 plane`). Removing the unused
     groups stopped gmsh from emitting them.
8. **Material properties used raw SI units on a mm/N-scaled model.**
   Geometry is generated in millimeters (`Part.makeBox` from the spec's
   `*_mm` fields) and loads are applied directly in Newtons, so CalculiX
   needs a consistent mm/N/tonne system — E and density need to be in MPa
   (N/mm²) and tonne/mm³. Instead, `generate_input_deck()` wrote the raw SI
   values from `spec["material"]` (Pa, kg/m³) straight into the deck.
   Verified empirically on the cantilever case: reported tip deflection was
   exactly ~10^6 too small (Pa vs. MPa is off by 10^6). Fixed by converting
   `E / 1e6` and `density * 1e-12` when writing the material block.
9. **`config.simulation.num_threads` was dead config.** Present in the
   dataclass and settable from `config.yaml`, but `simulator.py` never
   passed it to the CalculiX subprocess as `OMP_NUM_THREADS` (or the
   CalculiX-specific `CCX_NPROC_STIFFNESS`/`CCX_NPROC_EQUATION_SOLVER`
   vars) — so it always ran single-threaded regardless of what was
   configured. On a ~213K-element mesh (the cantilever case's "fine"
   density; 924,954 equations), single-threaded CalculiX didn't finish
   inside the default 600s timeout. Fixed by wiring all three env vars from
   `config.simulation.num_threads`, and bumped `config.yaml` to
   `num_threads: 16` / `timeout: 1200` given a reasonably large pod (this
   session had 128 cores available). CalculiX's default `spooles` solver
   still factors single-threaded regardless (only matrix assembly and
   stress recovery parallelize) — expect a large mesh to still take
   several minutes; if that's too slow, a coarser mesh density is a real
   lever (§3's density tiers), not another bug to chase.
10. **Stage 5 (Analysis) is permanently disabled.**
    `feagpt.analysis.analyzer` doesn't export a `ResultAnalyzer` class (or
    it was renamed) — `pipeline.py` catches this at init as a non-fatal
    warning, so every run silently skips post-processing. Practical
    consequence: `results_data` (max stress, displacement, safety factor)
    is never populated even on a fully successful solve — pull results
    straight from CalculiX's `.frd` output instead (see the cantilever
    beam result below for the parsing approach). **Not fixed by the
    patch** — implementing it would mean writing FeaGPT's actual analysis
    module, which is a much larger lift than the fixes above; flagged here
    so it's not mistaken for another quick patch.

With all of the above except #3 and #10 fixed, the cantilever beam case
(§4) now runs end-to-end: reported tip deflection came out to **0.393mm**
against the **0.400mm** analytical target (1.6% deviation, comfortably
inside the 15% tolerance).

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
between the way there was in ALL-FEM; see the callout at the top). Command
below uses `python main.py run` per the known-fix #3 substitution in §3 —
run it from `~/FeaGPT` with the `feagpt` conda env active:

```bash
python main.py run "Analyze a cantilever beam, 500mm long, 50mm square cross-section, steel, with 1000N downward force at the free end" 2>&1 | tee ~/feagpt_beam.log
```

**With the §3.5 patch applied, this actually ran to completion on
2026-07-15: tip deflection came out to 0.393mm against the 0.400mm
analytical target (1.6% deviation — PASS, well within the 15% tolerance).**
Solve was 924,954 equations via CalculiX's direct solver, ~13 minutes
wall-clock on 16 threads. Max von Mises stress came out to ~164 MPa, at the
exact loaded node — that's an expected FEM point-load artifact (a true
concentrated force in a continuum mesh produces locally unbounded stress
right at the node, independent of mesh quality), not something to chase.

Because Stage 5/Analysis is disabled (§3.5, known fix #10), `main.py`'s own
printed "Max stress"/"Max displacement" lines will show `N/A` regardless of
solve success — pull the real numbers from CalculiX's `.frd` output
instead. The Y-displacement of the `NLOAD` node (logged by `mesher.py` at
meshing time, e.g. "NLOAD: node 3630 on free face") is the tip deflection;
find it in the `*DISP` block:

```bash
NODE=3630  # substitute the node number mesher.py logged for this run
awk '/DISP/{c++} c==1' results/job.frd | grep -E "^ -1 *${NODE} "
```

The three values on that line are X/Y/Z displacement in mm (Y is the
loaded direction here). For max von Mises stress across the whole model,
parse the `*STRESS` block (6 tensor components per node: Sxx, Syy, Szz,
Sxy, Syz, Szx) and compute von Mises directly — there's no single grep for
this one; a short Python loop over the block is simplest.

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
python main.py run "Analyze a monolithic compliant forceps-scissors surgical instrument, 5mm overall diameter, 15mm overall length, made of 316L stainless steel with Young's modulus 193000 MPa, Poisson's ratio 0.3, density 8000 kg/m^3. The jaw mechanism is a chain of straight flexure beam links, mirrored about the centerline: L1=3mm rear actuation link, L2=4mm connector link, L3=8mm main flexure beam inclined at 6 degrees to the centerline passing through the sheath, L4=9mm jaw beam, L5=2mm and L6=2mm jaw-tip segments inclined at 2 degrees forming a sharp cutting tip, and a 0.85mm lever link L7 perpendicular to L3 near the sheath entrance. Beam cross-section width 0.4mm. Extrude L1 and L2 with thickness 1mm; extrude L4, L5, and L6 with thickness 0.5mm. A rigid cylindrical sheath of 5mm outer diameter surrounds the L3/L4 region. Fix the jaw tip end. Apply a 10mm displacement to the sheath in the jaw-opening axial direction. Perform static structural analysis and report the jaw displacement normal to the opening direction." 2>&1 | tee ~/feagpt_forceps_opening.log
```

Compare the reported jaw-normal displacement to **1.049mm**.

### 5b. Cutting force (scissor function)

Same geometry block, different BC/load — reuse the geometry description
verbatim and swap only the boundary-condition sentence:

```bash
python main.py run "Analyze a monolithic compliant forceps-scissors surgical instrument, 5mm overall diameter, 15mm overall length, made of 316L stainless steel with Young's modulus 193000 MPa, Poisson's ratio 0.3, density 8000 kg/m^3. The jaw mechanism is a chain of straight flexure beam links, mirrored about the centerline: L1=3mm rear actuation link, L2=4mm connector link, L3=8mm main flexure beam inclined at 6 degrees to the centerline passing through the sheath, L4=9mm jaw beam, L5=2mm and L6=2mm jaw-tip segments inclined at 2 degrees forming a sharp cutting tip, and a 0.85mm lever link L7 perpendicular to L3 near the sheath entrance. Beam cross-section width 0.4mm. Extrude L1 and L2 with thickness 1mm; extrude L4, L5, and L6 with thickness 0.5mm. A rigid cylindrical sheath of 5mm outer diameter surrounds the L3/L4 region. Apply a fully fixed cylindrical support to the sheath. Apply a 20 N-mm torsional moment at the L2-L3 junction. Fix the jaw tip and report the reaction force there as the cutting force. Perform static structural analysis." 2>&1 | tee ~/feagpt_scissor_force.log
```

Compare the reported tip reaction force to **0.654 N**.

### 5c. Additional jaw opening (reverse sheath travel)

```bash
python main.py run "Analyze a monolithic compliant forceps-scissors surgical instrument, 5mm overall diameter, 15mm overall length, made of 316L stainless steel with Young's modulus 193000 MPa, Poisson's ratio 0.3, density 8000 kg/m^3. The jaw mechanism is a chain of straight flexure beam links, mirrored about the centerline: L1=3mm rear actuation link, L2=4mm connector link, L3=8mm main flexure beam inclined at 6 degrees to the centerline passing through the sheath, L4=9mm jaw beam, L5=2mm and L6=2mm jaw-tip segments inclined at 2 degrees forming a sharp cutting tip, and a 0.85mm lever link L7 perpendicular to L3 near the sheath entrance. Beam cross-section width 0.4mm. Extrude L1 and L2 with thickness 1mm; extrude L4, L5, and L6 with thickness 0.5mm. A rigid cylindrical sheath of 5mm outer diameter surrounds the L3/L4 region. Fix the jaw tip end. Apply a 2mm displacement to the sheath in the reverse axial direction, opposite the jaw-opening direction. Perform static structural analysis and report the additional jaw-tip displacement beyond the neutral position." 2>&1 | tee ~/feagpt_forceps_additional_opening.log
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
- **`ModuleNotFoundError: No module named 'FreeCAD'`** — known fix #2 (§3):
  conda-forge's `freecad` puts its compiled bindings in `$CONDA_PREFIX/lib`,
  not `site-packages`, so they're never on `sys.path` by default. The
  current bootstrap script bakes a `PYTHONPATH` fix into a conda
  `activate.d` hook so this should self-heal on `conda activate feagpt`; if
  it still happens, check `echo $PYTHONPATH` includes `$CONDA_PREFIX/lib`
  and that the hook files exist under
  `$CONDA_PREFIX/etc/conda/activate.d/`.
- **`OSError: libXft.so.2: cannot open shared object file` on `import
  gmsh`** — known fix #1 (§3): `requirements.txt` pins `gmsh` via pip,
  which shadows the conda-forge `gmsh` with a PyPI wheel needing a system
  X11 font library not present on a stock pod. Fixed by the bootstrap's
  `apt-get install libxft2` step; if it recurs, run that manually and
  re-check with `python3 -c "import gmsh"`.
- **`feagpt: command not found`, or `pip install -e .` fails with
  `AssertionError: Exactly one .egg-info should have been produced, but
  found 0`** — known fix #3 (§3): this repo's `setup.py` is empty and has
  no `pyproject.toml` or `entry_points`, so there is no `feagpt` console
  command to install, full stop — this isn't a transient install failure to
  retry. Use `python main.py run "..."` (also `interactive`, `batch`) from
  `~/FeaGPT` instead; confirm with `python3 main.py --help`.
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
- **Gemini API errors (401/403/quota)** — confirm `echo $GEMINI_API_KEY` is
  non-empty in the *current* shell (env vars don't persist across new tmux
  windows/panes — re-set in each one; use the `read -s -p` snippet in §3
  rather than typing the raw key inline so it never lands in shell
  history), and that the pod has outbound network access to
  `generativelanguage.googleapis.com`. Note there's also no
  `python-dotenv`/`load_dotenv()` anywhere in this repo — `.env` /
  `.env.example` are not actually read despite existing; a real exported
  env var is the only thing `feagpt/config.py` looks at. If a key is ever
  exposed somewhere it shouldn't be (chat log, committed file, shared
  terminal), rotate it immediately at
  [Google AI Studio](https://aistudio.google.com/apikey) rather than
  assuming the exposure can be cleaned up after the fact.

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
