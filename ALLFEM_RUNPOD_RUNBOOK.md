# ALL-FEM RunPod runbook — cantilever + scissor experiments

Distilled from a live setup on 2026-07-12. Follow this on a **fresh** pod for
a one-shot run (the false starts we hit the first time are already fixed
into the script below). Covers both `test_allfem_cantilever.py` (§0-5) and
`test_allfem_scissor.py`, including its `--framework two-agent`/`multi-agent`
modes (§7).

Known-good result (cantilever): `rushikesh_67/qwen3-short_think-fenics-local`
PASSes (8.7% relative error vs. Euler-Bernoulli beam theory).
`rushikesh_67/llama3.2-2new` FAILs — it emits legacy dolfin code but with a
scalar `FunctionSpace` instead of a `VectorFunctionSpace`, which is a real
bug in that model's output, not an environment problem.

---

## 0. One-time RunPod console setup

1. Deploy a Pod with a GPU. An A40 (46 GB) or similar comfortably runs the
   3B/32B tier. Don't use the 120B gpt-oss fine-tune without an
   A100-80GB/H100 — check the model's Ollama page for VRAM needs. This
   applies equally to the scissor experiment: its FE problem is mm-scale
   and cheap to solve, so GPU load is dominated by the LLM, not the solver.
2. Use the latest PyTorch template.
3. (Optional) Under **Expose HTTP Ports**, add `11434` only if you want to
   hit Ollama from outside the pod — not required for this script.
4. Open the pod's Web Terminal.
5. Set up claude
```bash
curl -fsSL https://claude.ai/install.sh | bash 

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc 
```

## 1. Start a persistent tmux session

Do this first so a dropped SSH/web-terminal connection doesn't kill a
multi-minute install.

```bash
# if tmux isn't already installed on the pod image:
apt-get update -qq && apt-get install -y tmux

tmux new -s allfem
```

Everything below runs inside this session. To check on it from another
shell later: `tmux attach -t allfem`, or non-interactively:
`tmux capture-pane -t allfem:0 -p | tail -40`.

## 2. Get the repo onto the pod

No `runpodctl send`/`receive` needed — every script and test harness lives
in this git repo, so just clone it (or pull the latest) directly on the
pod, inside the tmux session:

```bash
cd ~   # or /workspace — just be consistent
git clone https://github.com/omnaathg/allfem_experiments.git
cd allfem_experiments
```

If you already have a clone from a previous session and just pushed new
changes locally, pull instead of re-cloning:

```bash
cd ~/allfem_experiments && git pull
```

The repo is public, so no auth is needed for either clone or pull. Confirm
the files are there:
`ls -la runpod_allfem_cantilever.sh test_allfem_cantilever.py test_allfem_scissor.py`

## 3. Run it (cantilever only — see §7 to bootstrap-and-run the scissor test)

```bash
chmod +x runpod_allfem_cantilever.sh
./runpod_allfem_cantilever.sh                                    # default: cantilever, qwen3-short_think-fenics-local
# or: ./runpod_allfem_cantilever.sh rushikesh_67/llama3.2-2new    # cantilever, different model
```

The script only ever runs `test_allfem_cantilever.py` — it takes `--model`
as its sole `$1` and does not accept a second positional arg to pick a
different test script (any extra args are silently ignored). To bootstrap
the environment and then run the scissor test in one go, run this script
once for the cantilever test (or just to build the `fenics` conda env),
then call `test_allfem_scissor.py` directly per §7.

Pipe through `tee` if you want a saved log as well as live output:

```bash
./runpod_allfem_cantilever.sh 2>&1 | tee ~/allfem_install.log
```

### What it does, in order

1. **Ollama** — installs if missing, starts `ollama serve` in the
   background, pulls the model (~6–34 GB depending on model; skips if
   already pulled).
2. **Conda + FEniCS** — installs Miniconda if missing, accepts Anaconda's
   default-channel Terms of Service non-interactively (required on a fresh
   pod or `conda create` fails with `CondaToSNonInteractiveError`), then
   creates a `fenics` conda env with the **legacy** `fenics` (dolfin
   2019.1) package + `mshr` (mesh-generation companion the models' code
   tends to import) + `mpich` + `python=3.11`, all in one `conda create` to
   avoid a second solve.
3. **Sanity check** — confirms the target test script is next to the
   script.
4. **Run the test** — prompts the model, saves the generated code, executes
   it, and compares against the reference value (cantilever: PASS if <15%
   relative error vs. Euler-Bernoulli; scissor: see §7's looser verdict
   bands).

Whole thing is idempotent — safe to re-run after a pod restart; already-done
steps are skipped and it jumps straight back to the test.

**Important:** this wrapper script only forwards `--model` to the test
harness (`python3 "$SCRIPT_DIR/$TEST_SCRIPT" --model "$MODEL"`). It does
**not** pass through `--framework`, `--base-model`, `--coordinator-mode`, or
any of the scissor harness's other flags. Use it to bootstrap the
environment (once), then invoke `test_allfem_scissor.py` directly for
anything beyond the two-agent default — see §7.

## 4. Troubleshooting reference (already baked into the script, kept here for context)

- **`CondaToSNonInteractiveError`** on `conda create` → the script now runs
  `conda tos accept --override-channels --channel <main|r>` before creating
  the env. If you ever hit this manually: `source /opt/conda/bin/activate`
  first, then run those two `conda tos accept` commands.
- **`ModuleNotFoundError: No module named 'dolfin'`** → the model emitted
  legacy FEniCS code but the env has `fenics-dolfinx` (modern API) instead
  of `fenics` (legacy). Fix: set `FENICS_PACKAGE="fenics"` in the script,
  `conda env remove -y -n fenics`, re-run. Applies to both test scripts —
  both prompts allow the model to choose dolfin or dolfinx.
- **`ModuleNotFoundError: No module named 'mshr'`** → legacy dolfin env is
  missing the `mshr` companion package (used for CSG mesh generation). Fix:
  add `mshr` to the `conda create` package list (already done in this
  script), or `conda install -y -c conda-forge mshr` into the existing env.
- **`ufl.log.UFLException: Symmetric part of tensor with rank != 2 is
  undefined`** → not an environment issue — the model generated a scalar
  `FunctionSpace` instead of a `VectorFunctionSpace` for a vector mechanics
  problem. This is a genuine bug in that model's code generation
  (`rushikesh_67/llama3.2-2new` did this); try a different model.

## 5. Trying another model (cantilever)

```bash
source /opt/conda/bin/activate fenics
ollama pull <model-tag>
python3 ~/allfem_experiments/test_allfem_cantilever.py --model <model-tag>
```

No need to re-run the whole script — Ollama and the conda env are already
set up; this just pulls the new model and re-runs the test directly.

## 6. Pushing script changes back

Since the pod now has its own clone, any edits you make to the scripts on
the pod should be pushed from your local machine's copy instead (the pod
clone is read-only from GitHub's point of view unless you set up your own
credentials there). Edit locally, `git push`, then `git pull` on the pod to
pick up the change — no file-transfer step required either way.

## 7. Running the scissor experiment (two-agent and multi-agent)

`test_allfem_scissor.py` tests the scissor/cutting-mode sub-analysis from
Libu George B & Bharanidaran (2020) — see the script's module docstring for
the full surrogate-model caveats (no jaw crossover, no contact elements,
text-only geometry input, etc.) and the PLAUSIBLE/QUESTIONABLE/FAIL verdict
bands (a looser, order-of-magnitude check against the paper's 0.654N, not a
tight PASS/FAIL like the cantilever test).

Once the environment exists (§0-3 above, run at least once against either
test), activate it and call the harness directly:

```bash
source /opt/conda/bin/activate fenics
cd ~/allfem_experiments
```

**Two-agent** (reproduces the leaderboard's per-model condition — Coder,
then one Debugger fix attempt on failure):

```bash
python3 test_allfem_scissor.py \
  --model rushikesh_67/qwen3-short_think-fenics-local \
  --framework two-agent
```

**Multi-agent, fixed dispatch order** (Formulator → Planner →
[Coder↔Corrector↔Executor loop] → Evaluator → Admin, one model standing in
for every role):

```bash
python3 test_allfem_scissor.py \
  --model rushikesh_67/qwen3-short_think-fenics-local \
  --framework multi-agent --coordinator-mode fixed
```

**Multi-agent, dynamic Coordinator, faithful base-vs-fine-tuned split**
(mirrors the paper's Figure 4 GPT-OSS vs. GPT-OSS-FT tiers — `--model`
drives the Coder/Corrector roles, `--base-model` drives
Coordinator/Formulator/Planner/Evaluator). Models load sequentially per
role, not concurrently, so VRAM is bounded by the larger of the two tags,
not their sum:

```bash
ollama pull llama3.3:70b
python3 test_allfem_scissor.py \
  --model rushikesh_67/qwen3-short_think-fenics-local \
  --base-model llama3.3:70b \
  --framework multi-agent --coordinator-mode dynamic
```

Expect multi-agent runs to take noticeably longer than the cantilever test
or the two-agent scissor run — up to 5 Ollama calls per attempt (Formulator,
Planner, Coder, Evaluator, plus any Corrector/Debugger retries), more if the
`--max-debug-turns` retry loop triggers (default: 1 for two-agent, 3 for
multi-agent; override with `--max-debug-turns`).

If `--base-model` is omitted in multi-agent mode, the script prints a
`NOTE:` and falls back to reusing `--model` for every role — still runs
fine, just doesn't exercise the paper's base-vs-fine-tuned split.

See `python3 test_allfem_scissor.py --help` for the full flag list,
including per-role overrides (`--coder-model`, `--corrector-model`,
`--coordinator-model`, `--formulator-model`, `--planner-model`,
`--evaluator-model`) and `--max-coordinator-turns` (safety cap on the
dynamic Coordinator's routing rounds, default 8).

## 8. Scissor test: recurring model bug categories (fed into the Debugger)

Both `qwen3-short_think-fenics-local` (two-agent) and the
`llama3.3:70b`-coordinated multi-agent run FAILed outright on the scissor
problem — never producing a parseable `SCISSOR_FORCE_N`. Hand-fixing the
two-agent output (see `generated_fenics_scissor_code.py`, regenerated fresh
each run — not tracked in git) surfaced six distinct bug categories, several
of which don't crash at all (the script runs and prints a number, just a
physically meaningless one):

1. **Missing start point** — building the centerline as only segment *end*
   points (never storing the very first vertex) silently drops the L1
   segment and moves the "free end" to the wrong location.
2. **Zero-width polygon** — closing the bare centerline into a polygon with
   no thickness offset gives a degenerate sliver, not a real cross-section.
3. **Mesher crash from kink geometry** — offsetting each side of a kink
   with its own segment's normal creates a second corner point only
   `~half_thickness * sin(kink angle)` away from the first — far below the
   mesh's target cell size, which crashes CGAL. Needs a single mitred
   corner point per kink instead.
4. **Non-physical roller** — pinning both the top and bottom surface points
   of a support cross-section (instead of one) creates an internal
   force couple that resists rotation, i.e. it behaves like a second
   clamp, not a roller.
5. **Wrong reaction-force method** — computing `assemble(dot(u_sol, v)*ds(...))`
   integrates the *displacement field*, which has the wrong physical units
   and isn't a force at all, though it runs without error. The correct
   approach is the discrete equilibrium residual,
   `assemble(action(a, u_sol) - L)`, summed over the constrained boundary's
   DOFs.
6. **Traction in the wrong component** (the one that actually mattered
   most): bending stress `sigma_xx(y)` belongs in the *axial* component of
   the Cauchy traction on an end face (`t = sigma . n`), not the transverse
   component. Putting it in the wrong component doesn't error either — it
   silently produces a solution ~1000x too stiff (near-zero deflection).

These are now baked into `COMMON_PITFALLS` in `test_allfem_scissor.py`,
which is fed into both debugger-role prompts (`DEBUG_INSTRUCTIONS` for
two-agent, `CODER_RETRY_INSTRUCTIONS` for multi-agent) — but *not* the
initial Coder prompt, so first-attempt behavior is unaffected. Whether this
actually helps either model self-correct on retry hasn't been tested yet.

With all six fixed by hand, `qwen3-short_think-fenics-local`'s two-agent
output gives a mesh-converged `SCISSOR_FORCE_N ≈ 1.357 N` against the
paper's reference of 0.654 N (ratio 2.07x — inside the harness's own
0.2x–5x "PLAUSIBLE" band for this simplified surrogate).

### Open geometry gap: L7

The paper's parameter list includes an `L7 = 0.85mm` "separate lever link"
that `PROBLEM_STATEMENT`/`PARAMS` in `test_allfem_scissor.py` never
mentions at all — L1-L6 only. Per clarification: L7 attaches at the L2-L3
junction (not a continuation of the main chain), protrudes transversely,
and is what the outer casing/sheath actually contacts to actuate the
mechanism (the casing moves opposite to the tip, levering against L7's
tip) — the main chain isn't touched by the sheath directly.

Added L7 by hand as a small triangular tab fused (CSG union) onto the main
chain at the L2-L3 junction, with the roller/sheath-contact constraint
moved from the main chain's surface to the tab's tip. Result was
essentially unchanged (`1.357 N` either way, <0.01% difference) — L7's
0.85mm length is short enough relative to the ~28mm mechanism that it
doesn't meaningfully perturb the tip's far-field transverse reaction under
this loading. Still open: whether the roller there should constrain the
lever tip's *transverse* displacement (current assumption, carried over
from the pre-L7 model) or its *axial* displacement (arguably a better
match for "casing moves opposite to the tip" — i.e. genuine axial
casing motion levered into a transverse/moment effect via the 0.85mm arm),
which would be a materially different mechanism and likely change the
result more.
