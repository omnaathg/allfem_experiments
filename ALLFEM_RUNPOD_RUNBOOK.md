# ALL-FEM cantilever test on RunPod — runbook

Distilled from a live setup on 2026-07-12. Follow this on a **fresh** pod for
a one-shot run (the false starts we hit the first time are already fixed
into the script below).

Known-good result: `rushikesh_67/qwen3-short_think-fenics-local` PASSes
(8.7% relative error vs. Euler-Bernoulli beam theory). `rushikesh_67/llama3.2-2new`
FAILs — it emits legacy dolfin code but with a scalar `FunctionSpace` instead
of a `VectorFunctionSpace`, which is a real bug in that model's output, not
an environment problem.

---

## 0. One-time RunPod console setup

1. Deploy a Pod with a GPU. An A40 (46 GB) or similar comfortably runs the
   two models above. Don't use the 120B gpt-oss fine-tune without an
   A100-80GB/H100 — check the model's Ollama page for VRAM needs.
2. Use the latest PyTorch template.
3. (Optional) Under **Expose HTTP Ports**, add `11434` only if you want to
   hit Ollama from outside the pod — not required for this script.
4. Open the pod's Web Terminal.

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

No `runpodctl send`/`receive` needed — `runpod_allfem_cantilever.sh` and
`test_allfem_cantilever.py` live in the git repo, so just clone it (or pull
the latest) directly on the pod, inside the tmux session:

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

The repo is public, so no auth is needed for either `clone` or `pull`.
Confirm the files are there:
`ls -la runpod_allfem_cantilever.sh test_allfem_cantilever.py`

## 3. Run it

From inside the `allfem_experiments` directory:

```bash
chmod +x runpod_allfem_cantilever.sh
./runpod_allfem_cantilever.sh                                    # default model: qwen3-short_think-fenics-local
# or: ./runpod_allfem_cantilever.sh rushikesh_67/llama3.2-2new    # to try a different model
```

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
   2019.1) package + `mshr` (mesh-generation companion the model's code
   imports) + `mpich` + `python=3.11`, all in one `conda create` to avoid a
   second solve.
3. **Sanity check** — confirms `test_allfem_cantilever.py` is next to the
   script.
4. **Run the test** — prompts the model for FEniCS code solving a
   cantilever beam problem, saves the generated code to
   `generated_fenics_code.py`, executes it, and compares the tip deflection
   against the Euler-Bernoulli analytical solution (PASS if <15% relative
   error and correct sign).

Whole thing is idempotent — safe to re-run after a pod restart; already-done
steps are skipped and it jumps straight back to the test.

## 4. Troubleshooting reference (already baked into the script, kept here for context)

- **`CondaToSNonInteractiveError`** on `conda create` → the script now runs
  `conda tos accept --override-channels --channel <main|r>` before creating
  the env. If you ever hit this manually: `source /opt/conda/bin/activate`
  first, then run those two `conda tos accept` commands.
- **`ModuleNotFoundError: No module named 'dolfin'`** → the model emitted
  legacy FEniCS code but the env has `fenics-dolfinx` (modern API) instead
  of `fenics` (legacy). Fix: set `FENICS_PACKAGE="fenics"` in the script,
  `conda env remove -y -n fenics`, re-run.
- **`ModuleNotFoundError: No module named 'mshr'`** → legacy dolfin env is
  missing the `mshr` companion package (used for CSG mesh generation, and
  the models we tried both `import mshr`). Fix: add `mshr` to the
  `conda create` package list (already done in this script), or
  `conda install -y -c conda-forge mshr` into the existing env.
- **`ufl.log.UFLException: Symmetric part of tensor with rank != 2 is
  undefined`** → not an environment issue — the model generated a scalar
  `FunctionSpace` instead of a `VectorFunctionSpace` for a vector mechanics
  problem. This is a genuine bug in that model's code generation
  (`rushikesh_67/llama3.2-2new` did this); try a different model.

## 5. Trying another model

```bash
source /opt/conda/bin/activate fenics
ollama pull <model-tag>
python3 ~/allfem_experiments/test_allfem_cantilever.py --model <model-tag>
```

No need to re-run the whole script — Ollama and the conda env are already
set up; this just pulls the new model and re-runs step 4 directly.

## 6. Pushing script changes back

Since the pod now has its own clone, any edits you make to the scripts
*on the pod* should be pushed from your local machine's copy instead (the
pod clone is read-only from GitHub's point of view unless you set up your
own credentials there). Edit locally, `git push`, then `git pull` on the
pod to pick up the change — no file-transfer step required either way.
