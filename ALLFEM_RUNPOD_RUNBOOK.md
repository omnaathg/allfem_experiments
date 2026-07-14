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

## 3. Run it (cantilever, or a quick-start scissor two-agent run)

```bash
chmod +x runpod_allfem_cantilever.sh
./runpod_allfem_cantilever.sh                                    # default: cantilever, qwen3-short_think-fenics-local
# or: ./runpod_allfem_cantilever.sh rushikesh_67/llama3.2-2new    # cantilever, different model
# or: ./runpod_allfem_cantilever.sh rushikesh_67/qwen3-short_think-fenics-local test_allfem_scissor.py
#     # scissor experiment, two-agent framework (the script's only mode — see §7 for multi-agent)
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
