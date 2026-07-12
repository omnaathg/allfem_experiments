#!/usr/bin/env python3
"""
test_allfem_scissor.py

ALL-FEM test #2: the "scissor/cutting mode" sub-analysis from

  Libu George B & R. Bharanidaran, "Design of multifunctional compliant
  forceps for medical application," Australian Journal of Mechanical
  Engineering 20(3), 2020. DOI: 10.1080/14484846.2020.1747151

This harness can run in two agent-orchestration modes, mirroring the two
evaluation conditions used on the ALL-FEM results leaderboard
(https://fenics-llm.github.io/Results/):

  --framework two-agent   (default)
      Coder agent generates code; if it fails to run, a Debugger agent
      gets one shot at fixing it using the runtime error. This is how
      the per-model results (e.g. "Qwen 3 32B Finetuned") on the
      leaderboard were produced -- pass --model to pick which of your
      local Ollama tags plays that role.

  --framework multi-agent
      A closer local approximation of the leaderboard's separate
      "Multi-Agent framework" condition (https://github.com/fenics-llm/
      Results_ALL-FEM/tree/main/Multi-Agent%20Framework). NOTE: that
      repo only publishes final generated code/output/plots for each
      benchmark, not the orchestration source itself, and its result
      files are all prefixed "oss-multi-", i.e. it was run with
      GPT-OSS 120B, not Qwen. There is no public runnable "multi-agent
      framework" to invoke directly -- the fenics-llm GitHub org has
      exactly two public repos (the results archive and the Jekyll
      site), neither containing agent orchestration code. This mode is
      therefore our own reconstruction of the pattern the paper
      describes ("agentic framework orchestrates multiple specialized
      agents... to formulate problems as PDEs, generate and debug code
      and visualize the results", embedded in "a multi-agent workflow
      with runtime feedback"):
        1. Formulator agent: restates the free-text problem as an
           explicit PDE / boundary-value spec.
        2. Coder agent: generates FEniCS code from that spec.
        3. Debugger agent: on failure, retries with the runtime error,
           up to --max-debug-turns times (default 3, vs. 1 for
           two-agent).
        4. Reviewer agent: sanity-checks the final numeric result in
           plain language (informational only -- does not change the
           PASS/FAIL verdict, which stays a fixed numeric comparison
           for reproducibility).
      You can point each stage at a different local model with
      --formulator-model / --debugger-model / --reviewer-model; all
      default to --model if not set.

IMPORTANT — read before trusting the verdict:
This is a deliberately simplified SURROGATE of the paper's scissor-mode
FE model, not a faithful reproduction. The real paper's model:
  - is a 3D solid with a jaw crossover + separate lever link (L7) that
    is hard to specify unambiguously in text,
  - uses three frictionless CONTACT regions between the flexure body
    and a rigid sheath to apply the actuation load (contact FE is
    outside what FEniCS/ALL-FEM handles as a standard building block),
  - doesn't state the beam width or E/nu for 316L stainless steel
    explicitly (I filled in a width assumption and standard handbook
    material properties below — both flagged inline).
This script instead asks the model to build a straight-chain multi-
segment beam (no crossover, no L7, no contact — moment and supports
applied directly as boundary conditions) using the paper's actual
segment lengths, angles, and thicknesses. Treat the comparison against
the paper's reported 0.654N as an order-of-magnitude sanity check, not
a precision validation — unlike the cantilever test, tight numerical
agreement isn't a reasonable bar here given the approximations.

REQUIREMENTS: same as test_allfem_cantilever.py (Ollama running
locally with the ALL-FEM model(s) pulled, and a FEniCS/dolfinx
environment on PATH as `python3`).

USAGE
    # reproduce the "Qwen 3 32B Finetuned" leaderboard condition
    python3 test_allfem_scissor.py \
        --model rushikesh_67/qwen3-short_think-fenics-local \
        --framework two-agent

    # approximate the "Multi-Agent framework" leaderboard condition
    python3 test_allfem_scissor.py \
        --model rushikesh_67/gpt-oss-finetuned \
        --framework multi-agent
"""

import argparse
import json
import re
import subprocess
import sys
import textwrap
import urllib.request

# ---------------------------------------------------------------------------
# 1. Geometry / material, taken from the paper where stated, assumed
#    where not (assumptions flagged).
# ---------------------------------------------------------------------------
PARAMS = dict(
    L1=3.0, L2=4.0, L3=8.0, L4=9.0, L5=2.0, L6=2.0,   # mm, from paper
    theta_deg=6.0, delta_deg=2.0,                      # degrees, from paper
    T1=1.0, T2=0.5,                                    # mm, from paper
    width_mm=1.0,                                      # mm, ASSUMED (not in paper)
    E_pa=193.0e9, nu=0.3,                               # 316L SS, ASSUMED standard values
    moment_N_mm=20.0,                                   # N*mm, from paper
)

REFERENCE_FORCE_N = 0.654  # paper's reported scissor/cutting reaction force,
                            # baseline geometry, WITH contact + real sheath —
                            # our surrogate omits both, see module docstring.

# The physics/geometry/BC description on its own (no "write code" framing),
# so it can be fed to the Formulator agent in multi-agent mode, or embedded
# directly in the Coder prompt in two-agent mode.
PROBLEM_STATEMENT = textwrap.dedent(f"""\
    A 3D linear elasticity problem for a multi-segment compliant beam
    mechanism (a simplified surrogate for a surgical forceps/scissors
    flexure) under an applied bending moment. Report the reaction force
    at the fixed end.

    Geometry: model the part as a single chain of 6 straight
    rectangular-cross-section beam segments joined end-to-end along a
    piecewise-linear centerline in the x-z plane, in this order,
    starting from the free (loaded) end and ending at the fixed (tip)
    end:
      - L1 = {PARAMS['L1']} mm, thickness T1 = {PARAMS['T1']} mm
      - L2 = {PARAMS['L2']} mm, thickness T1 = {PARAMS['T1']} mm,
        collinear with L1
      - L3 = {PARAMS['L3']} mm, thickness tapering linearly from
        T1 = {PARAMS['T1']} mm to T2 = {PARAMS['T2']} mm along its
        length, collinear with L1/L2
      - a kink of angle theta = {PARAMS['theta_deg']} degrees at the
        L3-L4 junction
      - L4 = {PARAMS['L4']} mm, thickness T2 = {PARAMS['T2']} mm
      - L5 = {PARAMS['L5']} mm, thickness T2 = {PARAMS['T2']} mm,
        collinear with L4
      - a kink of angle delta = {PARAMS['delta_deg']} degrees at the
        L5-L6 junction
      - L6 = {PARAMS['L6']} mm, thickness T2 = {PARAMS['T2']} mm

    All segments share a constant width w = {PARAMS['width_mm']} mm
    (out-of-plane, i.e. perpendicular to the x-z bending plane).

    Boundary conditions:
      - The far end of L6 (the tip) is fully fixed (zero displacement).
      - At the L2-L3 junction (representing where a support sheath
        holds the part), restrain displacement in the two directions
        transverse to the local beam axis at that point, leaving axial
        translation and rotation free (a roller/cylindrical-support-
        like constraint). If this is difficult to implement precisely,
        a reasonable simplification is acceptable -- state what you
        used in a code comment.

    Loading: apply a pure bending moment of {PARAMS['moment_N_mm']} N*mm
    ({PARAMS['moment_N_mm']/1000.0} N*m) at the free end of L1, about
    the axis perpendicular to the x-z plane (i.e. bending within the
    plane of the beam chain). Implement this as a Neumann boundary
    condition: a traction on the L1 end face that varies linearly
    across the face's thickness direction, with zero net force and a
    net moment equal to {PARAMS['moment_N_mm']/1000.0} N*m.

    Material: linear isotropic elasticity, 316L stainless steel,
    E = {PARAMS['E_pa']:.3e} Pa, nu = {PARAMS['nu']}.
    """)

CODE_INSTRUCTIONS = textwrap.dedent("""\
    Write a complete, runnable Python script using FEniCS (dolfin or
    dolfinx) that solves the following problem. Use a mesh with
    reasonable resolution given the small (mm-scale) geometry (note:
    convert all lengths to meters when building the mesh/solver so
    units are consistent with E in Pa). Solve the static linear
    elasticity variational problem, then compute and PRINT the
    magnitude of the total reaction force (in the direction transverse
    to the local beam axis) on the fixed end face (the L6 tip), in
    this exact format on its own line:

    SCISSOR_FORCE_N = <value>

    Return only the code, in a single Python code block.
    """)

# Original single-prompt form, used by the "two-agent" Coder stage.
PROMPT = CODE_INSTRUCTIONS + "\n" + PROBLEM_STATEMENT

FORMULATOR_INSTRUCTIONS = textwrap.dedent("""\
    You are the Formulator agent in a multi-agent FEA pipeline. Read
    the problem description below and restate it as a precise,
    unambiguous boundary-value problem specification: the governing
    PDE, the domain geometry (explicit segment lengths/angles/
    thicknesses and how they connect), all boundary conditions (exact
    location and type), and the loading (exact magnitude, direction,
    and how it's applied). Do not write any code. Be terse; use a
    numbered list.

    PROBLEM:
    """)

DEBUG_INSTRUCTIONS = textwrap.dedent("""\
    You are the Debugger agent in a multi-agent FEA pipeline. The
    following Python/FEniCS script was written to solve the problem
    below, but it failed. Fix the script so it runs successfully and
    still prints the required result line. Return the FULL corrected
    script in a single Python code block -- do not return a diff or a
    partial snippet.

    PROBLEM:
    {problem}

    PREVIOUS CODE:
    ```python
    {code}
    ```

    ERROR / OUTPUT FROM RUNNING THE PREVIOUS CODE (may be truncated):
    {error_tail}
    """)

REVIEWER_INSTRUCTIONS = textwrap.dedent("""\
    You are the Reviewer agent in a multi-agent FEA pipeline. A FEniCS
    simulation was run to estimate a reaction force for a compliant-
    beam scissor mechanism (a simplified surrogate; contact and a real
    sheath were omitted). Given the result below, write one short
    paragraph (3-5 sentences) sanity-checking whether it is physically
    plausible, noting any red flags (wrong order of magnitude, wrong
    sign, or a likely unit error) a reviewer should double check
    before trusting it. Do not write code.

    Literature reference value (different, more detailed model — not
    directly comparable): {reference} N
    Value produced by our surrogate FE model: {result}
    """)


# ---------------------------------------------------------------------------
# 2. Ollama call (identical pattern to test_allfem_cantilever.py)
# ---------------------------------------------------------------------------
def call_ollama(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"ERROR: could not reach Ollama at {host} ({e}).\n"
              f"Is `ollama serve` running, and is the model pulled?", file=sys.stderr)
        sys.exit(1)
    return body.get("response", "")


def extract_code(reply: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
    if match:
        return match.group(1).strip()
    return reply.strip()


# ---------------------------------------------------------------------------
# 3. Run generated code and parse its output
# ---------------------------------------------------------------------------
def run_generated_code(code: str, out_path: str = "generated_fenics_scissor_code.py"):
    with open(out_path, "w") as f:
        f.write(code)
    print(f"[saved generated code -> {out_path}]")

    try:
        result = subprocess.run(
            [sys.executable, out_path],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: generated script timed out after 600s.")
        return None, "TimeoutExpired: script did not finish within 600s."

    print("----- generated script stdout -----")
    print(result.stdout)
    combined = result.stdout
    if result.returncode != 0:
        print("----- generated script stderr -----")
        print(result.stderr)
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        return None, combined

    m = re.search(r"SCISSOR_FORCE_N\s*=\s*(-?[\d.eE+-]+)", result.stdout)
    if not m:
        print("WARNING: could not find 'SCISSOR_FORCE_N = <value>' in stdout.")
        return None, combined
    return float(m.group(1)), combined


# ---------------------------------------------------------------------------
# 4. Agent stages
# ---------------------------------------------------------------------------
def formulator_stage(model: str, host: str) -> str:
    reply = call_ollama(model, FORMULATOR_INSTRUCTIONS + PROBLEM_STATEMENT, host)
    print(f"\n[Formulator agent, model={model}] reply:\n{reply}")
    return reply.strip()


def coder_stage(model: str, host: str, spec_text: str) -> tuple:
    prompt = CODE_INSTRUCTIONS + "\n" + spec_text
    reply = call_ollama(model, prompt, host)
    code = extract_code(reply)
    print(f"\n[Coder agent, model={model}] extracted code:\n{code}")
    return code, prompt


def debugger_stage(model: str, host: str, problem_text: str, code: str, error_tail: str) -> str:
    prompt = DEBUG_INSTRUCTIONS.format(
        problem=problem_text, code=code, error_tail=error_tail[-4000:]
    )
    reply = call_ollama(model, prompt, host)
    new_code = extract_code(reply)
    print(f"\n[Debugger agent, model={model}] corrected code:\n{new_code}")
    return new_code


def reviewer_stage(model: str, host: str, force_fem):
    result_str = "FAIL (no valid value produced)" if force_fem is None else f"{force_fem:.4f} N"
    prompt = REVIEWER_INSTRUCTIONS.format(reference=REFERENCE_FORCE_N, result=result_str)
    reply = call_ollama(model, prompt, host)
    print(f"\n[Reviewer agent, model={model}] assessment:\n{reply.strip()}")
    return reply.strip()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rushikesh_67/llama3.2-2new",
                     help="Default Ollama model tag for every agent stage "
                          "(e.g. rushikesh_67/qwen3-short_think-fenics-local "
                          "or rushikesh_67/gpt-oss-finetuned)")
    ap.add_argument("--framework", choices=["two-agent", "multi-agent"],
                     default="two-agent",
                     help="two-agent: Coder + one Debugger fix attempt "
                          "(matches the per-model leaderboard results). "
                          "multi-agent: Formulator + Coder + up-to-N "
                          "Debugger retries + Reviewer (local approximation "
                          "of the leaderboard's Multi-Agent framework "
                          "condition -- see module docstring).")
    ap.add_argument("--max-debug-turns", type=int, default=None,
                     help="Override number of debug/fix retries "
                          "(default: 1 for two-agent, 3 for multi-agent)")
    ap.add_argument("--formulator-model", default=None,
                     help="Model for the Formulator stage (multi-agent "
                          "only); defaults to --model")
    ap.add_argument("--debugger-model", default=None,
                     help="Model for the Debugger stage; defaults to --model")
    ap.add_argument("--reviewer-model", default=None,
                     help="Model for the Reviewer stage (multi-agent "
                          "only); defaults to --model")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--skip-run", action="store_true",
                     help="Only run the Formulator/Coder stage(s) and print "
                          "the result, don't execute the generated code or "
                          "run any Debugger/Reviewer stages")
    args = ap.parse_args()

    formulator_model = args.formulator_model or args.model
    debugger_model = args.debugger_model or args.model
    reviewer_model = args.reviewer_model or args.model
    max_debug_turns = args.max_debug_turns
    if max_debug_turns is None:
        max_debug_turns = 3 if args.framework == "multi-agent" else 1

    print("=" * 70)
    print(f"Model under test: {args.model}")
    print(f"Framework:        {args.framework}  (max debug turns: {max_debug_turns})")
    print("Test: ALL-FEM scissor/cutting-mode surrogate "
          "(Libu George & Bharanidaran 2020)")
    print("=" * 70)

    # --- Formulator stage (multi-agent only) -------------------------------
    if args.framework == "multi-agent":
        formulation = formulator_stage(formulator_model, args.host)
        spec_text = (
            "FORMAL SPECIFICATION (from the Formulator agent):\n"
            + formulation
            + "\n\nORIGINAL PROBLEM STATEMENT (for reference):\n"
            + PROBLEM_STATEMENT
        )
    else:
        spec_text = PROBLEM_STATEMENT

    # --- Coder stage ---------------------------------------------------------
    code, _ = coder_stage(args.model, args.host, spec_text)

    print(f"\nReference (paper, baseline geometry, WITH contact + real "
          f"sheath): {REFERENCE_FORCE_N} N")
    print("Our surrogate omits contact and the physical sheath -- see the "
          "module docstring. Treat this as an order-of-magnitude sanity "
          "check, not a precision validation.")

    if args.skip_run:
        with open("generated_fenics_scissor_code.py", "w") as f:
            f.write(code)
        print("[--skip-run] wrote generated_fenics_scissor_code.py, not executing.")
        return

    # --- Run + Debugger loop --------------------------------------------------
    turns_used = 0
    force_fem, output = run_generated_code(code)
    while force_fem is None and turns_used < max_debug_turns:
        turns_used += 1
        print(f"\n--- Debugger turn {turns_used}/{max_debug_turns} ---")
        code = debugger_stage(debugger_model, args.host, spec_text, code, output)
        force_fem, output = run_generated_code(code)

    # --- Reviewer stage (multi-agent only) ------------------------------------
    if args.framework == "multi-agent":
        reviewer_stage(reviewer_model, args.host, force_fem)

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"Framework:      {args.framework}")
    print(f"Debugger turns used: {turns_used}/{max_debug_turns}")
    if force_fem is None:
        print("FAIL: generated code did not run successfully or did not "
              "print a parseable SCISSOR_FORCE_N value, even after the "
              "Debugger stage(s).")
        sys.exit(1)

    print(f"FEM reaction force (surrogate model): {force_fem:.4f} N")
    print(f"Paper reference (real model):         {REFERENCE_FORCE_N} N")

    if force_fem <= 0:
        verdict = "FAIL — non-positive or zero reaction force; model is not transmitting the moment correctly."
    else:
        ratio = force_fem / REFERENCE_FORCE_N
        print(f"Ratio to reference:                   {ratio:.2f}x")
        if 0.2 <= ratio <= 5.0:
            verdict = "PLAUSIBLE — within an order of magnitude of the paper's reported value, given the surrogate's approximations."
        elif 0.05 <= ratio <= 20.0:
            verdict = "QUESTIONABLE — positive but notably off; inspect the generated moment BC and geometry for unit/setup errors."
        else:
            verdict = "FAIL — result is wildly different (likely a units error, e.g. mm vs m, or a broken moment BC)."
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
