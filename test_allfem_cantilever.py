#!/usr/bin/env python3
"""
test_allfem_cantilever.py

A self-contained harness for trying an ALL-FEM fine-tuned model
(https://fenics-llm.github.io/) on a concrete FEA problem and checking
whether the code it generates actually produces a physically sensible
answer.

WHAT IT DOES
1. Sends a fixed prompt (a classic clamped-beam-under-gravity linear
   elasticity problem) to a locally running Ollama model.
2. Extracts the Python/FEniCS code block from the model's reply.
3. Executes that code in a subprocess.
4. Parses the tip deflection the generated script prints out.
5. Compares it against the closed-form Euler-Bernoulli beam-theory
   deflection, so you get a quantitative sense of whether the model's
   FEA solution is in the right ballpark (not just whether the code runs).

REQUIREMENTS (install on your own machine, not in this sandbox):
  - Ollama installed and running: https://ollama.com/download
  - The ALL-FEM model pulled, e.g.:
        ollama pull rushikesh_67/llama3.2-2new          # 3B, fastest
        ollama pull rushikesh_67/qwen3-short_think-fenics-local
        ollama pull rushikesh_67/llama3.3-fenics-3new
        ollama pull rushikesh_67/gpt-oss-finetuned       # 120B, best reported accuracy
  - A working FEniCS/FEniCSx environment on PATH as `python3`
    (e.g. `conda install -c conda-forge fenics-dolfinx mpich pyvista` or
    the legacy `fenics` package). The generated code's import style
    (dolfin vs dolfinx) depends on what the model was trained to emit —
    check the corpus/model card before picking an environment.

USAGE
    python3 test_allfem_cantilever.py --model rushikesh_67/llama3.2-2new
"""

import argparse
import json
import re
import subprocess
import sys
import textwrap
import urllib.request

# ---------------------------------------------------------------------------
# 1. The test prompt
# ---------------------------------------------------------------------------
# Classic "clamped elastic beam under self-weight" problem (same benchmark
# used in the well-known FEniCS tutorial by Langtangen & Logg). It's a good
# probe because it has a simple, well-known closed-form check.

BEAM_PARAMS = dict(
    L=1.0,       # length (m)
    W=0.2,       # cross-section width (m)
    H=0.2,       # cross-section height (m)
    E=1.0e5,     # Young's modulus (Pa)
    nu=0.3,      # Poisson's ratio
    rho=1.0,     # density (kg/m^3)
    g=0.16,      # gravitational acceleration used in the tutorial (m/s^2)
)

PROMPT = textwrap.dedent(f"""\
    Write a complete, runnable Python script using FEniCS (dolfin or
    dolfinx) that solves a 3D linear elasticity problem for a clamped
    elastic beam under its own weight (gravity loading).

    Geometry: a box domain [0, {BEAM_PARAMS['L']}] x [0, {BEAM_PARAMS['W']}]
    x [0, {BEAM_PARAMS['H']}] meters (length x width x height).

    Boundary conditions: the face at x = 0 is fully clamped (zero
    displacement). All other faces are traction-free.

    Loading: self-weight only, i.e. a body force f = (0, 0, -rho*g) with
    rho = {BEAM_PARAMS['rho']} kg/m^3 and g = {BEAM_PARAMS['g']} m/s^2.

    Material: linear isotropic elasticity with Young's modulus
    E = {BEAM_PARAMS['E']:.1e} Pa and Poisson's ratio nu = {BEAM_PARAMS['nu']}.

    Use a structured mesh (e.g. BoxMesh) with reasonable resolution.
    Solve the variational problem, then evaluate and PRINT the vertical
    (z) displacement at the free end centerline point
    ({BEAM_PARAMS['L']}, {BEAM_PARAMS['W']/2}, {BEAM_PARAMS['H']/2})
    in this exact format on its own line:

    TIP_DEFLECTION_M = <value>

    Return only the code, in a single Python code block.
    """)

# ---------------------------------------------------------------------------
# 2. Analytical reference (Euler-Bernoulli beam theory)
# ---------------------------------------------------------------------------
def analytical_tip_deflection(p=BEAM_PARAMS):
    """Closed-form tip deflection for a cantilever under uniform self-weight.

    q = rho*g*A  (distributed load per unit length, A = W*H)
    I = W*H^3/12 (second moment of area)
    delta = q*L^4 / (8*E*I) = 3*rho*g*L^4 / (2*E*H^2)
    """
    L, W, H, E, rho, g = p["L"], p["W"], p["H"], p["E"], p["rho"], p["g"]
    return -3.0 * rho * g * L**4 / (2.0 * E * H**2)


# ---------------------------------------------------------------------------
# 3. Ollama call
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
    # No fenced block found — assume the whole reply is code.
    return reply.strip()


# ---------------------------------------------------------------------------
# 4. Run generated code and parse its output
# ---------------------------------------------------------------------------
def run_generated_code(code: str, out_path: str = "generated_fenics_code.py"):
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
        return None, ""

    print("----- generated script stdout -----")
    print(result.stdout)
    if result.returncode != 0:
        print("----- generated script stderr -----")
        print(result.stderr)
        return None, result.stdout

    m = re.search(r"TIP_DEFLECTION_M\s*=\s*(-?[\d.eE+-]+)", result.stdout)
    if not m:
        print("WARNING: could not find 'TIP_DEFLECTION_M = <value>' in stdout.")
        return None, result.stdout
    return float(m.group(1)), result.stdout


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rushikesh_67/llama3.2-2new",
                     help="Ollama model tag to test (default: 3B fine-tuned model)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--skip-run", action="store_true",
                     help="Only print the prompt + generated code, don't execute it")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Model under test: {args.model}")
    print("=" * 70)
    print("\nPROMPT SENT:\n" + PROMPT)

    reply = call_ollama(args.model, PROMPT, args.host)
    print("\nRAW MODEL REPLY:\n" + reply)

    code = extract_code(reply)
    print("\nEXTRACTED CODE:\n" + code)

    delta_analytical = analytical_tip_deflection()
    print(f"\nAnalytical (Euler-Bernoulli) tip deflection: {delta_analytical:.6e} m")

    if args.skip_run:
        with open("generated_fenics_code.py", "w") as f:
            f.write(code)
        print("[--skip-run] wrote generated_fenics_code.py, not executing.")
        return

    delta_fem, stdout = run_generated_code(code)

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    if delta_fem is None:
        print("FAIL: generated code did not run successfully or did not "
              "print a parseable TIP_DEFLECTION_M value.")
        sys.exit(1)

    rel_err = abs((delta_fem - delta_analytical) / delta_analytical) * 100
    same_sign = (delta_fem < 0) == (delta_analytical < 0)
    print(f"FEM tip deflection:         {delta_fem:.6e} m")
    print(f"Analytical tip deflection:  {delta_analytical:.6e} m")
    print(f"Relative error:             {rel_err:.1f}%")
    print(f"Correct sign (downward)?    {same_sign}")

    if not same_sign:
        print("\nVERDICT: FAIL — deflection direction is wrong.")
    elif rel_err < 15:
        print("\nVERDICT: PASS — within expected FEM-vs-beam-theory tolerance (<15%).")
    elif rel_err < 50:
        print("\nVERDICT: MARGINAL — right order of magnitude but notably off; "
              "inspect the generated mesh/material setup.")
    else:
        print("\nVERDICT: FAIL — result is off by more than 50%; likely a "
              "modeling or unit error in the generated code.")


if __name__ == "__main__":
    main()
