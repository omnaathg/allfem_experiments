#!/usr/bin/env python3
"""
test_allfem_scissor.py

ALL-FEM test #2: the "scissor/cutting mode" sub-analysis from

  Libu George B & R. Bharanidaran, "Design of multifunctional compliant
  forceps for medical application," Australian Journal of Mechanical
  Engineering 20(3), 2020. DOI: 10.1080/14484846.2020.1747151

Same harness pattern as test_allfem_cantilever.py: prompt a local Ollama
model, extract the generated FEniCS code, run it, compare the printed
result against a reference value.

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
locally with the ALL-FEM model pulled, and a FEniCS/dolfinx
environment on PATH as `python3`).

USAGE
    python3 test_allfem_scissor.py --model rushikesh_67/llama3.2-2new
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

PROMPT = textwrap.dedent(f"""\
    Write a complete, runnable Python script using FEniCS (dolfin or
    dolfinx) that solves a 3D linear elasticity problem for a
    multi-segment compliant beam mechanism (a simplified surrogate for
    a surgical forceps/scissors flexure) under an applied bending
    moment, and reports the reaction force at the fixed end.

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

    Use a mesh with reasonable resolution given the small (mm-scale)
    geometry (note: convert all lengths to meters when building the
    mesh/solver so units are consistent with E in Pa). Solve the
    static linear elasticity variational problem, then compute and
    PRINT the magnitude of the total reaction force (in the direction
    transverse to the local beam axis) on the fixed end face (the L6
    tip), in this exact format on its own line:

    SCISSOR_FORCE_N = <value>

    Return only the code, in a single Python code block.
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
        return None, ""

    print("----- generated script stdout -----")
    print(result.stdout)
    if result.returncode != 0:
        print("----- generated script stderr -----")
        print(result.stderr)
        return None, result.stdout

    m = re.search(r"SCISSOR_FORCE_N\s*=\s*(-?[\d.eE+-]+)", result.stdout)
    if not m:
        print("WARNING: could not find 'SCISSOR_FORCE_N = <value>' in stdout.")
        return None, result.stdout
    return float(m.group(1)), result.stdout


# ---------------------------------------------------------------------------
# 4. Main
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
    print("Test: ALL-FEM scissor/cutting-mode surrogate "
          "(Libu George & Bharanidaran 2020)")
    print("=" * 70)
    print("\nPROMPT SENT:\n" + PROMPT)

    reply = call_ollama(args.model, PROMPT, args.host)
    print("\nRAW MODEL REPLY:\n" + reply)

    code = extract_code(reply)
    print("\nEXTRACTED CODE:\n" + code)

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

    force_fem, stdout = run_generated_code(code)

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    if force_fem is None:
        print("FAIL: generated code did not run successfully or did not "
              "print a parseable SCISSOR_FORCE_N value.")
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
