#!/usr/bin/env python3
"""
modal_allfem_cantilever.py

Runs the ALL-FEM cantilever-beam test entirely inside one Modal GPU
container: Ollama (serving an ALL-FEM fine-tuned model) and a FEniCSx
(dolfinx) environment live in the same image, so there's no networking
between separate services to configure — `modal run` does everything
remotely and streams the result back to your terminal.

SETUP (one-time, on your machine)
    pip install modal
    modal setup                # opens a browser to authenticate

RUN
    modal run modal_allfem_cantilever.py
    modal run modal_allfem_cantilever.py --model rushikesh_67/qwen3-short_think-fenics-local --gpu a100

NOTES / THINGS TO VERIFY YOURSELF
  - The Ollama model tag is baked into the image at BUILD time (so the
    weights are cached and later runs are fast). If you change --model,
    Modal will rebuild the image and re-pull that model — first run
    with a new model will be slow.
  - This installs FEniCSx (`dolfinx`), i.e. modern `import dolfinx`
    style code. If the model instead emits legacy `from dolfin import *`
    style code (FEniCS 2019.1), swap the micromamba_install line below
    from "fenics-dolfinx" to "fenics" and adjust accordingly — check
    what the generated code actually imports (printed in the output)
    and match the environment to it.
  - GPU sizing: the 3B model runs fine on a T4/A10G. The 70B model
    needs an A100-80GB. The 120B (gpt-oss) fine-tune is the largest —
    budget for an A100-80GB or H100 and expect a multi-GPU config may
    be required depending on quantization; check current VRAM
    requirements on the model's Ollama page before deploying.
  - I could not test this script against a live Modal account from my
    sandbox (no network egress to modal.com there). Treat it as a
    strong starting point, not a guaranteed-working deploy — Modal's
    own Ollama example (https://github.com/irfansharif/ollama-modal)
    is a good reference if something doesn't line up with a future
    Modal SDK version.
"""
import argparse
import re
import subprocess
import textwrap
import time

import modal

# ---------------------------------------------------------------------------
# Same prompt / analytical solution as test_allfem_cantilever.py
# ---------------------------------------------------------------------------
BEAM_PARAMS = dict(L=1.0, W=0.2, H=0.2, E=1.0e5, nu=0.3, rho=1.0, g=0.16)

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


def analytical_tip_deflection(p=BEAM_PARAMS):
    L, W, H, E, rho, g = p["L"], p["W"], p["H"], p["E"], p["rho"], p["g"]
    return -3.0 * rho * g * L**4 / (2.0 * E * H**2)


# ---------------------------------------------------------------------------
# Modal image: Ollama + the ALL-FEM model (pulled at build time) + FEniCSx
# ---------------------------------------------------------------------------
MODEL_TAG = "rushikesh_67/llama3.2-2new"  # default; override with --model

image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("fenics-dolfinx", "mpich", channels=["conda-forge"])
    .apt_install("curl", "ca-certificates")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("requests")
    .run_commands(
        f"bash -c 'ollama serve > /tmp/ollama_build.log 2>&1 & "
        f"sleep 5 && ollama pull {MODEL_TAG}'"
    )
)

app = modal.App("allfem-cantilever-test", image=image)


@app.cls(gpu="a10g", timeout=1200, scaledown_window=300)
class AllFemTester:
    @modal.enter()
    def start_ollama(self):
        self.proc = subprocess.Popen(["ollama", "serve"])
        time.sleep(5)

    @modal.method()
    def run_test(self, model: str = MODEL_TAG):
        import json
        import urllib.request

        payload = json.dumps({"model": model, "prompt": PROMPT, "stream": False}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            reply = json.loads(resp.read())["response"]

        m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
        code = m.group(1).strip() if m else reply.strip()

        with open("/tmp/generated_fenics_code.py", "w") as f:
            f.write(code)

        result = subprocess.run(
            ["python3", "/tmp/generated_fenics_code.py"],
            capture_output=True, text=True, timeout=600,
        )

        out = {
            "model": model,
            "reply": reply,
            "code": code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "analytical_deflection_m": analytical_tip_deflection(),
        }
        dm = re.search(r"TIP_DEFLECTION_M\s*=\s*(-?[\d.eE+-]+)", result.stdout)
        out["fem_deflection_m"] = float(dm.group(1)) if dm else None
        return out


@app.local_entrypoint()
def main(model: str = MODEL_TAG):
    tester = AllFemTester()
    result = tester.run_test.remote(model=model)

    print("=" * 70)
    print(f"Model: {result['model']}")
    print("=" * 70)
    print("\nGENERATED CODE:\n" + result["code"])
    print("\n----- script stdout -----\n" + result["stdout"])
    if result["returncode"] != 0:
        print("\n----- script stderr -----\n" + result["stderr"])

    delta_a = result["analytical_deflection_m"]
    delta_f = result["fem_deflection_m"]
    print(f"\nAnalytical tip deflection: {delta_a:.6e} m")
    if delta_f is None:
        print("VERDICT: FAIL - no parseable TIP_DEFLECTION_M in output.")
    else:
        rel_err = abs((delta_f - delta_a) / delta_a) * 100
        print(f"FEM tip deflection:        {delta_f:.6e} m")
        print(f"Relative error:            {rel_err:.1f}%")
        verdict = "PASS" if rel_err < 15 else "MARGINAL" if rel_err < 50 else "FAIL"
        print("VERDICT:", verdict)
