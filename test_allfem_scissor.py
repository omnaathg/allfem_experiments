#!/usr/bin/env python3
"""
test_allfem_scissor.py

ALL-FEM test #2: the "scissor/cutting mode" sub-analysis from

  Libu George B & R. Bharanidaran, "Design of multifunctional compliant
  forceps for medical application," Australian Journal of Mechanical
  Engineering 20(3), 2020. DOI: 10.1080/14484846.2020.1747151

This harness supports two agent-orchestration modes, mirroring the two
evaluation conditions on the ALL-FEM results leaderboard
(https://fenics-llm.github.io/Results/):

  --framework two-agent   (default)
      Coder agent generates code; if it fails to run, a Debugger agent
      gets one shot at fixing it using the runtime error. This is how
      the per-model results (e.g. "Qwen 3 32B Finetuned") on the
      leaderboard were produced -- pass --model to pick which of your
      local Ollama tags plays that role.

  --framework multi-agent
      Models the paper's actual "Multi agent workflow" (Figure 4 of the
      ALL-FEM paper), which the user supplied directly since the
      fenics-llm GitHub org does not publish the orchestration source
      (its Results_ALL-FEM repo only has final generated
      code/output/plots per benchmark, prefixed "oss-multi-", i.e. run
      with GPT-OSS 120B). Figure 4 shows:

        - a shared GROUP CHAT transcript that all agents read/post to
        - a COORDINATOR (group-chat-manager agent, GPT-OSS) that
          decides who speaks next, choosing only among the "eligible"
          agents: Formulator, Planner, FEniCS Coder, Evaluator
        - FORMULATOR + PLANNER (GPT-OSS, assistant agents) restate the
          problem as a PDE/BC spec and a numerical solution strategy
        - a FEniCS CODER + CORRECTOR subsystem (both GPT-OSS-FT, i.e.
          the *fine-tuned* checkpoint, unlike the other roles): Coder
          writes code, Corrector reviews/fixes it, an EXECUTOR (a
          non-LLM "user proxy" tool-runner) runs it, and on an
          execution error control returns to the Coder (not the
          Corrector) to regenerate -- this loop repeats up to
          --max-debug-turns times
        - an EVALUATOR (GPT-OSS) assesses the final result and an
          ADMIN (non-LLM user-proxy gate) combines the Executor's
          success/failure with the Evaluator's assessment into the
          final verdict before EXIT

      This script implements that graph faithfully in terms of agent
      roles, model-tier split (fine-tuned vs. base), and eligibility
      (Corrector/Executor/Admin are never directly dispatched by the
      Coordinator -- they only run inside the Coder subsystem or at
      the very end). The actual per-role prompts and the Coordinator's
      routing logic are our own reconstruction, since those aren't
      published -- only the architecture diagram is.

      --coordinator-mode fixed    (default)
          The Coordinator's dispatch order is hardcoded to
          Formulator -> Planner -> [Coder<->Corrector<->Executor loop]
          -> Evaluator -> Admin -> exit, matching the paper's typical
          path. No LLM call is spent on routing. Much easier to
          reproduce and debug on small local models.

      --coordinator-mode dynamic
          The Coordinator is an actual LLM call: each turn it reads
          the group-chat transcript so far and picks the next agent
          from {Formulator, Planner, FEniCS Coder, Evaluator}, exactly
          as in Figure 4. Bounded by --max-coordinator-turns (default
          8) as a safety cap, since small local models can route
          unreliably; if the cap is hit before the Coder subsystem or
          Evaluator have run, they're run anyway before Admin's final
          call so every result is still comparable.

      Model tiers (mirrors GPT-OSS vs. GPT-OSS-FT in Figure 4):
        --model             fine-tuned tag under test; default for
                             --coder-model / --corrector-model
        --base-model        "reasoning" tag for the non-coding agents;
                             default for --coordinator-model /
                             --formulator-model / --planner-model /
                             --evaluator-model. If you don't pass this,
                             it falls back to --model too (with a
                             printed note) -- for a faithful split, pass
                             a separate, non-fine-tuned local tag here.
        Any of the six --*-model flags can still be set individually.

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
This script instead asks the model(s) to build a straight-chain multi-
segment beam (no crossover, no L7, no contact — moment and supports
applied directly as boundary conditions) using the paper's actual
segment lengths, angles, and thicknesses. Treat the comparison against
the paper's reported 0.654N as an order-of-magnitude sanity check, not
a precision validation — unlike the cantilever test, tight numerical
agreement isn't a reasonable bar here given the approximations.

GEOMETRY INPUT IS TEXT-ONLY -- this is a deliberate, noted limitation,
not an oversight. The model is never shown the paper's actual figures
(e.g. Figure 3b's sheath assembly, with its separate moving/fixed parts
and cylindrical support) or any CAD/mesh file. Every geometric and
boundary-condition detail below -- segment lengths, kink angles,
thickness taper, support locations -- is conveyed purely as prose in
PROBLEM_STATEMENT, and the model (none of the four ALL-FEM Ollama tags
are vision-capable) has to translate that description into mesh-
building and BC code with no visual reference at all. This is consistent
with how the ALL-FEM paper itself poses its 39 benchmarks (text-to-
FEniCS, not diagram-to-FEniCS), so it's kept as-is here rather than
switching to multimodal prompting or a structured/parametric geometry
input -- but it's a real source of ambiguity on top of the surrogate
simplifications above, and worth keeping in mind when reading a FAIL as
"the model can't do FEA" vs. "the model guessed wrong from a paragraph."

REQUIREMENTS: same as test_allfem_cantilever.py (Ollama running
locally with the ALL-FEM model(s) pulled, and a FEniCS/dolfinx
environment on PATH as `python3`).

USAGE
    # reproduce the "Qwen 3 32B Finetuned" leaderboard condition
    python3 test_allfem_scissor.py \
        --model rushikesh_67/qwen3-short_think-fenics-local \
        --framework two-agent

    # multi-agent, fixed dispatch order, one fine-tuned model doing all
    # coding roles and itself standing in for the base-model roles too
    python3 test_allfem_scissor.py \
        --model rushikesh_67/gpt-oss-finetuned \
        --framework multi-agent

    # multi-agent, dynamic Coordinator routing, faithful model-tier split
    python3 test_allfem_scissor.py \
        --model rushikesh_67/gpt-oss-finetuned \
        --base-model llama3.3:70b \
        --framework multi-agent --coordinator-mode dynamic
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
# so it can be fed to the Formulator agent, or embedded directly in the
# Coder prompt in two-agent mode.
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

# ---------------------------------------------------------------------------
# 1b. Agent prompts -- two-agent mode (Coder + Debugger, unchanged from the
#     original harness)
# ---------------------------------------------------------------------------
DEBUG_INSTRUCTIONS = textwrap.dedent("""\
    You are the Debugger agent in a two-agent FEA pipeline. The
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

# ---------------------------------------------------------------------------
# 1c. Agent prompts -- multi-agent mode (Figure 4: Coordinator, Formulator,
#     Planner, FEniCS Coder, Corrector, Executor, Evaluator, Admin)
# ---------------------------------------------------------------------------
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

PLANNER_INSTRUCTIONS = textwrap.dedent("""\
    You are the Planner agent in a multi-agent FEA pipeline. Given the
    formal specification below (produced by the Formulator agent),
    propose a concrete numerical solution strategy: what function
    space / element degree to use, how to build the mesh for this
    piecewise-linear, tapering geometry, how to represent the kinks
    and the thickness taper, how to implement the moment boundary
    condition numerically, and any solver settings worth specifying.
    Do not write code. Be terse; use a numbered list.

    FORMAL SPECIFICATION:
    {spec}
    """)

CODER_INSTRUCTIONS_MA = textwrap.dedent("""\
    You are the FEniCS Coder agent in a multi-agent FEA pipeline.
    Using the specification and solution plan below, write a complete,
    runnable Python script using FEniCS (dolfin or dolfinx) that
    solves the problem and PRINTS the magnitude of the total reaction
    force on the fixed end face in this exact format on its own line:

    SCISSOR_FORCE_N = <value>

    Return only the code, in a single Python code block.

    FORMAL SPECIFICATION:
    {spec}

    SOLUTION PLAN:
    {plan}
    """)

CODER_RETRY_INSTRUCTIONS = textwrap.dedent("""\
    You are the FEniCS Coder agent in a multi-agent FEA pipeline. Your
    previous script failed when executed (after already passing
    through the Corrector agent). Fix it and return the FULL corrected
    script in a single Python code block -- do not return a diff or a
    partial snippet.

    FORMAL SPECIFICATION:
    {spec}

    PREVIOUS CODE:
    ```python
    {code}
    ```

    ERROR / OUTPUT FROM RUNNING THE PREVIOUS CODE (may be truncated):
    {error_tail}
    """)

CORRECTOR_INSTRUCTIONS = textwrap.dedent("""\
    You are the Corrector agent in a multi-agent FEA pipeline. Review
    the following FEniCS/Python script for bugs, API misuse (dolfin vs
    dolfinx mismatches), unit errors, and missing pieces BEFORE it is
    executed. Fix whatever needs fixing; if it already looks correct,
    return it unchanged. Return the FULL script in a single Python
    code block -- do not return a diff or a partial snippet.

    FORMAL SPECIFICATION:
    {spec}

    SCRIPT TO REVIEW:
    ```python
    {code}
    ```
    """)

EVALUATOR_INSTRUCTIONS = textwrap.dedent("""\
    You are the Evaluator agent in a multi-agent FEA pipeline. A
    FEniCS simulation was run to estimate a reaction force for a
    compliant-beam scissor mechanism (a simplified surrogate; contact
    and a real sheath were omitted). Given the result below, write one
    short paragraph (3-5 sentences) sanity-checking whether it is
    physically plausible, noting any red flags (wrong order of
    magnitude, wrong sign, or a likely unit error) that Admin should
    know about before accepting it. Do not write code.

    Literature reference value (different, more detailed model — not
    directly comparable): {reference} N
    Value produced by our surrogate FE model: {result}
    """)

COORDINATOR_INSTRUCTIONS = textwrap.dedent("""\
    You are the Coordinator (group chat manager) in a multi-agent FEA
    pipeline. Below is the conversation so far. Decide which agent
    should speak next. You may ONLY choose from this exact list:
    Formulator, Planner, FEniCS Coder, Evaluator, Admin.

    Rules: don't pick an agent that has already spoken (see "agents
    already used" below) unless nothing else is left to pick. Only
    pick Admin once both "FEniCS Coder" and "Evaluator" appear in
    "agents already used". Reply with ONLY the agent name on its own
    line, nothing else -- no punctuation, no explanation.

    AGENTS ALREADY USED THIS RUN: {used}

    CONVERSATION SO FAR:
    {transcript}
    """)

COORDINATOR_ELIGIBLE = ["Formulator", "Planner", "FEniCS Coder", "Evaluator", "Admin"]


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
# 4a. Two-agent stages (Coder, Debugger) -- unchanged behavior
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4b. Multi-agent stages (Figure 4)
# ---------------------------------------------------------------------------
class GroupChat:
    """Minimal stand-in for the shared 'group chat' transcript in Figure 4."""

    def __init__(self):
        self.turns = []

    def post(self, speaker: str, text: str):
        self.turns.append((speaker, text))
        print(f"\n[group chat] {speaker} posted "
              f"({len(text)} chars, showing first 500):\n{text[:500]}")

    def render(self, max_chars: int = 6000) -> str:
        rendered = "\n\n".join(f"[{spk}]: {txt}" for spk, txt in self.turns)
        return rendered[-max_chars:]


def formulator_stage(model: str, host: str, chat: GroupChat) -> str:
    reply = call_ollama(model, FORMULATOR_INSTRUCTIONS + PROBLEM_STATEMENT, host)
    chat.post("Formulator", reply.strip())
    return reply.strip()


def planner_stage(model: str, host: str, chat: GroupChat, spec: str) -> str:
    reply = call_ollama(model, PLANNER_INSTRUCTIONS.format(spec=spec), host)
    chat.post("Planner", reply.strip())
    return reply.strip()


def corrector_stage(model: str, host: str, spec: str, code: str) -> str:
    reply = call_ollama(model, CORRECTOR_INSTRUCTIONS.format(spec=spec, code=code), host)
    new_code = extract_code(reply)
    print(f"\n[Corrector agent, model={model}] reviewed code:\n{new_code}")
    return new_code


def evaluator_stage(model: str, host: str, chat: GroupChat, force_fem) -> str:
    result_str = "FAIL (no valid value produced)" if force_fem is None else f"{force_fem:.4f} N"
    reply = call_ollama(
        model, EVALUATOR_INSTRUCTIONS.format(reference=REFERENCE_FORCE_N, result=result_str), host
    )
    chat.post("Evaluator", reply.strip())
    return reply.strip()


def admin_stage(force_fem, evaluator_text: str, turns_used: int, max_debug_turns: int) -> str:
    """Admin is a user-proxy/gate node in Figure 4 -- deterministic control
    flow, not an LLM call, combining the Executor's outcome with the
    Evaluator's assessment into the final verdict."""
    print("\n[Admin] Executor result: "
          f"{'FAIL' if force_fem is None else f'{force_fem:.4f} N'} "
          f"(after {turns_used}/{max_debug_turns} Coder<->Corrector<->Executor retries)")
    print(f"[Admin] Evaluator assessment:\n{evaluator_text}")

    if force_fem is None:
        return ("FAIL — generated code did not run successfully or did not "
                "print a parseable SCISSOR_FORCE_N value, even after the "
                "Coder<->Corrector<->Executor retry loop.")
    if force_fem <= 0:
        return "FAIL — non-positive or zero reaction force; model is not transmitting the moment correctly."
    ratio = force_fem / REFERENCE_FORCE_N
    print(f"[Admin] Ratio to reference: {ratio:.2f}x")
    if 0.2 <= ratio <= 5.0:
        return "PLAUSIBLE — within an order of magnitude of the paper's reported value, given the surrogate's approximations."
    elif 0.05 <= ratio <= 20.0:
        return "QUESTIONABLE — positive but notably off; inspect the generated moment BC and geometry for unit/setup errors."
    return "FAIL — result is wildly different (likely a units error, e.g. mm vs m, or a broken moment BC)."


def run_coder_subsystem(coder_model: str, corrector_model: str, host: str,
                         spec: str, plan: str, max_debug_turns: int):
    """The Coder<->Corrector<->Executor triangle in Figure 4: Coder writes
    code, Corrector reviews/fixes it BEFORE execution, Executor runs it,
    and on an execution error control returns to the Coder (not the
    Corrector) to regenerate. Dispatched by the Coordinator as one unit."""
    prompt = CODER_INSTRUCTIONS_MA.format(spec=spec, plan=plan)
    reply = call_ollama(coder_model, prompt, host)
    code = extract_code(reply)
    print(f"\n[FEniCS Coder agent, model={coder_model}] extracted code:\n{code}")

    code = corrector_stage(corrector_model, host, spec, code)

    turns_used = 0
    force_fem, output = run_generated_code(code)
    while force_fem is None and turns_used < max_debug_turns:
        turns_used += 1
        print(f"\n--- Executor reported an error; back to FEniCS Coder "
              f"(retry {turns_used}/{max_debug_turns}) ---")
        retry_prompt = CODER_RETRY_INSTRUCTIONS.format(
            spec=spec, code=code, error_tail=output[-4000:]
        )
        reply = call_ollama(coder_model, retry_prompt, host)
        code = extract_code(reply)
        print(f"\n[FEniCS Coder agent, model={coder_model}] corrected code:\n{code}")
        code = corrector_stage(corrector_model, host, spec, code)
        force_fem, output = run_generated_code(code)

    return code, force_fem, output, turns_used


def coordinator_stage(model: str, host: str, chat: GroupChat, used: list) -> str:
    """The dynamic Coordinator: an actual LLM call that reads the group
    chat transcript and picks the next agent from the eligible pool
    (Formulator, Planner, FEniCS Coder, Evaluator, Admin -- exactly the
    'yellow' agents in Figure 4; Corrector/Executor are never directly
    dispatched, they only run inside the Coder subsystem)."""
    prompt = COORDINATOR_INSTRUCTIONS.format(
        used=", ".join(used) if used else "(none yet)",
        transcript=chat.render(),
    )
    reply = call_ollama(model, prompt, host).strip()
    print(f"\n[Coordinator agent, model={model}] chose: {reply!r}")
    for candidate in COORDINATOR_ELIGIBLE:
        if candidate.lower() in reply.lower():
            return candidate
    print("[Coordinator] could not parse a valid choice from the reply; "
          "falling back to the fixed dispatch order for this turn.")
    return None


def run_multi_agent(args, max_debug_turns: int):
    chat = GroupChat()
    chat.post("prompt", PROBLEM_STATEMENT)

    used = []
    formulation = None
    plan = None
    final_code = None
    force_fem = None
    output = ""
    turns_used = 0
    evaluator_text = None

    def do_formulator():
        nonlocal formulation
        formulation = formulator_stage(args.formulator_model, args.host, chat)
        used.append("Formulator")

    def do_planner():
        nonlocal plan
        if formulation is None:
            do_formulator()
        plan = planner_stage(args.planner_model, args.host, chat, formulation)
        used.append("Planner")

    def do_coder_subsystem():
        nonlocal final_code, force_fem, output, turns_used
        if formulation is None:
            do_formulator()
        if plan is None:
            do_planner()
        final_code, force_fem, output, turns_used = run_coder_subsystem(
            args.coder_model, args.corrector_model, args.host,
            formulation, plan, max_debug_turns,
        )
        used.append("FEniCS Coder")
        chat.post(
            "FEniCS Coder subsystem",
            f"Final result after {turns_used}/{max_debug_turns} retries: "
            f"{'FAIL' if force_fem is None else f'{force_fem:.4f} N'}",
        )

    def do_evaluator():
        nonlocal evaluator_text
        if "FEniCS Coder" not in used:
            do_coder_subsystem()
        evaluator_text = evaluator_stage(args.evaluator_model, args.host, chat, force_fem)
        used.append("Evaluator")

    dispatch = {
        "Formulator": do_formulator,
        "Planner": do_planner,
        "FEniCS Coder": do_coder_subsystem,
        "Evaluator": do_evaluator,
    }

    if args.coordinator_mode == "fixed":
        print("\n[Coordinator] (fixed order) dispatching: "
              "Formulator -> Planner -> FEniCS Coder -> Evaluator -> Admin")
        do_formulator()
        do_planner()
        do_coder_subsystem()
        do_evaluator()
    else:
        turn = 0
        while turn < args.max_coordinator_turns:
            turn += 1
            choice = coordinator_stage(args.coordinator_model, args.host, chat, used)
            if choice is None:
                # Fall back to whatever's still missing, in the paper's typical order.
                choice = next(
                    (c for c in ["Formulator", "Planner", "FEniCS Coder", "Evaluator"]
                     if c not in used),
                    "Admin",
                )
            if choice == "Admin":
                if "FEniCS Coder" in used and "Evaluator" in used:
                    break
                print("[Coordinator] Admin was chosen but FEniCS Coder/Evaluator "
                      "haven't both run yet -- ignoring and continuing.")
                continue
            if choice in used and choice != "Admin":
                print(f"[Coordinator] {choice} has already spoken this run; "
                      "skipping the no-op re-dispatch.")
                continue
            fn = dispatch.get(choice)
            if fn:
                fn()
        else:
            print(f"[Coordinator] hit --max-coordinator-turns={args.max_coordinator_turns} "
                  "before Admin was reached.")
        # Safety net: make sure both required stages ran before Admin, even
        # if the dynamic router never got there.
        if "FEniCS Coder" not in used:
            do_coder_subsystem()
        if "Evaluator" not in used:
            do_evaluator()

    verdict = admin_stage(force_fem, evaluator_text, turns_used, max_debug_turns)
    chat.post("Admin", verdict)
    return final_code, force_fem, turns_used, verdict


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rushikesh_67/llama3.2-2new",
                     help="Fine-tuned Ollama model tag under test. Default "
                          "for --coder-model and --corrector-model.")
    ap.add_argument("--base-model", default=None,
                     help="Ollama model tag for the non-coding 'reasoning' "
                          "agents (Coordinator/Formulator/Planner/Evaluator), "
                          "mirroring GPT-OSS (base) vs. GPT-OSS-FT (fine-"
                          "tuned) in the paper's Figure 4. Defaults to "
                          "--model if not given (with a printed note).")
    ap.add_argument("--framework", choices=["two-agent", "multi-agent"],
                     default="two-agent",
                     help="two-agent: Coder + one Debugger fix attempt "
                          "(matches the per-model leaderboard results). "
                          "multi-agent: full Figure-4 graph -- Coordinator, "
                          "Formulator, Planner, FEniCS Coder, Corrector, "
                          "Executor, Evaluator, Admin.")
    ap.add_argument("--coordinator-mode", choices=["fixed", "dynamic"],
                     default="fixed",
                     help="multi-agent only. fixed: hardcoded dispatch "
                          "order, no LLM routing call. dynamic: an actual "
                          "LLM call each turn picks the next agent from the "
                          "group chat transcript, as in Figure 4.")
    ap.add_argument("--max-debug-turns", type=int, default=None,
                     help="Number of Coder<->Corrector<->Executor retries "
                          "after an execution error (default: 1 for "
                          "two-agent, 3 for multi-agent)")
    ap.add_argument("--max-coordinator-turns", type=int, default=8,
                     help="Safety cap on Coordinator dispatch rounds in "
                          "--coordinator-mode dynamic")
    ap.add_argument("--coder-model", default=None, help="Defaults to --model")
    ap.add_argument("--corrector-model", default=None, help="Defaults to --model")
    ap.add_argument("--coordinator-model", default=None, help="Defaults to --base-model")
    ap.add_argument("--formulator-model", default=None, help="Defaults to --base-model")
    ap.add_argument("--planner-model", default=None, help="Defaults to --base-model")
    ap.add_argument("--evaluator-model", default=None, help="Defaults to --base-model")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--skip-run", action="store_true",
                     help="Only run the Formulator/Coder stage(s) and print "
                          "the result, don't execute the generated code or "
                          "run any Debugger/Corrector/Evaluator/Admin stages")
    args = ap.parse_args()

    base_model = args.base_model or args.model
    if args.base_model is None and args.framework == "multi-agent":
        print(f"NOTE: --base-model not given; Coordinator/Formulator/Planner/"
              f"Evaluator will use the same tag as --model ({args.model}). "
              f"For a faithful GPT-OSS (base) vs. GPT-OSS-FT (fine-tuned) "
              f"split, pass a separate non-fine-tuned local tag, e.g. "
              f"--base-model llama3.3:70b")

    args.coder_model = args.coder_model or args.model
    args.corrector_model = args.corrector_model or args.model
    args.coordinator_model = args.coordinator_model or base_model
    args.formulator_model = args.formulator_model or base_model
    args.planner_model = args.planner_model or base_model
    args.evaluator_model = args.evaluator_model or base_model

    max_debug_turns = args.max_debug_turns
    if max_debug_turns is None:
        max_debug_turns = 3 if args.framework == "multi-agent" else 1

    print("=" * 70)
    print(f"Model under test:  {args.model}")
    if args.framework == "multi-agent":
        print(f"Base/reasoning model: {base_model}")
        print(f"Coordinator mode:  {args.coordinator_mode}")
    print(f"Framework:         {args.framework}  (max debug turns: {max_debug_turns})")
    print("Test: ALL-FEM scissor/cutting-mode surrogate "
          "(Libu George & Bharanidaran 2020)")
    print("=" * 70)

    if args.skip_run:
        if args.framework == "multi-agent":
            formulation = formulator_stage(args.formulator_model, args.host, GroupChat())
            plan = planner_stage(args.planner_model, args.host, GroupChat(), formulation)
            code, _ = (CODER_INSTRUCTIONS_MA.format(spec=formulation, plan=plan), None)
            reply = call_ollama(args.coder_model, code, args.host)
            code = extract_code(reply)
        else:
            code, _ = coder_stage(args.model, args.host, PROBLEM_STATEMENT)
        with open("generated_fenics_scissor_code.py", "w") as f:
            f.write(code)
        print("[--skip-run] wrote generated_fenics_scissor_code.py, not executing.")
        return

    if args.framework == "multi-agent":
        args.max_coordinator_turns = args.max_coordinator_turns
        final_code, force_fem, turns_used, verdict = run_multi_agent(args, max_debug_turns)
    else:
        code, _ = coder_stage(args.model, args.host, PROBLEM_STATEMENT)
        print(f"\nReference (paper, baseline geometry, WITH contact + real "
              f"sheath): {REFERENCE_FORCE_N} N")
        print("Our surrogate omits contact and the physical sheath -- see the "
              "module docstring. Treat this as an order-of-magnitude sanity "
              "check, not a precision validation.")

        turns_used = 0
        force_fem, output = run_generated_code(code)
        while force_fem is None and turns_used < max_debug_turns:
            turns_used += 1
            print(f"\n--- Debugger turn {turns_used}/{max_debug_turns} ---")
            code = debugger_stage(args.model, args.host, PROBLEM_STATEMENT, code, output)
            force_fem, output = run_generated_code(code)

        if force_fem is None:
            verdict = ("FAIL — generated code did not run successfully or did not "
                       "print a parseable SCISSOR_FORCE_N value, even after the "
                       "Debugger stage.")
        elif force_fem <= 0:
            verdict = "FAIL — non-positive or zero reaction force; model is not transmitting the moment correctly."
        else:
            ratio = force_fem / REFERENCE_FORCE_N
            if 0.2 <= ratio <= 5.0:
                verdict = "PLAUSIBLE — within an order of magnitude of the paper's reported value, given the surrogate's approximations."
            elif 0.05 <= ratio <= 20.0:
                verdict = "QUESTIONABLE — positive but notably off; inspect the generated moment BC and geometry for unit/setup errors."
            else:
                verdict = "FAIL — result is wildly different (likely a units error, e.g. mm vs m, or a broken moment BC)."

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"Framework:            {args.framework}"
          + (f" ({args.coordinator_mode} coordinator)" if args.framework == "multi-agent" else ""))
    print(f"Debugger/retry turns used: {turns_used}/{max_debug_turns}")
    if force_fem is None:
        print("FAIL: no valid SCISSOR_FORCE_N was produced.")
        print("\nVERDICT:", verdict)
        sys.exit(1)

    print(f"FEM reaction force (surrogate model): {force_fem:.4f} N")
    print(f"Paper reference (real model):         {REFERENCE_FORCE_N} N")
    if force_fem > 0:
        print(f"Ratio to reference:                   {force_fem / REFERENCE_FORCE_N:.2f}x")
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
