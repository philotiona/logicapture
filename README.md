# Logicapture

**Find a visual rule, explain it, and draw new examples.**

Logicapture reads the two groups in a Bongard problem and looks for an executable rule separating them. Its distinguishing workflow goes beyond labeling the supplied images: it constructs a new matching drawing and a counterexample, then re-reads their pixels to check them against the retained rules.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://127.0.0.1:8501>. `run.ps1` also starts the app.

For a fresh environment, use Python 3.11 or later and install `requirements.txt`. Bongard rule discovery requires only these base dependencies; it does not download a vision or language model.

## Find a rule

**Find a rule** is the opening screen for two-group Bongard puzzles:

1. **Choose your input.** Upload the complete page, supply separate A/B drawings, or choose **Try an example**. For separate drawings, turn on **Drawings include printed panel borders** only when the border is part of the printed puzzle layout. Leave it off when an outline is part of the drawing.
2. **Check the groups.** Make sure the displayed crops belong to the correct group. For an incorrectly split page, open **Check or fix page splitting**.
3. Press **Find the rule**. The explanation and a new Group A drawing and Group B counterexample appear together. New drawings are checked from their rendered pixels against every retained rule.

Open **Why this rule?** to inspect a given drawing's measured evidence and alternative explanations. Upload new drawings under **Test the rule** to check them against the same fit. When a test-border setting is available, it applies only to test drawings. A copy of a given example is marked as such. You can download the evidence as a JSON report even when the solver cannot verify a rule.

Use **Arrow directions** under **Try an example** to see reasoning along a curve rather than a closed-set shape classifier. The other two examples demonstrate convexity and elongation. Example names are not passed to the solver.

All supplied examples are used for inference. The app automatically tries the traced path/region rule engine and then validated numeric drawing rules. It no longer asks users to choose an algorithm, representation, search depth, CNN configuration, language model or benchmark protocol. A failure remains **No verified rule found**; an image-similarity score is not substituted for a rule.

## Local Lab

Select **Local Lab → Learn and make a drawing** to try the built-in drawings. You can also upload a full page or 3–8 drawings for each group. Logicapture fits a local visual classifier from the two groups, then joins whole portions of two Group A drawings at a blank seam. It checks the rendered candidate with the frozen classifier and rejects near-copies. **Test an independent drawing** uses the same fit.

This is a CPU-only experiment with no model service or pretrained weights. Reference checks and a candidate chosen by the classifier give evidence about the learned contrast. Independent drawings are needed to assess generalization. Weak contrasts and scenes that cannot be recombined can yield no suggestion.

## Print Inspector

Select **Print Inspector** in the top bar. Try the simulated photos first, or download the A4 pack and print the good, missing, extra and misplaced examples. Upload a photo or use the camera with all four markers visible. The aligned overlay identifies where the sample differs from the reference, including cases with the same total mark count.

For your own layout, draw separate dark marks inside the blank template and upload a photo as the accepted reference. Photograph subsequent cards with the same corner marker layout. The app runs on the CPU and asks for a retake if it cannot register a usable card. It is a demo for flat cards with separate, high-contrast marks.

## Sort images

For practical two-group image sorting, use the separate **Sort images** workspace:

1. Name the groups, upload labeled examples, and press **Learn from these examples**.
2. Add the new images or a ZIP and press **Sort these images**.
3. Review individual labels and **Download sorted images + report**.

The download preserves original file bytes and contains CSV/JSON evidence and the reusable sorting setup. Uncertain images stay in review; corrections never silently change the learned model. Photo sorting needs the optional local model from `requirements-general.txt`; drawing tasks also work with the base dependencies. See [sorting instructions](docs/IMAGE_TASKS.md).

**РГР coursework** is a separate assignment workspace for model testing, photo controls, local CNN adaptation and reports. It needs `requirements-neural.txt`. See [assignment instructions](docs/RGR.md).

## Scope and evidence

Rule discovery is bounded: use 3–8 clear black-and-white drawings per group. It handles supported geometric, topological and path/arrow distinctions. Arbitrary semantic concepts and animated problems remain unsupported. Multiple explanations can fit the same examples, and support checks do not prove the human-intended rule.

The saved Foundalis development audit found executable candidates on 36 of 408 static problems; 62 of 72 answered queries were correct and 10 were wrong. Most problems remain unresolved. These are development coverage figures, not universal rule recovery. Read [the full comparison and its failures](docs/BOUNDARY_RULES.md).

[The current product brief](docs/PROJECT_BRIEF.md) defines the scope. [The interface revision](docs/PRODUCT_REVISION.md) explains the simplified workflows. Existing baselines, historical experiments and benchmark records remain available through Python modules and documentation rather than product controls.

## Development

```powershell
.\.venv\Scripts\python.exe -m compileall -q src app.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

`src/reasoning/bongard.py` defines the product's rule-discovery policy. `src/reasoning/drawing.py` routes between structural programs and numeric measurements. `src/dsl`, `src/scene`, `src/synthesis` and the constructive drawing generators provide executable semantics and rendering. `src/image_tasks.py` supplies reusable classification tasks and lossless exports. The interface is in `src/interface`.
