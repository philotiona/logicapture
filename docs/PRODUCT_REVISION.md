# Product revision: infer, explain, generate

The previous interface mixed a symbolic guessing game, two competing Bongard interfaces, research baselines, optional large models, benchmarks, sorting and coursework. Users had to choose implementation details before seeing a result. Some choices bypassed validated drawing reasoning or reduced the number of examples available to it.

The revised app opens on Bongard and exposes three purposeful workspaces: **Bongard**, **Sort images**, and **РГР coursework**.

| Previous control | Current behavior |
|---|---|
| Akinator navigation, session settings and yes/no game | Removed from the application |
| Experiments dashboard | Removed; reproducible research modules and artifacts remain outside the product flow |
| General versus structural Bongard interfaces | One guided Bongard interface |
| Perception pipeline, CNN triggers, representation and depth | Automatic traced structural rules, then validated numeric rules |
| Optional language-model hypotheses and model installation commands on Bongard | Removed from Bongard |
| Automatic withheld examples and evaluation switches | Use all given examples; test genuinely new drawings afterward |
| Dataset catalogs mixed with upload sources | Three input methods and three curated examples |
| Separate read, infer and generate buttons | One Find the rule action with automatic checked generation |
| Similarity classification after rule discovery fails | Explicit unsolved result; similarity-based sorting has its own workspace |
| Repeated learn/sort actions after successful processing | Present the next stage and its relevant action |

The project's distinguishing workflow is visible in the result: an executable separating explanation, the measured evidence in the original examples, and a newly constructed matching drawing and counterexample. The generated images are checked again from pixels against every retained rule, rather than merely trusting the generator's symbolic scene.

Only input correction and evidence inspection remain optional in Bongard. These controls have a concrete meaning for the user's problem. Raw programs, refit diagnostics and reproducibility details are downloadable, rather than prerequisites to using the app.

The revision changes the product flow, not the saved historical benchmark results. It does not establish a new rule-recovery accuracy figure. The existing failure counts remain documented.
