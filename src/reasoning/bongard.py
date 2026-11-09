"""The product's single Bongard policy: infer, explain and verify new drawings.

Image similarity and language guesses never count as recovering a Bongard rule.
Historical baselines remain available to the evaluation modules.
"""
from dataclasses import dataclass

from src.active import GenerationError
from src.vision.pretrained import image_digest
from .drawing import fit_drawing_rules
from .examples import generate_checked_examples
from .general import FittedVisualTask, validate_episode_images


@dataclass
class BongardResult:
    model: FittedVisualTask | None
    diagnostics: dict
    examples: tuple
    generation_error: str | None = None

    def to_dict(self):
        return {"rule_found": self.model is not None, "diagnostics": self.diagnostics,
                "generated_verification": [example.verification for example in self.examples],
                "generation_error": self.generation_error}


def infer_problem(positive, negative, *, progress=None, remove_panel_ruling=True):
    if min(len(positive), len(negative)) < 3:
        raise ValueError("Add at least three different drawings to each group. Four to six are preferable.")
    if max(len(positive), len(negative)) > 8:
        raise ValueError("Use at most eight drawings in each group for verified rule discovery.")
    images = validate_episode_images(positive, negative)
    count = len(positive)
    if {image_digest(image) for image in images[:count]} & {image_digest(image) for image in images[count:]}:
        raise ValueError("The same drawing appears in both groups. Correct the groups before finding a rule.")
    if any(len({image_digest(image) for image in group}) != len(group) for group in (images[:count], images[count:])):
        raise ValueError("A group repeats the same drawing. Use distinct examples so the rule can be checked.")
    fitted, attempt, pipeline = fit_drawing_rules(images[:count], images[count:], progress=progress,
                                                   remove_panel_ruling=remove_panel_ruling)
    diagnostics = {"input": "pixels and group membership only", "query_labels_used": False,
                   "all_given_examples_used": True, "local_only": True,
                   "remove_panel_ruling": remove_panel_ruling,
                   "selected_pipeline": pipeline, "structural_attempt": attempt,
                   "logical_explanation": attempt["candidates"][0]["text"] if fitted else None}
    if fitted is None:
        return BongardResult(None, diagnostics, ())
    ids = tuple([f"A{i+1:02d}" for i in range(count)] + [f"B{i+1:02d}" for i in range(len(negative))])
    # This wrapper's structural branch uses the frozen programs directly;
    # it never needs a statistical encoder or query-dependent fitting.
    model = FittedVisualTask(None, fitted, diagnostics, "adaptive-structure-dsl-v1", ids,
                             frozenset(image_digest(image) for image in images))
    if progress:
        progress("Drawing a new matching example and counterexample, then checking their pixels")
    try:
        examples = tuple(generate_checked_examples(diagnostics))
        return BongardResult(model, diagnostics, examples)
    except GenerationError as error:
        return BongardResult(model, diagnostics, (), str(error))
