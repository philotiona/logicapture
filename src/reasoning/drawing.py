"""Shared executable drawing inference for uploads, camera, and coverage audit."""
from .structural import fit_structural_support
from .measurement_rules import fit_measurement_rules


def fit_drawing_rules(positive, negative, *, progress=None):
    fit, attempt = fit_structural_support(positive,negative,progress=progress)
    if fit is not None:
        return fit, attempt, "executable_structural_rules"
    measured, diagnostic = fit_measurement_rules(positive,negative,progress=progress)
    if measured is None and diagnostic["reason"] == "outside_measurement_rule_domain":
        attempt["measurement_reason"] = diagnostic["reason"]
        return None,attempt,"executable_structural_rules"
    diagnostic["structural_reason"] = attempt["reason"]
    return measured, diagnostic, "executable_measurement_rules"
