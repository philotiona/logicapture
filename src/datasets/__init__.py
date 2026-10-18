"""Strict local-root adapters for independent Bongard evaluation datasets."""
from .adapters import HOI_SPLITS, LOGO_SPLITS, load_hoi, load_logo, load_openworld, load_openworld_cohort, load_classic
from .episodes import (DatasetEpisode, DatasetError, ImageRef, PreparedEpisode,
                       Query, SolverInput, SplitProvenance, audit_split_overlap)

__all__ = ["DatasetEpisode", "DatasetError", "ImageRef", "PreparedEpisode", "Query", "SolverInput",
           "SplitProvenance", "HOI_SPLITS", "LOGO_SPLITS", "load_hoi", "load_logo", "load_openworld", "load_openworld_cohort", "load_classic", "audit_split_overlap"]
