"""Independent parsers for documented official JSON/layout formats.

No upstream training or data-loader code is imported or executed. Box and
concept annotations are never added to the solver's pixel-only input.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

from .episodes import DatasetEpisode, DatasetError, ImageRef, Query, SplitProvenance, resolve_image

HOI_SPLITS = ("train",)+tuple(f"{partition}_{condition}" for partition in ("val", "test")
                             for condition in ("seen_obj_seen_act", "seen_obj_unseen_act",
                                               "unseen_obj_seen_act", "unseen_obj_unseen_act"))
LOGO_SPLITS = ("train", "val", "test_ff", "test_bd", "test_hd_comb", "test_hd_novel")
HOI_SOURCE = "https://zenodo.org/records/7079175"
LOGO_SOURCE = "https://github.com/NVlabs/Bongard-LOGO"


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > 256*1024*1024:
        raise DatasetError("Annotation JSON exceeds the 256MB parsing limit")
    payload = path.read_bytes()
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DatasetError("Duplicate JSON object key in dataset annotations")
            result[key] = value
        return result
    def invalid_constant(value):
        raise DatasetError(f"Non-finite JSON constant in dataset annotations: {value}")
    try:
        return json.loads(payload, object_pairs_hook=unique_object, parse_constant=invalid_constant), sha256(payload).hexdigest()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetError(f"Invalid dataset JSON: {path.name}") from error


def _limit(value):
    if value is not None and (type(value) is not int or value < 1):
        raise DatasetError("limit must be a positive integer or None")


def load_hoi(annotation_file, image_root, *, split, require_files=True, limit=None,
             openimages_scale=1.6, source_url=HOI_SOURCE):
    """Read official [positive[7], negative[7], concept] episodes.

    Filename/split consistency is checked for official filenames. A renamed
    sample can be supplied with explicit split and provenance source_url.
    ``require_files=False`` validates metadata only; it does not run a solver.
    """
    _limit(limit)
    if split not in HOI_SPLITS:
        raise DatasetError(f"Unknown HOI split: {split}")
    filename = Path(annotation_file).name
    if filename.startswith("bongard_hoi_") and filename != f"bongard_hoi_{split}.json":
        raise DatasetError("Annotation filename does not match the requested HOI split")
    records, digest = _read_json(annotation_file)
    if not isinstance(records, list) or not records:
        raise DatasetError("HOI annotations must contain a nonempty episode list")
    episodes = []
    for index, record in enumerate(records[:limit]):
        if not isinstance(record, list) or len(record) != 3 or not isinstance(record[2], str):
            raise DatasetError(f"Malformed HOI episode {index}")
        groups = []
        for group in record[:2]:
            if not isinstance(group, list) or len(group) != 7:
                raise DatasetError("Official HOI episodes require exactly seven images per group")
            refs = []
            for item in group:
                if not isinstance(item, dict) or not {"im_path", "crop_bbox"}.issubset(item):
                    raise DatasetError("HOI image is missing im_path or crop_bbox")
                if not isinstance(item["crop_bbox"], (list, tuple)):
                    raise DatasetError("HOI crop_bbox must be four integer pixel coordinates")
                relative = item["im_path"]
                # The official source loader applies this resolution correction
                # to OpenImages only. It is explicit here for alternate exports.
                scale = openimages_scale if isinstance(relative, str) and "openimages" in Path(relative).parts else 1.
                refs.append(ImageRef(Path(image_root), relative, item["crop_bbox"], scale))
            groups.append(tuple(refs))
        episode = DatasetEpisode("bongard-hoi", f"{split}:{index:06d}", split,
            groups[0][:-1], groups[1][:-1], (Query(groups[0][-1], True), Query(groups[1][-1], False)),
            SplitProvenance(source_url, digest, split, note="Official annotated crop; OpenImages coordinate correction applied."),
            hidden_concept=record[2])
        if require_files:
            episode.validate_files()
        episodes.append(episode)
    return tuple(episodes)


def load_logo(root, *, split, split_file=None, require_files=True, limit=None):
    """Read ShapeBongard_V2 split JSON and ff/bd/hd/images/TASK/{1,0}/*.png.

    Uses sorted first-six support and final-query files. No generation programs
    or shape words from task names enter the solver input.
    """
    _limit(limit)
    if split not in LOGO_SPLITS:
        raise DatasetError(f"Unknown LOGO split: {split}")
    root = Path(root).resolve()
    split_file = Path(split_file) if split_file else root/"ShapeBongard_V2_split.json"
    splits, digest = _read_json(split_file)
    if not isinstance(splits, dict) or split not in splits:
        raise DatasetError("LOGO split manifest is missing the requested split")
    owner = {}
    for name, tasks in splits.items():
        if name not in LOGO_SPLITS or not isinstance(tasks, list):
            raise DatasetError("Unrecognized LOGO split manifest structure")
        for task in tasks:
            if not isinstance(task, str) or not re.fullmatch(r"(?:ff|bd|hd)_[A-Za-z0-9_.+-]+", task):
                raise DatasetError("Invalid LOGO task identifier")
            if task in owner:
                raise DatasetError("Repeated LOGO task within or across splits")
            owner[task] = name
    tasks = sorted(splits[split])
    if not tasks:
        raise DatasetError("Requested LOGO split contains no tasks")
    episodes = []
    for task in tasks[:limit]:
        family = task.split("_", 1)[0]
        groups = []
        for label in ("1", "0"):
            directory = resolve_image(root, f"{family}/images/{task}/{label}")
            if not directory.is_dir():
                raise FileNotFoundError(f"LOGO image directory is missing: {directory}")
            names = sorted(p.name for p in directory.iterdir() if p.suffix.lower() == ".png")
            if len(names) != 7:
                raise DatasetError("Official LOGO episodes require exactly seven PNG images per group")
            refs = tuple(ImageRef(root, f"{family}/images/{task}/{label}/{name}") for name in names)
            groups.append(refs)
        opaque = sha256(task.encode()).hexdigest()[:16]
        episode = DatasetEpisode("bongard-logo", opaque, split, groups[0][:-1], groups[1][:-1],
            (Query(groups[0][-1], True), Query(groups[1][-1], False)),
            SplitProvenance(LOGO_SOURCE, digest, split), hidden_concept=task)
        if require_files:
            episode.validate_files()
        episodes.append(episode)
    return tuple(episodes)


def load_openworld(annotation_file, image_root, *, split, image_index, support_per_group=6,
                   require_files=True, limit=None, episode_ids=None, allow_missing_support=False):
    """Read official OpenWorld JSON using an explicit portable filename index.

    Original filenames embed URLs and are invalid on Windows. ``image_index``
    maps each exact original imageFiles entry to a safe local relative path.
    The last image of each original group remains the held-out query. Choosing
    fewer support examples is an explicitly different shot condition.
    """
    _limit(limit)
    if split not in ("train", "val", "test"):
        raise DatasetError("Unknown OpenWorld split")
    if type(support_per_group) is not int or not 1 <= support_per_group <= 6:
        raise DatasetError("OpenWorld support_per_group must be between one and six")
    records, digest = _read_json(annotation_file)
    index, _ = _read_json(image_index)
    if not isinstance(index, dict) or not isinstance(index.get("paths"), dict):
        raise DatasetError("OpenWorld portable image index requires a paths mapping")
    if index.get("annotation_sha256") not in (None, digest):
        raise DatasetError("OpenWorld image index refers to different annotations")
    if not isinstance(records, list) or not records:
        raise DatasetError("OpenWorld annotations must be a nonempty episode list")
    if episode_ids is not None:
        if not isinstance(episode_ids, (tuple, list)) or not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise DatasetError("OpenWorld episode_ids must be a nonempty unique list")
        if not set(episode_ids).issubset({item.get("uid") for item in records if isinstance(item, dict)}):
            raise DatasetError("OpenWorld requested identifier is absent from the official split")
        records = [record for record in records if record.get("uid") in episode_ids]
    seen, episodes = set(), []
    for item in records[:limit]:
        if not isinstance(item, dict) or not re.fullmatch(r"\d{4}", str(item.get("uid", ""))):
            raise DatasetError("Invalid OpenWorld episode identifier")
        uid, files = item["uid"], item.get("imageFiles")
        if uid in seen:
            raise DatasetError("Duplicate OpenWorld episode identifier")
        seen.add(uid)
        if not isinstance(files, list) or len(files) != 14 or any(not isinstance(p, str) for p in files):
            raise DatasetError("OpenWorld episodes require fourteen image references")
        groups = []
        for role, offset in (("pos", 0), ("neg", 7)):
            supports = files[offset:offset+6]
            if allow_missing_support:
                supports = [original for original in supports if original in index["paths"]]
            if len(supports) < support_per_group:
                raise FileNotFoundError(f"Not enough locally available OpenWorld support images: {uid}/{role}")
            selected = supports[:support_per_group]+[files[offset+6]]
            expected = files[offset:offset+7]
            for position, original in enumerate(expected):
                if not original.startswith(f"images/{uid}/{role}__{position}__"):
                    raise DatasetError("OpenWorld image order or group marker is inconsistent")
            refs = []
            for original in selected:
                if original not in index["paths"]:
                    raise FileNotFoundError(f"OpenWorld image is not available in the local index: {uid}/{role}")
                refs.append(ImageRef(Path(image_root), index["paths"][original]))
            groups.append(tuple(refs))
        episode = DatasetEpisode("bongard-openworld", uid, split, groups[0][:-1], groups[1][:-1],
            (Query(groups[0][-1], True), Query(groups[1][-1], False)),
            SplitProvenance("https://huggingface.co/datasets/rujiewu/Bongard-OpenWorld", digest, split,
                            note=f"{support_per_group}-shot; official final image in each group remains query."+
                            (" Support uses first available original support images; availability-filtered condition." if allow_missing_support else "")),
            hidden_concept=item.get("concept"))
        if require_files:
            episode.validate_files()
        episodes.append(episode)
    return tuple(episodes)


def load_openworld_cohort(root, *, cohort_file, require_files=True):
    """Load an explicitly frozen availability report, with no silent resampling."""
    root = Path(root).resolve()
    cohort, _ = _read_json(cohort_file)
    if not isinstance(cohort, dict) or cohort.get("format") != "openworld-cohort-v1":
        raise DatasetError("Invalid OpenWorld cohort manifest")
    annotation = resolve_image(root, cohort["annotation_file"], require_file=True)
    if cohort.get("source_annotation_sha256") != sha256(annotation.read_bytes()).hexdigest():
        raise DatasetError("OpenWorld frozen cohort annotation hash changed")
    index = resolve_image(root, cohort["image_index"], require_file=True)
    if cohort.get("image_index_sha256") is not None and cohort["image_index_sha256"] != sha256(index.read_bytes()).hexdigest():
        raise DatasetError("OpenWorld frozen image index hash changed")
    return load_openworld(annotation, root,
                          split=cohort["split"], image_index=index,
                          support_per_group=cohort["support_per_group"],
                          episode_ids=cohort["episode_ids"], allow_missing_support=cohort.get("allow_missing_support", False),
                          require_files=require_files)


def load_classic(root, *, manifest_file=None, require_files=True, limit=None):
    """Read locally prepared classic cells; five support plus one query/group."""
    _limit(limit)
    root = Path(root).resolve()
    manifest, digest = _read_json(manifest_file or root/"episodes.json")
    if not isinstance(manifest, dict) or manifest.get("format") != "classic-bongard-cells-v1":
        raise DatasetError("Invalid classic Bongard cell manifest")
    items, split = manifest.get("items"), manifest.get("split")
    if not isinstance(items, list) or not items or not isinstance(split, str) or not split:
        raise DatasetError("Classic manifest needs items and explicit split provenance")
    episodes, seen = [], set()
    for item in items[:limit]:
        if not isinstance(item, dict) or not re.fullmatch(r"p\d{4}", str(item.get("id", ""))):
            raise DatasetError("Invalid classic problem identifier")
        if item["id"] in seen:
            raise DatasetError("Duplicate classic problem identifier")
        seen.add(item["id"])
        groups = []
        for key in ("positive", "negative"):
            paths = item.get(key)
            if not isinstance(paths, list) or len(paths) != 6:
                raise DatasetError("Classic problems need exactly six images per group")
            groups.append(tuple(ImageRef(root, path) for path in paths))
        episode = DatasetEpisode("classic-bongard", item["id"], split, groups[0][:-1], groups[1][:-1],
            (Query(groups[0][-1], True), Query(groups[1][-1], False)),
            SplitProvenance(manifest.get("source_url", "https://osf.io/95dks/"), digest, split,
                            protocol="first-five-support-last-query-per-group-v1",
                            note="Within-problem heldout protocol; no official train/test partition is claimed."))
        if require_files:
            episode.validate_files()
        episodes.append(episode)
    return tuple(episodes)
