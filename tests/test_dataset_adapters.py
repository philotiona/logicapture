from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.datasets import (DatasetError, DatasetEpisode, ImageRef, Query, SplitProvenance,
                          SolverInput, load_hoi, load_logo, load_classic, load_openworld,
                          audit_split_overlap)

FIXTURE = Path(__file__).parent/"fixtures/datasets/hoi_val_unseen_obj_unseen_act_one.json"
SPLIT = "val_unseen_obj_unseen_act"


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def image_file(root, name, number, *, size=(32, 24)):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), number, np.uint8)
    image[number % size[1], :, 0] = 255
    Image.fromarray(image).save(path)
    return ImageRef(root, name)


def small_episode(root):
    refs = [image_file(root, f"image{i}.png", 20+i) for i in range(6)]
    return DatasetEpisode("test", "opaque-id", "test", tuple(refs[:2]), tuple(refs[2:4]),
                          (Query(refs[4], True), Query(refs[5], False)),
                          SplitProvenance("https://example.test/dataset", "abc", "test"),
                          hidden_concept="SECRET CONCEPT MUST NOT REACH SOLVER")


def test_real_hoi_annotation_preserves_official_support_query_split(tmp_path):
    episode, = load_hoi(FIXTURE, tmp_path, split=SPLIT, require_files=False)
    assert len(episode.support_positive) == len(episode.support_negative) == 6
    assert episode.hidden_concept == "swing++tennis_racket"
    assert episode.queries[0].image.relative_path == "openimages/images/7fbc6a69016992fe.jpg"
    assert episode.queries[1].image.relative_path == "openimages/images/b0fbf7b520eef8f7.jpg"
    assert [q.label for q in episode.queries] == [True, False]
    assert episode.queries[0].image.coordinate_scale == 1.6
    assert episode.support_positive[0].coordinate_scale == 1
    assert episode.provenance.annotation_sha256 == sha256(FIXTURE.read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError):
        load_hoi(FIXTURE, tmp_path, split=SPLIT)


def test_solver_boundary_contains_only_pixels_and_support_roles(tmp_path):
    episode = small_episode(tmp_path)
    prepared = episode.prepare(seed=23)
    assert {f.name for f in fields(SolverInput)} == {"positive", "negative", "queries"}
    assert all(type(image) is np.ndarray and image.dtype == np.uint8 and image.shape[2] == 3
               for group in (prepared.input.positive, prepared.input.negative, prepared.input.queries) for image in group)
    assert not hasattr(prepared.input, "query_labels")
    assert not hasattr(prepared.input, "episode_id")
    assert not hasattr(prepared.input, "hidden_concept")
    assert prepared.query_labels == tuple(episode.queries[i].label for i in prepared.query_order)
    assert prepared.query_order == episode.prepare(seed=23).query_order
    assert len({episode.prepare(seed=i).query_order for i in range(12)}) == 2


@pytest.mark.parametrize("path", ["../x.jpg", "images/../../x.jpg", "/absolute.jpg", "C:/x.jpg",
                                 "C:x.jpg", "\\\\server\\share.jpg", "file:///tmp/x.jpg", "https://host/x.jpg",
                                 "images\\..\\x.jpg", "nul", "COM1.jpg", "name. ", "\x00.jpg"])
def test_root_escape_and_ambiguous_windows_paths_are_rejected(tmp_path, path):
    with pytest.raises(DatasetError):
        ImageRef(tmp_path, path)


def test_symlink_escape_is_rejected_at_parse_and_use_time(tmp_path):
    root = tmp_path/"root"
    root.mkdir()
    outside = tmp_path/"outside.png"
    Image.new("RGB", (8, 8)).save(outside)
    try:
        (root/"link.png").symlink_to(outside)
    except OSError:
        pytest.skip("Host does not permit creating test symlinks")
    with pytest.raises(DatasetError):
        ImageRef(root, "link.png")


def test_duplicate_source_crop_and_byte_copy_leakage_are_rejected(tmp_path):
    episode = small_episode(tmp_path)
    with pytest.raises(DatasetError, match="Repeated source"):
        replace(episode, queries=(Query(replace(episode.support_positive[0], crop_bbox=(1,1,8,8)), True),))
    original = episode.support_positive[0].path.read_bytes()
    copied = tmp_path/"copied.png"
    copied.write_bytes(original)
    changed = replace(episode, queries=(Query(ImageRef(tmp_path, "copied.png"), True), episode.queries[1]))
    with pytest.raises(DatasetError, match="Byte-identical"):
        changed.prepare()


def test_hoi_openimages_crop_scale_applies_without_mutating_annotation(tmp_path):
    source = np.arange(40*50*3, dtype=np.uint8).reshape(40,50,3)
    Image.fromarray(source).save(tmp_path/"image.png")
    original = [2,3,10,12]
    ref = ImageRef(tmp_path, "image.png", original, 1.6)
    assert np.array_equal(ref.load_rgb(), source[4:19,3:16])
    assert original == [2,3,10,12]
    with pytest.raises(DatasetError, match="empty"):
        ImageRef(tmp_path, "image.png", (100,100,200,200)).load_rgb()


def test_real_hoi_corruptions_are_rejected_instead_of_repairing_labels(tmp_path):
    records = json.loads(FIXTURE.read_bytes())
    records[0][0].pop()
    with pytest.raises(DatasetError, match="seven"):
        load_hoi(write_json(tmp_path/"sample.json", records), tmp_path, split=SPLIT, require_files=False)
    with pytest.raises(DatasetError, match="split"):
        load_hoi(FIXTURE, tmp_path, split="test_made_up", require_files=False)
    with pytest.raises(DatasetError, match="filename"):
        load_hoi(write_json(tmp_path/"bongard_hoi_train.json", []), tmp_path, split=SPLIT, require_files=False)
    records = json.loads(FIXTURE.read_bytes())
    records[0][0][0]["crop_bbox"] = None
    with pytest.raises(DatasetError, match="crop_bbox"):
        load_hoi(write_json(tmp_path/"sample.json", records), tmp_path, split=SPLIT, require_files=False)


def test_logo_layout_and_split_isolation(tmp_path):
    # Identifier is from the official preview; generated pixels are parser tests.
    task = "ff_nact6_0292"
    for label in ("0", "1"):
        for i in range(7):
            image_file(tmp_path, f"ff/images/{task}/{label}/{i}.png", 30+int(label)*20+i)
    split_file = write_json(tmp_path/"ShapeBongard_V2_split.json", {"train": [], "val": [], "test_ff": [task]})
    episode, = load_logo(tmp_path, split="test_ff")
    assert len(episode.support_positive) == 6
    assert episode.queries[0].image.relative_path.endswith("/1/6.png")
    assert task not in episode.episode_id
    assert episode.hidden_concept == task
    write_json(split_file, {"train": [task], "test_ff": [task]})
    with pytest.raises(DatasetError, match="across splits"):
        load_logo(tmp_path, split="test_ff")


def test_duplicate_json_keys_are_rejected(tmp_path):
    manifest = tmp_path/"split.json"
    manifest.write_text('{"test_ff":[],"test_ff":[]}', encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate JSON"):
        load_logo(tmp_path, split="test_ff", split_file=manifest)


def test_cross_split_reuse_is_reported_separately(tmp_path):
    episode = small_episode(tmp_path)
    result = audit_split_overlap([episode, replace(episode, split="val")])
    assert result["overlapping_source_images"] == 6
    assert all(splits == ["test", "val"] for splits in result["images"].values())
    checked = audit_split_overlap([episode, replace(episode,split="val")],check_content=True)
    assert checked["byte_identical_across_splits"] == 6


def test_openworld_windows_index_keeps_semantic_filenames_out_of_input(tmp_path):
    files = [f"images/0008/{role}__{i}__https:__semantic-secret.jpg" for role in ("pos", "neg") for i in range(7)]
    source = [{"uid":"0008", "concept":"secret concept", "caption":"secret caption", "imageFiles":files}]
    annotation = write_json(tmp_path/"bongard_ow_test.json", source)
    paths = {}
    for i, original in enumerate(files):
        ref = image_file(tmp_path, f"portable/{i}.png", 40+i)
        paths[original] = ref.relative_path
    index = write_json(tmp_path/"index.json", {"paths":paths, "annotation_sha256":sha256(annotation.read_bytes()).hexdigest()})
    episode, = load_openworld(annotation, tmp_path, split="test", image_index=index)
    assert len(episode.prepare().input.queries) == 2
    assert episode.hidden_concept == "secret concept"
    paths[files[-1]] = "../escape.png"
    write_json(index, {"paths":paths})
    with pytest.raises(DatasetError):
        load_openworld(annotation, tmp_path, split="test", image_index=index)


def test_classic_uses_original_sixth_cell_as_query(tmp_path):
    groups = []
    for role in ("positive", "negative"):
        groups.append([image_file(tmp_path, f"{role}/{i}.png", 20+i+(40 if role=="negative" else 0)).relative_path
                       for i in range(6)])
    write_json(tmp_path/"episodes.json", {"format":"classic-bongard-cells-v1", "split":"classic-test-021-100",
               "items":[{"id":"p0021", "positive":groups[0], "negative":groups[1]}]})
    episode, = load_classic(tmp_path)
    assert len(episode.support_positive) == len(episode.support_negative) == 5
    assert episode.queries[0].image.relative_path == "positive/5.png"
    assert episode.provenance.protocol.startswith("first-five")
    assert episode.hidden_concept is None


def test_openworld_availability_condition_keeps_original_query_fixed(tmp_path):
    files = [f"images/0008/{role}__{i}__https:__example.jpg" for role in ("pos", "neg") for i in range(7)]
    annotation = write_json(tmp_path/"annotations.json", [{"uid":"0008", "imageFiles":files}])
    paths = {}
    for i, original in enumerate(files):
        if i in (0,8):
            continue
        paths[original] = image_file(tmp_path, f"{i}.png", 50+i).relative_path
    index = write_json(tmp_path/"index.json", {"paths":paths})
    with pytest.raises(FileNotFoundError):
        load_openworld(annotation,tmp_path,split="test",image_index=index,support_per_group=3)
    episode,=load_openworld(annotation,tmp_path,split="test",image_index=index,support_per_group=3,
                           allow_missing_support=True,episode_ids=["0008"])
    assert [ref.relative_path for ref in episode.support_positive] == ["1.png","2.png","3.png"]
    assert episode.queries[0].image.relative_path == "6.png"
    assert episode.queries[1].image.relative_path == "13.png"
    del paths[files[6]]
    write_json(index,{"paths":paths})
    with pytest.raises(FileNotFoundError):
        load_openworld(annotation,tmp_path,split="test",image_index=index,support_per_group=3,allow_missing_support=True)
    with pytest.raises(DatasetError,match="absent"):
        load_openworld(annotation,tmp_path,split="test",image_index=index,episode_ids=["9999"])


def test_frozen_openworld_retrieval_refuses_to_mutate_existing_run(tmp_path, monkeypatch):
    from src.datasets import fetch_openworld
    write_json(tmp_path/"cohort_test.json", {"already":"frozen"})
    def forbidden_network(*args, **kwargs):
        pytest.fail("A frozen run must reject refresh before any network request")
    monkeypatch.setattr(fetch_openworld,"_get",forbidden_network)
    with pytest.raises(DatasetError, match="frozen cohort"):
        fetch_openworld.fetch_openworld_subset(tmp_path,split="test",count=10)
