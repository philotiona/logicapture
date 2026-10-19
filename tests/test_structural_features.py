"""Numeric evidence generalizes across unnamed shapes and combinations."""
import cv2
import numpy as np
import pytest

from src.vision.structural_features import StructuralEncoder, StructuralFeatureEncoder, structural_descriptor


def panel(side=192):
    return np.full((side, side, 3), 255, np.uint8)


def value(encoded, name, row=0):
    index = encoded.names.index(name)
    assert encoded.valid[row, index], name
    return float(encoded.values[row, index])


def asymmetric_panel():
    image = panel()
    points = np.array(((25, 30), (93, 20), (158, 68), (135, 122), (69, 164), (39, 100)), np.int32)
    cv2.polylines(image, [points], True, (0, 0, 0), 3)
    cv2.circle(image, (78, 80), 12, (0, 0, 0), -1)
    cv2.line(image, (143, 155), (169, 133), (0, 0, 0), 3)
    return image


def test_schema_empty_panels_and_missing_measurements_are_distinct():
    encoder = StructuralEncoder()
    result = encoder.encode([panel()])
    assert result.values.shape == result.valid.shape == (1, len(encoder.feature_names))
    assert len(result.names) > 200 and len(set(result.names)) == len(result.names)
    assert np.isfinite(result.values).all()
    assert value(result, "invariant.component_count") == 0
    assert value(result, "invariant.hole_count") == 0
    assert value(result, "invariant.ink_fraction") == 0
    index = result.names.index("invariant.arrangement_pair_distance.mean")
    assert not result.valid[0, index]
    assert result.values[0, index] == 0
    empty = encoder.encode([])
    assert empty.values.shape == empty.valid.shape == (0, len(result.names))


def test_numeric_features_are_independent_of_other_episode_images():
    encoder = StructuralEncoder()
    image = asymmetric_panel()
    single = encoder.encode([image])
    combined = encoder.encode([image, panel(), np.zeros_like(image)])
    np.testing.assert_array_equal(single.values[0], combined.values[0])
    np.testing.assert_array_equal(single.valid[0], combined.valid[0])


def test_d4_invariants_keep_orientation_in_separate_channels():
    encoder = StructuralEncoder()
    image = asymmetric_panel()
    variants = [view.copy() for k in range(4)
                for view in (np.rot90(image, k), np.rot90(image, k)[:, ::-1])]
    result = encoder.encode(variants)
    invariant = np.asarray(result.groups["invariant"])
    for row in range(1, len(result.values)):
        np.testing.assert_array_equal(result.valid[0, invariant], result.valid[row, invariant])
        np.testing.assert_allclose(result.values[0, invariant], result.values[row, invariant], atol=2e-4, rtol=2e-5)
    oriented = np.asarray(result.groups["oriented"])
    assert not np.allclose(result.values[0, oriented], result.values[2, oriented])


def test_topology_measures_holes_nesting_and_short_branches_without_names():
    nested, branch = panel(), panel()
    cv2.circle(nested, (96, 96), 70, (0, 0, 0), 3)
    cv2.circle(nested, (96, 96), 38, (0, 0, 0), 3)
    cv2.circle(nested, (96, 96), 7, (0, 0, 0), -1)
    cv2.line(branch, (24, 100), (168, 100), (0, 0, 0), 3)
    cv2.line(branch, (96, 100), (96, 87), (0, 0, 0), 3)
    result = StructuralEncoder().encode([nested, branch])
    assert value(result, "invariant.component_count", 0) == 3
    assert value(result, "invariant.hole_count", 0) == 2
    assert value(result, "invariant.euler_number", 0) == 1
    assert value(result, "invariant.nesting_depth_max", 0) == 4
    assert value(result, "invariant.skeleton_endpoint_count", 1) == 3
    assert value(result, "invariant.skeleton_junction_count", 1) == 1


def test_arbitrary_boundary_spectra_retain_information_beyond_shape_classes():
    images = []
    for frequency in (0, 5, 9):
        image = panel()
        theta = np.linspace(0, 2*np.pi, 500, endpoint=False)
        radius = 58*(1+.18*np.cos(frequency*theta)) if frequency else np.full_like(theta, 58)
        points = np.column_stack((96+radius*np.cos(theta), 96+radius*np.sin(theta))).astype(np.int32)
        cv2.fillPoly(image, [points], (0, 0, 0))
        images.append(image)
    result = StructuralEncoder().encode(images)
    indices = [i for i, name in enumerate(result.names) if "boundary." in name]
    assert np.linalg.norm(result.values[0, indices]-result.values[1, indices]) > .1
    assert np.linalg.norm(result.values[1, indices]-result.values[2, indices]) > .1
    assert value(result, "invariant.component_concavity.mean", 1) > value(result, "invariant.component_concavity.mean", 0)+.02


def test_component_arrangement_changes_without_changing_component_shapes():
    line, square, clustered = panel(), panel(), panel()
    for x, y in ((30, 96), (74, 96), (118, 96), (162, 96)):
        cv2.circle(line, (x, y), 6, (0, 0, 0), -1)
    for x, y in ((40, 40), (152, 40), (40, 152), (152, 152)):
        cv2.circle(square, (x, y), 6, (0, 0, 0), -1)
    for x, y in ((70, 70), (95, 70), (70, 95), (95, 95)):
        cv2.circle(clustered, (x, y), 6, (0, 0, 0), -1)
    result = StructuralEncoder().encode([line, square, clustered])
    assert all(value(result, "invariant.component_count", i) == 4 for i in range(3))
    assert value(result, "invariant.arrangement_covariance_minor", 0) < .001
    assert value(result, "invariant.arrangement_covariance_minor", 1) > .49
    assert value(result, "invariant.arrangement_nearest_distance.mean", 1) > 3*value(result, "invariant.arrangement_nearest_distance.mean", 2)


@pytest.mark.parametrize("frequency", [5, 13])
def test_novel_shape_families_keep_independent_topology_arrangement_combinations(frequency):
    # The second family's frequency lies beyond the stored boundary Fourier
    # orders.  Topology and arrangement still describe it without a class name.
    images, expected = [], []
    for outline in (False, True):
        for aligned in (False, True):
            image = panel()
            centers = [(30, 96), (74, 96), (118, 96), (162, 96)] if aligned else [(44, 44), (148, 44), (44, 148), (148, 148)]
            theta = np.linspace(0, 2*np.pi, 400, endpoint=False)
            radius = 12*(1+.18*np.cos(frequency*theta))
            for x, y in centers:
                points = np.column_stack((x+radius*np.cos(theta), y+radius*np.sin(theta))).astype(np.int32)
                if outline:
                    cv2.polylines(image, [points], True, (0, 0, 0), 2)
                else:
                    cv2.fillPoly(image, [points], (0, 0, 0))
            images.append(image)
            expected.append((outline, aligned))
    encoded = StructuralEncoder().encode(images)
    for row, (outline, aligned) in enumerate(expected):
        assert value(encoded, "invariant.component_count", row) == 4
        assert value(encoded, "invariant.hole_count", row) == (4 if outline else 0)
        minor = value(encoded, "invariant.arrangement_covariance_minor", row)
        assert minor < .001 if aligned else minor > .49


def test_image_resolution_change_does_not_change_the_measurement_definition():
    image = asymmetric_panel()
    larger = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    encoded = StructuralEncoder().encode([image, larger])
    np.testing.assert_array_equal(encoded.values[0], encoded.values[1])
    np.testing.assert_array_equal(encoded.valid[0], encoded.valid[1])


def test_orientation_sensitive_channels_distinguish_horizontal_and_vertical():
    image = panel()
    cv2.line(image, (30, 96), (162, 96), (0, 0, 0), 3)
    result = StructuralEncoder().encode([image, np.rot90(image).copy()])
    assert value(result, "oriented.mass_angular.real_2", 0) > .9
    assert value(result, "oriented.mass_angular.real_2", 1) < -.9
    assert value(result, "invariant.mass_angular.power_2", 0) == pytest.approx(value(result, "invariant.mass_angular.power_2", 1))


def test_transparent_pixels_and_nonfinite_damage_do_not_become_fake_shapes():
    transparent = np.zeros((64, 64, 4), np.uint8)
    damaged = np.ones((64, 64, 3), np.float32)
    damaged[:10, :, 0] = np.nan
    result = StructuralEncoder().encode([transparent, damaged, np.empty((0, 0, 3)), np.array(["bad"])])
    assert value(result, "invariant.ink_fraction", 0) == 0
    assert result.diagnostics[1]["status"] == "damaged"
    assert value(result, "quality.nonfinite_fraction", 1) > .15
    assert not result.valid[1, list(result.groups["invariant"])].any()
    assert result.diagnostics[2]["status"] == result.diagnostics[3]["status"] == "invalid"
    assert value(result, "quality.input_valid", 2) == 0
    assert np.isfinite(result.values).all()


def test_float_gray_color_and_photo_like_inputs_share_schema():
    rng = np.random.default_rng(6)
    image = asymmetric_panel()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result = StructuralEncoder().encode([image, gray, gray.astype(np.float32)/255,
                                         rng.integers(0, 256, (120, 170, 3), dtype=np.uint8)])
    np.testing.assert_array_equal(result.values[0], result.values[1])
    np.testing.assert_array_equal(result.values[1], result.values[2])
    assert all(item["status"] == "ok" for item in result.diagnostics)
    assert value(result, "quality.gray_entropy", 3) > .5
    assert value(result, "quality.saturation_mean", 3) > .1


def test_full_ink_width_is_bounded_instead_of_opencv_no_background_sentinel():
    result = StructuralEncoder().encode([np.zeros((192, 192, 3), np.uint8)])
    assert value(result, "invariant.ink_fraction") == 1
    assert value(result, "invariant.ink_width.max") <= 1
    assert result.diagnostics[0]["warnings"]


def test_shared_feature_adapter_keeps_explicit_missingness_and_names():
    encoded = StructuralEncoder().encode([panel(), asymmetric_panel()])
    batch = encoded.as_feature_batch(("blank", "drawing"))
    assert batch.image_ids == ("blank", "drawing")
    assert batch.global_features.shape == (2, len(encoded.names)*2)
    np.testing.assert_array_equal(batch.global_features[:, len(encoded.names):], encoded.valid.astype(np.float32))
    np.testing.assert_array_equal(batch.extra_views["measurements_only"], encoded.values)
    assert len(batch.metadata["feature_names"]) == batch.global_features.shape[1]
    assert "support-only" in batch.metadata["normalization"]
    assert "measurements_only" not in batch.metadata["automatic_feature_views"]
    assert "validity_indicators" not in batch.metadata["automatic_feature_views"]
    assert set(batch.metadata["automatic_feature_views"]) == {
        "global", "topology", "boundary", "morphology", "arrangement", "symmetry", "texture", "oriented"}
    for family, columns in batch.metadata["feature_families"].items():
        assert batch.extra_views[family].shape == (2, len(columns)*2)
        np.testing.assert_array_equal(batch.extra_views[family][:, len(columns):], encoded.valid[:, columns])


def test_model_agnostic_encoder_implements_common_contract():
    encoder = StructuralFeatureEncoder(side=96)
    support = encoder.encode([panel(), asymmetric_panel()], ("a", "b"))
    query = encoder.encode([asymmetric_panel()], ("query",))
    assert support.encoder_id == query.encoder_id == encoder.encoder_id
    assert support.global_features.shape[1] == query.global_features.shape[1] == len(encoder.feature_names)
    assert encoder.encode([]).global_features.shape == (0, len(encoder.feature_names))


def test_single_descriptor_has_none_for_undefined_measurements():
    descriptor = structural_descriptor(panel())
    assert descriptor["values"]["invariant.component_count"] == 0
    assert descriptor["values"]["invariant.arrangement_pair_distance.mean"] is None
    assert descriptor["diagnostics"]["status"] == "ok"


def test_strict_mode_rejects_bad_input_and_analysis_limits():
    with pytest.raises(ValueError):
        StructuralEncoder(side=16)
    with pytest.raises(ValueError):
        StructuralEncoder(strict=True).encode([np.zeros((10, 10, 2))])
