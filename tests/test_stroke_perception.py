"""Shape-free stroke perception: relations are intrinsic, not screen angles."""
import cv2
import numpy as np
import pytest

from src.vision.stroke_graph import analyze_stroke, thin


def draw_curve(kind="s", directions=(1, 1), positions=(.23, .72), thickness=3):
    t = np.linspace(0, 1, 600)
    if kind == "s":
        points = np.column_stack([160+72*np.sin(2*np.pi*t), 30+250*t])
        closed = False
    elif kind == "u":
        theta = np.linspace(.1*np.pi, .9*np.pi, 600)
        points = np.column_stack([160+112*np.cos(theta), 45+185*np.sin(theta)])
        closed = False
    elif kind == "loop":
        theta = t*2*np.pi
        points = np.column_stack([160+108*np.cos(theta), 160+75*np.sin(theta)])
        closed = True
    elif kind == "straight":
        points = np.column_stack([30+260*t, 70+190*t])
        closed = False
    else:
        raise ValueError(kind)
    mask = np.zeros((320, 320), np.uint8)
    cv2.polylines(mask, [points.astype(np.int32)], closed, 255, thickness, cv2.LINE_AA)
    for position, direction in zip(positions, directions):
        i = round(position*(len(points)-1))
        tangent = points[min(len(points)-1, i+3)]-points[max(0, i-3)]
        tangent = tangent/np.linalg.norm(tangent)*direction
        normal = np.array([-tangent[1], tangent[0]])
        triangle = np.array([points[i]+tangent*10,
                             points[i]-tangent*7+normal*7,
                             points[i]-tangent*7-normal*7])
        cv2.fillPoly(mask, [triangle.astype(np.int32)], 255, cv2.LINE_AA)
    return ((mask > 100)*255).astype(np.uint8)


@pytest.mark.parametrize("kind", ["s", "u", "loop", "straight"])
@pytest.mark.parametrize("directions,expected", [((1, 1), "same"), ((-1, -1), "same"),
                                                ((1, -1), "opposed"), ((-1, 1), "opposed")])
def test_arrow_relation_uses_path_traversal(kind, directions, expected):
    path = analyze_stroke(draw_curve(kind, directions))
    assert path.topology == ("closed" if kind == "loop" else "open")
    assert len(path.arrows) == 2
    assert path.arrow_relation == expected


@pytest.mark.parametrize("kind", ["s", "u", "loop", "straight"])
def test_unmarked_curve_does_not_invent_arrows(kind):
    path = analyze_stroke(draw_curve(kind, directions=()))
    assert not path.arrows
    assert path.arrow_relation == "unknown"


def test_single_arrow_cannot_establish_direction_agreement():
    path = analyze_stroke(draw_curve(directions=(1,)))
    assert len(path.arrows) == 1
    assert path.arrow_relation == "unknown"


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_quarter_turn_and_mirror_preserve_intrinsic_relation(turns):
    mask = draw_curve("u", (1, 1))
    transformed = np.rot90(mask, turns)[:, ::-1].copy()
    assert analyze_stroke(transformed).arrow_relation == "same"


@pytest.mark.parametrize("kind,angle,relation,directions", [
    ("u", 37, "same", (1, 1)), ("s", -23, "opposed", (1, -1)),
    ("loop", 61, "opposed", (-1, 1)), ("straight", 14, "same", (-1, -1))])
def test_non_axis_rotation_of_visible_heads(kind, angle, relation, directions):
    mask = draw_curve(kind, directions)
    rotation = cv2.getRotationMatrix2D((160, 160), angle, 1)
    transformed = cv2.warpAffine(mask, rotation, (320, 320), flags=cv2.INTER_LINEAR)
    transformed = ((transformed > 100)*255).astype(np.uint8)
    assert analyze_stroke(transformed).arrow_relation == relation


def test_symmetric_widening_is_unknown_instead_of_an_arrow_direction():
    mask = np.zeros((200, 200), np.uint8)
    cv2.line(mask, (20, 100), (180, 100), 255, 3)
    cv2.ellipse(mask, (65, 100), (10, 6), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (140, 100), (10, 6), 0, 0, 360, 255, -1)
    path = analyze_stroke(mask)
    assert path.arrow_relation == "unknown"
    assert all(a.direction == 0 for a in path.arrows)


def test_large_branch_is_unresolved_not_forced_to_a_curve():
    mask = np.zeros((200, 200), np.uint8)
    for endpoint in ((30, 30), (170, 30), (100, 180)):
        cv2.line(mask, (100, 90), endpoint, 255, 3)
    path = analyze_stroke(mask)
    assert path.topology == "branched"
    assert path.arrow_relation == "unknown"


def test_short_real_branch_is_not_silently_pruned_as_an_arrow_wing():
    mask = np.zeros((200, 200), np.uint8)
    cv2.line(mask, (30, 100), (170, 100), 255, 3)
    cv2.line(mask, (100, 100), (100, 89), 255, 3)
    path = analyze_stroke(mask)
    assert path.topology == "branched"
    assert path.graph_count("endpoint") == 3
    assert path.graph_count("junction") == 1
    assert path.arrow_relation == "unknown"


@pytest.mark.parametrize("thickness", [6, 8, 12, 16, 20])
def test_thick_v_join_does_not_become_two_opposing_arrows(thickness):
    mask = np.zeros((260, 260), np.uint8)
    cv2.polylines(mask, [np.array(((65, 180), (130, 75), (195, 180)), np.int32)], False, 255, thickness)
    path = analyze_stroke(mask)
    assert path is not None
    assert path.arrow_relation == "unknown"
    assert not any(a.reliable for a in path.arrows)


def test_empty_mask_and_filled_blob_have_no_direction_relation():
    blank = np.zeros((100, 100), np.uint8)
    assert analyze_stroke(blank) is None
    cv2.circle(blank, (50, 50), 20, 255, -1)
    path = analyze_stroke(blank)
    assert path is None or path.arrow_relation == "unknown"


@pytest.mark.parametrize("points", [((20, 160), (100, 30), (180, 160)),
                                   ((30, 30), (170, 30), (170, 170), (30, 170)),
                                   ((25, 70), (85, 20), (175, 60), (145, 160), (65, 170))])
def test_closed_cornered_outline_has_no_arrows(points):
    mask = np.zeros((200, 200), np.uint8)
    cv2.polylines(mask, [np.array(points, np.int32)], True, 255, 3)
    path = analyze_stroke(mask)
    assert path is not None
    assert path.topology == "closed"
    assert not path.arrows
    assert path.arrow_relation == "unknown"


def test_sharp_open_bends_are_not_arrowheads():
    mask = np.zeros((200, 200), np.uint8)
    cv2.polylines(mask, [np.array(((20, 40), (160, 90), (50, 160)), np.int32)], False, 255, 3)
    path = analyze_stroke(mask)
    assert path.topology == "open"
    assert not path.arrows


def test_thinning_preserves_main_loop_hole():
    mask = draw_curve("loop", directions=())
    skeleton = thin(mask)
    _, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    assert hierarchy is not None
    assert any(row[3] >= 0 for row in hierarchy[0])
