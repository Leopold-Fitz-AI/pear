import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "camera_reference", Path(__file__).resolve().parents[2] / "examples/eval/libero_camera_reference.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def calibration():
    return {
        "width": 512,
        "height": 512,
        "fovy_degrees": 90.0,
        "camera_base_cm": [0.0, 0.0, 10.0],
        "camera_to_base_rotation": np.eye(3).tolist(),
        "tool_base_cm": [0.0, 0.0, 0.0],
    }


def test_rotated_opengl_projection_and_perspective(calibration):
    pixels, valid = module.project(calibration, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 5]])
    np.testing.assert_allclose(pixels, [[255.5, 255.5], [229.9, 255.5], [255.5, 229.9], [204.3, 255.5]])
    assert valid.all()


def test_camera_center_is_not_tool_center(calibration):
    calibration["camera_base_cm"][0] = 5
    pixels, _ = module.project(calibration, [calibration["tool_base_cm"]])
    assert pixels[0, 0] == pytest.approx(383.5)


def test_behind_camera_and_offscreen_points_not_clamped(calibration):
    pixels, valid = module.project(calibration, [[0, 0, 10], [0, 0, 11], [20, 0, 0]])
    assert valid.tolist() == [False, False, True]
    assert np.isnan(pixels[:2]).all()
    assert pixels[2, 0] < 0


def test_joint_rigid_transform_preserves_projection(calibration):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    expected, _ = module.project(calibration, points)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([30.0, 50.0, 0.0])
    calibration["camera_base_cm"] = (rotation @ calibration["camera_base_cm"] + shift).tolist()
    calibration["camera_to_base_rotation"] = rotation.tolist()
    actual, _ = module.project(calibration, points @ rotation.T + shift)
    np.testing.assert_allclose(actual, expected)


def test_overlay_is_deterministic_and_does_not_mutate_raw_image(calibration):
    image = Image.new("RGB", (512, 512), "gray")
    original = image.tobytes()
    painted = module.overlay(image, calibration)
    assert image.tobytes() == original
    assert painted.tobytes() != original
    assert painted.tobytes() == module.overlay(image, calibration).tobytes()
    with pytest.raises(ValueError, match="dimensions"):
        module.overlay(Image.new("RGB", (16, 16)), calibration)


def test_capture_reads_only_robot_and_camera(calibration):
    def lookup(name, expected):
        assert name == expected
        return 0

    model = SimpleNamespace(
        body_name2id=lambda name: lookup(name, "robot0_base"),
        site_name2id=lambda name: lookup(name, "gripper0_grip_site"),
        camera_name2id=lambda name: lookup(name, "agentview"),
        cam_fovy=[90.0],
    )
    data = SimpleNamespace(
        body_xpos=np.array([[0.0, 0.0, 0.0]]),
        body_xmat=np.array([np.eye(3).ravel()]),
        cam_xpos=np.array([[0.0, 0.0, 0.1]]),
        cam_xmat=np.array([np.eye(3).ravel()]),
        site_xpos=np.array([[0.0, 0.0, 0.0]]),
    )
    scene = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data),
        robots=[SimpleNamespace(controller=SimpleNamespace(eef_name="gripper0_grip_site"))],
    )
    actual = module.capture_camera(scene, "agentview", 512, 512)
    assert actual == calibration
    data.cam_xpos[0, 0] = 9
    assert actual["camera_base_cm"][0] == 0  # snapshot, not a mutable simulation view
