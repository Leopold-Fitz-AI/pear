"""Optional camera/robot calibration references; NEVER reads task-object state.

The reference grid is at base z=0, NOT a depth map or detected support surface.
It provides metric image cues without object poses, masks, contacts or goals.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

GUIDANCE = """
Calibrated visual references (assisted protocol):
- Cyan XY grid: robot-base coordinates in centimeters on the REFERENCE plane z=0.
  Labels such as 40,20 mean x=40, y=20. This plane is NOT detected object depth or
  necessarily the support surface. Raised objects have parallax: use both views.
- Red TCP marks the measured tool-point projection. Magenta 0/5/10 marks the
  projections of the SAME tool XY at reference heights z=0/5/10 cm. They are
  geometric references, not object locations, detected grasps or free-space claims.
- The wrist camera is offset from the tool. Image center is NOT the grasp point.
  An object appearing large or centered does NOT establish alignment. Use the
  metric grid for coarse XY positioning while high, then both views and the actual
  finger tips for small corrections at the estimated object/rim height.
- After closing, test-lift only 3 cm and compare the object's position in BOTH
  views with the prior observation. A changed finger gap alone is NOT proof of a
  grasp; a thin rim CAN be held with a small gap. If the object stays on the table,
  reopen, retreat and correct XY; do not repeat long empty lifts.
- Keep clearance during transport. Verify the OBJECT (not only the tool) is above
  the destination in BOTH views before lowering. Lower in small increments near
  contact. Do not drag at table height to correct a large destination error.
- For pushing, approach the side opposite the intended travel; test a short push
  and verify object displacement. Moving the hand without object displacement is
  not progress. Reposition rather than repeat a long ineffective sweep.
""".strip()


def capture_camera(scene, camera, width, height):
    """Only calibrated camera intrinsics/extrinsics and robot kinematics."""
    sim = scene.sim
    base_id = sim.model.body_name2id("robot0_base")
    base_position = sim.data.body_xpos[base_id]
    base_rotation = sim.data.body_xmat[base_id].reshape(3, 3)
    camera_id = sim.model.camera_name2id(camera)
    tool_name = scene.robots[0].controller.eef_name
    tool_id = sim.model.site_name2id(tool_name)
    return {
        "width": int(width),
        "height": int(height),
        "fovy_degrees": float(sim.model.cam_fovy[camera_id]),
        "camera_base_cm": (base_rotation.T @ (sim.data.cam_xpos[camera_id] - base_position) * 100).tolist(),
        "camera_to_base_rotation": (base_rotation.T @ sim.data.cam_xmat[camera_id].reshape(3, 3)).tolist(),
        "tool_base_cm": (base_rotation.T @ (sim.data.site_xpos[tool_id] - base_position) * 100).tolist(),
    }


def calibrated_scene_class(scene_class):
    """Capture calibration at image sampling, not after physics has advanced."""

    class CalibratedScene(scene_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.camera_references = {}
            render = self.sim.render

            def render_with_reference(*args, **kwargs):
                camera = kwargs.get("camera_name")
                if camera in ("agentview", "robot0_eye_in_hand"):
                    self.camera_references[camera] = capture_camera(self, camera, kwargs["width"], kwargs["height"])
                return render(*args, **kwargs)

            self.sim.render = render_with_reference

    return CalibratedScene


def project(calibration, points):
    """Base-cm points -> pixels in PEAR's 180-degree-rotated OpenGL RGB frame.

    MuJoCo camera looks along local -z. Raw OpenGL is bottom-up; PEAR rotates
    both image axes. Off-camera/behind-camera points are NOT clipped into view.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    camera = np.asarray(calibration["camera_base_cm"])
    rotation = np.asarray(calibration["camera_to_base_rotation"])
    q = (points - camera) @ rotation
    depth = -q[:, 2]
    valid = np.isfinite(q).all(axis=1) & (depth > 1e-5)
    f = calibration["height"] / (2 * math.tan(math.radians(calibration["fovy_degrees"]) / 2))
    uv = np.full((len(points), 2), np.nan)
    center = (np.array([calibration["width"], calibration["height"]]) - 1) / 2
    uv[valid] = center - f * q[valid, :2] / depth[valid, None]
    return uv, valid


def overlay(image, calibration):
    """Paint only a known reference grid and robot markers; no object inference."""
    if image.size != (calibration["width"], calibration["height"]):
        raise ValueError("Image and calibration dimensions differ")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = image.size

    def visible(uv):
        return np.isfinite(uv).all() and 0 <= uv[0] < w and 0 <= uv[1] < h

    def polyline(points, color):
        pixels, valid = project(calibration, points)
        # Segment densely to avoid drawing across the near-plane singularity.
        for i in range(len(pixels) - 1):
            if valid[i] and valid[i + 1] and visible(pixels[i]) and visible(pixels[i + 1]):
                draw.line([tuple(pixels[i]), tuple(pixels[i + 1])], fill=color, width=1)

    for x in range(20, 91, 10):
        polyline([[x, y, 0] for y in np.linspace(-40, 40, 161)], (0, 200, 255, 85))
    for y in range(-40, 41, 10):
        polyline([[x, y, 0] for x in np.linspace(20, 90, 141)], (0, 200, 255, 85))
    for x in range(20, 81, 20):
        for y in range(-40, 41, 20):
            pixels, valid = project(calibration, [[x, y, 0]])
            if valid[0] and visible(pixels[0]):
                draw.text(
                    tuple(pixels[0] + [2, 2]),
                    f"{x},{y}",
                    fill=(190, 255, 255, 240),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0, 200),
                )
    tool = calibration["tool_base_cm"]
    markers = [(tool, "TCP", (255, 70, 70, 255))]
    markers += [([tool[0], tool[1], z], str(z), (255, 90, 255, 255)) for z in (0, 5, 10)]
    for point, name, color in markers:
        pixels, valid = project(calibration, [point])
        if valid[0] and visible(pixels[0]):
            u, v = pixels[0]
            draw.line([(u - 5, v), (u + 5, v)], fill=color, width=2)
            draw.line([(u, v - 5), (u, v + 5)], fill=color, width=2)
            draw.text((u + 6, v - 10), name, fill=color, stroke_width=1, stroke_fill=(0, 0, 0, 220))
    draw.rectangle((0, 0, w, 16), fill=(0, 0, 0, 190))
    draw.text((3, 2), "Reference XY grid: z=0 cm (NOT object depth) | red=TCP", fill="white")
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
