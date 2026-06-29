##
#
# Teleoperation front-end that maps two 3Dconnexion SpaceMouse joysticks to
# end-effector pose setpoints for the bimanual Trossen Stationary AI robot.
#
# The left SpaceMouse drives the left arm, the right one drives the right arm.
# Each device acts as a rate controller: deflecting it moves the corresponding
# end-effector pose setpoint. This system only *produces the setpoints*; a
# Drake DifferentialInverseKinematicsController (see differential_ik.py) reads
# them and computes the joint commands that the station tracks.
#
##

import contextlib
import threading
import time

import numpy as np
import pyspacemouse
from pydrake.all import (
    AbstractValue,
    AngleAxis,
    BusValue,
    Cylinder,
    LeafSystem,
    Meshcat,
    MultibodyPlant,
    Rgba,
    RigidTransform,
    RotationMatrix,
)

from differential_ik import EE_FRAMES, goal_frame_name
from spacemouse import list_spacemouse_paths


class SpacemouseController(LeafSystem):
    """A LeafSystem that turns two SpaceMice into end-effector pose setpoints.

    Each SpaceMouse integrates a desired pose for one arm's end-effector, plus
    a gripper opening. The desired poses are published as a BusValue keyed by
    fully-scoped frame name, ready to feed a DifferentialInverseKinematics-
    Controller's "desired_poses" input port.

    Output ports:
        - desired_poses: BusValue of RigidTransformd, one per end-effector
          frame, expressed in the world frame.
        - gripper_position: desired gripper joint positions (left, right).

    SpaceMouse mapping (per arm):
        - translate (x, y, z) -> end-effector translation in the world frame.
        - tilt/twist (roll, pitch, yaw) -> end-effector rotation about its own
          axes.
        - button 0 -> close gripper, button 1 -> open gripper (held to move).

    If `show_pose_targets` is True and a `meshcat` is given, a coordinate triad
    is drawn at each desired end-effector pose.
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        meshcat: Meshcat = None,
        *,
        period: float = 0.05,
        max_linear_speed: float = 0.15,
        max_angular_speed: float = 0.8,
        gripper_speed: float = 0.05,
        deadband: float = 0.05,
        require_devices: bool = True,
        show_pose_targets: bool = True,
    ):
        LeafSystem.__init__(self)

        # A private plant context is used purely for the initial forward
        # kinematics that seed the pose setpoints; it is never advanced in time.
        plant_context = plant.CreateDefaultContext()

        # The update period must divide the simulator's max_step_size and stay
        # large enough that the station's implicit-PD/CENIC integrator keeps the
        # arms in their upright gravity equilibrium. 0.05 s (with the default
        # max_step_size of 0.1 s) is validated; much smaller values can knock
        # the arms into a hanging equilibrium. See simulate.py's config.
        self._period = period
        self._max_linear_speed = max_linear_speed
        self._max_angular_speed = max_angular_speed
        self._gripper_speed = gripper_speed
        self._deadband = deadband

        # --- Per-arm bookkeeping -----------------------------------------
        q_default = plant.GetDefaultPositions()
        plant.SetPositions(plant_context, q_default)
        self._arms = []
        for side in ("left", "right"):
            grip_joint = plant.GetJointByName(
                f"follower_{side}_left_carriage_joint"
            )
            ee_frame = plant.GetFrameByName(EE_FRAMES[side])
            self._arms.append(
                {
                    "side": side,
                    "goal_frame": goal_frame_name(plant, side),
                    "grip_lower": grip_joint.position_lower_limits()[0],
                    "grip_upper": grip_joint.position_upper_limits()[0],
                    "X0": ee_frame.CalcPoseInWorld(plant_context),
                    "grip0": q_default[grip_joint.position_start()],
                    "target_path": f"/spacemouse_target/{side}",
                }
            )

        # --- State -------------------------------------------------------
        # One desired end-effector pose per arm (seeded from FK at q_default).
        self._pose_state = [
            self.DeclareAbstractState(AbstractValue.Make(arm["X0"]))
            for arm in self._arms
        ]
        # Desired gripper positions (left, right).
        self._grip_state = self.DeclareDiscreteState(
            np.array([arm["grip0"] for arm in self._arms])
        )

        # --- Devices -----------------------------------------------------
        self._devices = self._open_devices(
            num_devices=len(self._arms), require_devices=require_devices
        )

        # A background thread continuously drains the HID buffers into
        # self._latest so the control update reads a fresh state without
        # blocking. read() returns one report per call and the devices emit
        # them at ~100+ Hz; reading only once per control tick (20 Hz) would
        # let the buffers back up and the inputs lag further and further behind.
        self._latest = [(np.zeros(6), []) for _ in self._arms]
        self._reader_stop = threading.Event()
        self._reader_thread = None
        if self._devices:
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True
            )
            self._reader_thread.start()

        if meshcat is not None:
            meshcat.AddButton("Stop Simulation")

        # Optionally draw a coordinate triad at each desired pose in meshcat.
        self._meshcat = meshcat
        self._show_targets = show_pose_targets and meshcat is not None
        if self._show_targets:
            for arm in self._arms:
                self._add_triad(meshcat, arm["target_path"])

        # --- Ports and update --------------------------------------------
        self.DeclareAbstractOutputPort(
            "desired_poses",
            lambda: AbstractValue.Make(BusValue()),
            self._calc_desired_poses,
        )
        self.DeclareVectorOutputPort(
            "gripper_position", len(self._arms), self._calc_gripper
        )

        self.DeclarePeriodicUnrestrictedUpdateEvent(period, 0.0, self._update)
        if self._show_targets:
            self.DeclarePeriodicPublishEvent(period, 0.0, self._draw_targets)

    # ---------------------------------------------------------------------
    # Device handling
    # ---------------------------------------------------------------------
    @staticmethod
    def _open_devices(num_devices: int, require_devices: bool) -> list:
        """Open `num_devices` distinct physical SpaceMouse devices.

        Devices are opened by their hidraw path, which correctly distinguishes
        multiple physical mice (pyspacemouse.open(device_index) indexes HID
        usage collections rather than devices, so it cannot). The returned list
        is ordered by path -- swap the receivers (or the list) if the left/right
        assignment comes out reversed.

        If `require_devices` is False, too few devices yields an empty list so
        the controller runs with zero input (useful for testing/headless).
        """
        paths = list_spacemouse_paths()
        if len(paths) < num_devices:
            if require_devices:
                raise RuntimeError(
                    f"Found {len(paths)} physical SpaceMouse device(s), need "
                    f"{num_devices}: {paths}. If devices are connected but not "
                    "listed, check udev permissions (see 99-spacemouse.rules)."
                )
            return []
        return [pyspacemouse.open_by_path(p) for p in paths[:num_devices]]

    def _reader_loop(self):
        """Continuously drain the SpaceMice, caching the latest reading.

        Runs in a daemon thread. Each cached entry is a (twist, buttons) tuple
        where twist is [vx, vy, vz, wx, wy, wz] in [-1, 1] after deadbanding.
        Assigning the whole tuple is atomic under the GIL, so the control
        thread always sees a consistent snapshot.
        """
        while not self._reader_stop.is_set():
            for i, dev in enumerate(self._devices):
                s = dev.read()
                twist = np.array([s.x, s.y, s.z, s.roll, s.pitch, s.yaw])
                twist[np.abs(twist) < self._deadband] = 0.0
                self._latest[i] = (twist, list(s.buttons))
            time.sleep(0.001)

    def close(self):
        """Stop the reader thread and close the devices."""
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        for dev in self._devices:
            dev.close()
        self._devices = []

    def __del__(self):
        # Best-effort cleanup if the controller is garbage-collected.
        with contextlib.suppress(Exception):
            self.close()

    def _read_inputs(self) -> list:
        """Return the latest cached (twist, buttons) per arm.

        Missing devices report zero motion and no buttons. The actual hardware
        reads happen in the background reader thread (see _reader_loop).
        """
        return list(self._latest)

    # ---------------------------------------------------------------------
    # Periodic update: integrate the pose setpoints and gripper commands
    # ---------------------------------------------------------------------
    def _update(self, context, state):
        grip = context.get_discrete_state(self._grip_state).get_value().copy()
        inputs = self._read_inputs()

        for i, (arm, pose_idx, (twist, buttons)) in enumerate(
            zip(self._arms, self._pose_state, inputs, strict=True)
        ):
            # Integrate the end-effector pose setpoint from the twist.
            X_des = context.get_abstract_state(pose_idx).get_value()
            X_des = self._integrate_setpoint(X_des, twist)
            state.get_mutable_abstract_state(pose_idx).set_value(X_des)

            # Gripper: button 0 closes, button 1 opens (held to keep moving).
            direction = 0.0
            if len(buttons) >= 1 and buttons[0]:
                direction -= 1.0
            if len(buttons) >= 2 and buttons[1]:
                direction += 1.0
            grip[i] = np.clip(
                grip[i] + direction * self._gripper_speed * self._period,
                arm["grip_lower"],
                arm["grip_upper"],
            )

        state.get_mutable_discrete_state(self._grip_state).set_value(grip)

    def _integrate_setpoint(
        self, X_des: RigidTransform, twist: np.ndarray
    ) -> RigidTransform:
        """Advance an end-effector pose setpoint by one SpaceMouse twist."""
        v_lin = twist[:3] * self._max_linear_speed
        w_body = twist[3:] * self._max_angular_speed

        # Translation is applied in the world frame.
        p_new = X_des.translation() + v_lin * self._period

        # Rotation is applied about the end-effector's own axes (body frame).
        theta = np.linalg.norm(w_body) * self._period
        if theta > 1e-9:
            axis = w_body / np.linalg.norm(w_body)
            R_new = X_des.rotation() @ RotationMatrix(AngleAxis(theta, axis))
        else:
            R_new = X_des.rotation()
        return RigidTransform(R_new, p_new)

    # ---------------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------------
    def _calc_desired_poses(self, context, output):
        bus = BusValue()
        for arm, pose_idx in zip(self._arms, self._pose_state, strict=True):
            X_des = context.get_abstract_state(pose_idx).get_value()
            bus.Set(arm["goal_frame"], AbstractValue.Make(X_des))
        output.set_value(bus)

    def _calc_gripper(self, context, output):
        output.SetFromVector(
            context.get_discrete_state(self._grip_state).get_value()
        )

    # ---------------------------------------------------------------------
    # Meshcat pose-target visualization
    # ---------------------------------------------------------------------
    @staticmethod
    def _add_triad(meshcat, path, length=0.1, radius=0.004):
        """Draw a static RGB coordinate triad (X, Y, Z) under `path`.

        Each axis is a cylinder whose local pose orients its +z along the axis;
        the whole triad is later positioned with SetTransform(path, X)."""
        axes = {
            "x": (Rgba(1, 0, 0, 1), RotationMatrix.MakeYRotation(np.pi / 2)),
            "y": (Rgba(0, 1, 0, 1), RotationMatrix.MakeXRotation(-np.pi / 2)),
            "z": (Rgba(0, 0, 1, 1), RotationMatrix()),
        }
        for name, (color, R) in axes.items():
            meshcat.SetObject(
                f"{path}/{name}", Cylinder(radius, length), color
            )
            # Offset the cylinder so it starts at the frame origin.
            meshcat.SetTransform(
                f"{path}/{name}", RigidTransform(R, R @ [0, 0, length / 2])
            )

    def _draw_targets(self, context):
        for arm, pose_idx in zip(self._arms, self._pose_state, strict=True):
            X_des = context.get_abstract_state(pose_idx).get_value()
            self._meshcat.SetTransform(arm["target_path"], X_des)
