##
#
# Differential inverse kinematics for the bimanual Trossen robot, built on
# Drake's DifferentialInverseKinematicsController.
#
# build_differential_ik_controller() assembles the controller (one shared
# instance that tracks a Cartesian goal for each arm's end-effector), and
# JointCommandAssembler merges its arm position command with the gripper
# commands into the station's actuator-ordered q_des/v_des.
#
##

import numpy as np
from pydrake.all import (
    DifferentialInverseKinematicsController,
    DifferentialInverseKinematicsSystem,
    DofMask,
    JointLimits,
    LeafSystem,
    MultibodyPlant,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
    SpatialVelocity,
)

URDF = "models/urdf/stationary_ai.urdf"

# End-effector frames tracked by the diff IK, one per arm.
EE_FRAMES = {
    "left": "follower_left_ee_gripper_link",
    "right": "follower_right_ee_gripper_link",
}


def arm_velocity_indices(plant: MultibodyPlant) -> list[int]:
    """Velocity (= position, since v = q̇) indices of the 12 arm joints."""
    return [
        plant.GetJointByName(f"follower_{side}_joint_{i}").velocity_start()
        for side in ("left", "right")
        for i in range(6)
    ]


def goal_frame_name(plant: MultibodyPlant, side: str) -> str:
    """Fully-scoped (model::frame) name of an arm's end-effector frame."""
    return plant.GetFrameByName(EE_FRAMES[side]).scoped_name().to_string()


def build_differential_ik_controller(
    time_step: float = 0.05,
    max_linear_speed: float = 1.0,
    max_angular_speed: float = 2.0,
):
    """Build a DifferentialInverseKinematicsController for both arms.

    Returns (controller, plant, default_positions), where `plant` is the
    arm-only MultibodyPlant the IK runs on. This is a fresh model parsed from
    the URDF -- deliberately *not* the station's plant, which also contains the
    manipuland(s) and will eventually be swapped for hardware. The same arm-only
    plant should be used to build the SpacemouseController and the
    JointCommandAssembler so their frames and joint ordering stay consistent.

    The controller tracks one Cartesian goal per end-effector frame; goals are
    supplied on its "desired_poses" input port as a BusValue keyed by
    goal_frame_name(side).

    The caller should:
        - feed "estimated_state" with the full plant state [q; v],
        - feed "nominal_posture" with a reference posture (e.g. the returned
          default_positions),
        - call controller.set_initial_position(context, default_positions) on
          the simulation context so the command starts at the current config,
        - read the 12-dof "commanded_position" output (active arm dofs, in
          ascending joint order: left 0-5 then right 0-5).
    """
    # A RobotDiagram + collision checker is required by the diff IK system.
    builder = RobotDiagramBuilder(time_step=0.0)
    model = builder.parser().AddModels(URDF)[0]
    robot = builder.Build()

    checker = SceneGraphCollisionChecker(
        model=robot, robot_model_instances=[model], edge_step_size=0.05
    )
    # Use the checker's (owned) plant so the returned reference stays valid for
    # the lifetime of the controller.
    plant = checker.plant()

    # Command only the 12 arm joints; the gripper carriages stay passive here.
    active = arm_velocity_indices(plant)
    active_dof = DofMask([i in active for i in range(plant.num_velocities())])

    # Recipe: track the goals (least squares) with nullspace posture centering,
    # plus a joint-velocity-limit constraint. The constraint is essential: it
    # forbids any commanded velocity that would push a joint past its position
    # limits, so when the target leaves the reachable workspace the command
    # saturates at full extension instead of winding up and slamming the arm
    # into the table.
    S = DifferentialInverseKinematicsSystem
    recipe = S.Recipe()
    recipe.AddIngredient(S.LeastSquaresCost(S.LeastSquaresCost.Config()))
    recipe.AddIngredient(S.JointCenteringCost(S.JointCenteringCost.Config()))
    recipe.AddIngredient(
        S.JointVelocityLimitConstraint(
            S.JointVelocityLimitConstraint.Config(),
            JointLimits(plant, active_dof),
        )
    )

    ik_system = S(
        recipe,
        plant.world_frame().scoped_name().to_string(),  # task frame T = world
        checker,
        active_dof,
        time_step,
        1.0,  # K_VX: desired velocity exactly reaches the goal in one step
        SpatialVelocity(
            np.full(3, max_angular_speed), np.full(3, max_linear_speed)
        ),
    )
    controller = DifferentialInverseKinematicsController(
        ik_system, planar_rotation_dof_indices=[]
    )
    return controller, plant, plant.GetDefaultPositions()


class JointCommandAssembler(LeafSystem):
    """Merge the diff IK arm command and gripper commands into q_des/v_des.

    Input ports:
        - commanded_position: 12 arm joint positions from the diff IK
          controller (left 0-5 then right 0-5).
        - gripper_position: 2 gripper joint positions (left, right).

    Output ports:
        - q_des: desired joint positions in the station's actuator order.
        - v_des: desired joint velocities (fixed at zero; PD tracks position).
    """

    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)

        active = arm_velocity_indices(plant)
        gripper_q = [
            plant.GetJointByName(
                f"follower_{side}_left_carriage_joint"
            ).position_start()
            for side in ("left", "right")
        ]

        # For each actuated joint (in actuator order), record whether its
        # position comes from the arm command or the gripper command, and the
        # corresponding source index.
        self._sources = []  # list of (is_gripper, source_index)
        for ai in plant.GetJointActuatorIndices():
            actuator = plant.get_joint_actuator(ai)
            if not actuator.has_controller():
                continue
            q_idx = actuator.joint().position_start()
            if q_idx in active:
                self._sources.append((False, active.index(q_idx)))
            else:
                self._sources.append((True, gripper_q.index(q_idx)))
        self._num_actuators = len(self._sources)

        self.DeclareVectorInputPort("commanded_position", len(active))
        self.DeclareVectorInputPort("gripper_position", len(gripper_q))
        self.DeclareVectorOutputPort(
            "q_des", self._num_actuators, self._calc_q_des
        )
        self.DeclareVectorOutputPort(
            "v_des", self._num_actuators, self._calc_v_des
        )

    def _calc_q_des(self, context, output):
        arm = self.get_input_port(0).Eval(context)
        grip = self.get_input_port(1).Eval(context)
        q = np.array(
            [grip[i] if is_grip else arm[i] for is_grip, i in self._sources]
        )
        output.SetFromVector(q)

    def _calc_v_des(self, context, output):
        output.SetFromVector(np.zeros(self._num_actuators))
