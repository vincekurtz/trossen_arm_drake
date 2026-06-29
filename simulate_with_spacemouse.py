#!/usr/bin/env python

##
#
# Run an interactive simulation of the Trossen Stationary bimanual robot,
# teleoperated with two 3Dconnexion SpaceMouse joysticks (one per arm).
#
# Data flow:
#   SpacemouseController --desired_poses--> DifferentialInverseKinematics-
#   Controller --commanded_position--> JointCommandAssembler --q_des--> station
#
##


import numpy as np
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    EventStatus,
    MultibodyPlant,
    Multiplexer,
    Parser,
    RigidTransform,
    Simulator,
    SimulatorConfig,
)

from camera_viewer import CameraViewer
from differential_ik import (
    JointCommandAssembler,
    build_differential_ik_controller,
)
from simulation_station import SimulationStation
from spacemouse_controller import SpacemouseController

TIME_STEP = 0.05


def add_cube(plant: MultibodyPlant):
    """Adds a small cube to and positions it just above the table."""
    Parser(plant).AddModels("models/urdf/cube.urdf")
    cube_body = plant.GetBodyByName("cube_link")
    X = RigidTransform()
    X.set_translation([0.0, 0.0, 0.02])
    plant.SetDefaultFloatingBaseBodyPose(cube_body, X)


# Set up a Drake system diagram connecting the robot and the controller.
builder = DiagramBuilder()

# The simulation station represents the robot and scene.
station = builder.AddSystem(SimulationStation(add_custom_elements=add_cube))

# Drake's differential IK turns desired poses into joint position commands. It
# runs on its own arm-only plant (no manipuland, not the station's plant), and
# that same plant is reused below so all the teleop systems agree on frames and
# joint ordering.
diff_ik, ik_plant, q_nominal = build_differential_ik_controller(
    time_step=TIME_STEP
)
diff_ik = builder.AddSystem(diff_ik)

# The SpaceMouse controller turns the two joysticks into desired end-effector
# poses (and gripper openings). Passing meshcat adds the "Stop Simulation"
# button used below.
spacemouse = builder.AddSystem(
    SpacemouseController(ik_plant, station.meshcat, period=TIME_STEP)
)
assembler = builder.AddSystem(JointCommandAssembler(ik_plant))

# Feed the diff IK its inputs: full estimated state [q; v] and a nominal
# (home) posture for nullspace resolution.
state_mux = builder.AddSystem(
    Multiplexer([ik_plant.num_positions(), ik_plant.num_velocities()])
)
builder.Connect(station.GetOutputPort("q_hat"), state_mux.get_input_port(0))
builder.Connect(station.GetOutputPort("v_hat"), state_mux.get_input_port(1))
builder.Connect(
    state_mux.get_output_port(), diff_ik.GetInputPort("estimated_state")
)
nominal = builder.AddSystem(ConstantVectorSource(q_nominal))
builder.Connect(
    nominal.get_output_port(), diff_ik.GetInputPort("nominal_posture")
)

# SpacemouseController -> diff IK -> assembler -> station.
builder.Connect(
    spacemouse.GetOutputPort("desired_poses"),
    diff_ik.GetInputPort("desired_poses"),
)
builder.Connect(
    diff_ik.GetOutputPort("commanded_position"),
    assembler.GetInputPort("commanded_position"),
)
builder.Connect(
    spacemouse.GetOutputPort("gripper_position"),
    assembler.GetInputPort("gripper_position"),
)
builder.Connect(assembler.GetOutputPort("q_des"), station.GetInputPort("q_des"))
builder.Connect(assembler.GetOutputPort("v_des"), station.GetInputPort("v_des"))

# Add a system that shows a pop-up window with live RGB and depth images.
camera_names = ["top_camera", "bottom_camera", "left_camera", "right_camera"]
camera_viewer = builder.AddSystem(CameraViewer(camera_names, period=0.1))
for name in camera_names:
    builder.Connect(
        station.GetOutputPort(f"{name}.rgb_image"),
        camera_viewer.GetInputPort(f"{name}.rgb_image"),
    )
    builder.Connect(
        station.GetOutputPort(f"{name}.depth_image"),
        camera_viewer.GetInputPort(f"{name}.depth_image"),
    )

diagram = builder.Build()
context = diagram.CreateDefaultContext()

# Start the diff IK position command at the current configuration so it does
# not jump when the simulation begins.
diff_ik.set_initial_position(
    diff_ik.GetMyContextFromRoot(context), q_nominal
)

# Set up the simulator to use CENIC
simulator = Simulator(diagram, context)
config = SimulatorConfig()
# config.integration_scheme = "cenic"   # DEBUG: use discrete SAP
# config.accuracy = 1e-3
# config.max_step_size = TIME_STEP
# config.use_error_control = False
ApplySimulatorConfig(config, simulator)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Listen for the "Stop Simulation" button to stop the simulation.
simulator.set_monitor(
    lambda context: (
        EventStatus.Succeeded()
        if station.meshcat.GetButtonClicks("Stop Simulation") < 1
        else EventStatus.ReachedTermination(diagram, "Stopped by user.")
    )
)

# Run the simulation.
input("Waiting for meshcat... press [ENTER] to start simulating.")
print("")
print("Use the two SpaceMice to control the robot end-effectors.")
print("Press the 'Stop Simulation' button to quit.")
simulator.AdvanceTo(np.inf)
