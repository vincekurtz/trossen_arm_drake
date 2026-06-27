#!/usr/bin/env python

##
#
# Run a simple interactive simulation of the Trossen Stationary bimanual robot.
#
##


import numpy as np
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    EventStatus,
    MultibodyPlant,
    Parser,
    RigidTransform,
    Simulator,
    SimulatorConfig,
)

from meshcat_controller import MeshcatController
from simulation_station import SimulationStation


def add_cube(plant: MultibodyPlant):
    """Adds a small cube to and positions it just above the table."""
    Parser(plant).AddModels("models/urdf/cube.urdf")
    cube_body = plant.GetBodyByName("cube_link")
    X = RigidTransform()
    X.set_translation([0.0, 0.0, 0.02])
    plant.SetDefaultFloatingBaseBodyPose(cube_body, X)


# Set up a Drake system diagram connecting the robot and the controller
builder = DiagramBuilder()
station = builder.AddSystem(SimulationStation(add_custom_elements=add_cube))
controller = builder.AddSystem(
    MeshcatController(station.meshcat, station.plant)
)
builder.Connect(
    controller.GetOutputPort("q_des"), station.GetInputPort("q_des")
)
builder.Connect(
    controller.GetOutputPort("v_des"), station.GetInputPort("v_des")
)
diagram = builder.Build()
context = diagram.CreateDefaultContext()

# Set up the simulator to use CENIC
simulator = Simulator(diagram, context)
config = SimulatorConfig()
config.integration_scheme = "cenic"
config.accuracy = 1e-3
config.max_step_size = 0.1
config.use_error_control = True
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
print("Use the meshcat sliders to control the robot.")
print("Press the 'Stop Simulation' button to quit.")

simulator.AdvanceTo(np.inf)
