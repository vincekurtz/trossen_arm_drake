#!/usr/bin/env python

##
#
# Run a simple interactive simulation of the Trossen Stationary bimanual robot.
#
##

from functools import partial

import numpy as np
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    EventStatus,
    LeafSystem,
    Meshcat,
    MultibodyPlant,
    Multiplexer,
    Parser,
    RigidTransform,
    Simulator,
    SimulatorConfig,
)

from simulation_station import SimulationStation


def add_cube(plant: MultibodyPlant):
    """Adds a small cube to and positions it just above the table."""
    Parser(plant).AddModels("models/urdf/cube.urdf")
    cube_body = plant.GetBodyByName("cube_link")
    X = RigidTransform()
    X.set_translation([0.0, 0.0, 0.02])
    plant.SetDefaultFloatingBaseBodyPose(cube_body, X)

builder = DiagramBuilder()

station = builder.AddSystem(
    SimulationStation(add_custom_elements=add_cube)
)

# Add joint sliders to meshcat for setting desired joint angles.
slider_names = []
for actuator_index in station.plant.GetJointActuatorIndices():
    actuator = station.plant.get_joint_actuator(actuator_index)
    if actuator.has_controller():
        name = actuator.joint().name()
        lower_limit = actuator.joint().position_lower_limits()[0]
        upper_limit = actuator.joint().position_upper_limits()[0]
        default = actuator.joint().default_positions()[0]
        step = (upper_limit - lower_limit) / 100.0
        station.meshcat.AddSlider(
            name=name,
            min=lower_limit,
            max=upper_limit,
            step=step,
            value=default,
        )
        slider_names.append([name])
station.meshcat.AddButton("Stop Simulation")


# Add a little controller to send the slider values as joint position targets.
class MeshcatSliders(LeafSystem):
    """A system that outputs the values from meshcat sliders.

    An output port is created for each element in the list `slider_names`.
    Corresponding sliders with these names must have *already* been added to
    Meshcat via Meshcat.AddSlider().

    Adopted from https://github.com/RussTedrake/underactuated.
    """

    def __init__(self, meshcat: Meshcat, slider_names: list[str]):
        LeafSystem.__init__(self)

        self._meshcat = meshcat
        self._sliders = slider_names
        for i, slider_iterable in enumerate(self._sliders):
            port = self.DeclareVectorOutputPort(
                f"slider_group_{i}",
                len(slider_iterable),
                partial(self._DoCalcOutput, port_index=i),
            )
            port.disable_caching_by_default()

    def _DoCalcOutput(self, context, output, port_index):
        for i, slider in enumerate(self._sliders[port_index]):
            output[i] = self._meshcat.GetSliderValue(slider)


nu = station.plant.num_actuators()
assert len(slider_names) == nu, (
    "Number of sliders must match number of actuated joints."
)
sliders = builder.AddSystem(MeshcatSliders(station.meshcat, slider_names))
q_desired = builder.AddSystem(Multiplexer(nu))
v_desired = builder.AddSystem(ConstantVectorSource(np.zeros(nu)))

# Connect the sliders to the plant's desired state input port.
for i in range(nu):
    builder.Connect(sliders.get_output_port(i), q_desired.get_input_port(i))
builder.Connect(q_desired.get_output_port(), station.GetInputPort("q_des"))
builder.Connect(v_desired.get_output_port(), station.GetInputPort("v_des"))

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

# Run the simulation.
input("Waiting for meshcat... press [ENTER] to start simulating.")
print("")
print("Use the meshcat sliders to control the robot.")
print("Press the 'Stop Simulation' button to quit.")

simulator.set_monitor(
    lambda context: (
        EventStatus.Succeeded()
        if station.meshcat.GetButtonClicks("Stop Simulation") < 1
        else EventStatus.ReachedTermination(diagram, "Stopped by user.")
    )
)
simulator.AdvanceTo(np.inf)
