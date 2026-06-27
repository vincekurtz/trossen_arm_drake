from functools import partial

import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    Meshcat,
    MultibodyPlant,
    Multiplexer,
)


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

class MeshcatController(Diagram):
    """A system that reads joint targets from meshcat sliders.

    Outputs position and velocity targets for each PD-controlled joint in the
    given plant. Adds sliders for each joint to the given meshcat instance,
    along with a "Stop Simulation" button.

    Output ports:
        - q_des: desired joint positions for the robot.
        - v_des: desired joint velocities for the robot (fixed at zero).

    """
    def __init__(self, meshcat: Meshcat, plant: MultibodyPlant):
        Diagram.__init__(self)
        builder = DiagramBuilder()

        # Add sliders for each PD-controlled joint.
        slider_names = []
        for actuator_index in plant.GetJointActuatorIndices():
            actuator = plant.get_joint_actuator(actuator_index)
            if actuator.has_controller():
                name = actuator.joint().name()
                lower_limit = actuator.joint().position_lower_limits()[0]
                upper_limit = actuator.joint().position_upper_limits()[0]
                default = actuator.joint().default_positions()[0]
                step = (upper_limit - lower_limit) / 100.0
                meshcat.AddSlider(
                    name=name,
                    min=lower_limit,
                    max=upper_limit,
                    step=step,
                    value=default,
                )
                slider_names.append([name])
        meshcat.AddButton("Stop Simulation")

        nu = plant.num_actuators()
        assert len(slider_names) == nu, (
            "Number of sliders must match number of actuated joints."
        )

        # Convert slider values into Drake system output values.
        sliders = builder.AddSystem(MeshcatSliders(meshcat, slider_names))

        # Concatenate slider values into unified q_des, v_des output ports.
        q_desired = builder.AddSystem(Multiplexer(nu))
        v_desired = builder.AddSystem(ConstantVectorSource(np.zeros(nu)))
        for i in range(nu):
            builder.Connect(
                sliders.get_output_port(i), q_desired.get_input_port(i)
            )

        builder.ExportOutput(q_desired.get_output_port(), "q_des")
        builder.ExportOutput(v_desired.get_output_port(), "v_des")

        builder.BuildInto(self)




