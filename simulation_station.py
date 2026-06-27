from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ApplyVisualizationConfig,
    Demultiplexer,
    Diagram,
    DiagramBuilder,
    Multiplexer,
    Parser,
    SceneGraphConfig,
    StartMeshcat,
    VisualizationConfig,
)
from pydrake.common.yaml import yaml_load_file


class SimulationStation(Diagram):
    """A Drake system representing a simulated Trossen Stationary AI robot.

    Input ports:
        - q_des: desired joint positions for the robot.
        - v_des: desired joint velocities for the robot.

    Output ports:
        - q_hat: estimated current joint positions for the robot.
        - v_hat: estimated current joint velocities for the robot.
        - TODO(vincekurtz): RGBD camera images.

    By default, the scene consists of two follower arms and a table. Override
    the add_custom_elements() method to add additional objects to the scene.
    """
    def __init__(self):
        Diagram.__init__(self)

        self.meshcat = StartMeshcat()
        builder = DiagramBuilder()

        # Load the robot model.
        self.plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.0
        )
        model_indices = Parser(self.plant).AddModels(
            "models/urdf/stationary_ai.urdf"
        )

        # Add any custom elements to the simulation scene before finalizing
        self.add_custom_elements()
        self.plant.Finalize()

        # Enable hydroelastic contact.
        scene_graph_config = SceneGraphConfig()
        scene_graph_config.default_proximity_properties.compliance_type = (
            "compliant"
        )
        scene_graph.set_config(scene_graph_config)

        # TODO(vincekurtz): add RGBD cameras.

        # Connect the visualizer
        visualization_config = VisualizationConfig()
        visualization_config.publish_proximity = True
        # TODO(vincekurtz): consider adding a long publish_period to avoid
        # forced visualization publish events here.
        ApplyVisualizationConfig(
            visualization_config, builder=builder, meshcat=self.meshcat
        )

        # Load some custom config to make meshcat look a bit better
        meshcat_config = yaml_load_file("meshcat_config.yaml")
        for p in meshcat_config["initial_properties"]:
            self.meshcat.SetProperty(p["path"], p["property"], p["value"])
        self.meshcat.SetCameraPose([0.9, 0.0, 0.9], [0.0, 0.0, 0.4])

        # Connect q_desired and v_desired input ports to the plant's desired
        # state input port. The simulator will then track these targets with
        # implicit PD control.
        nu = self.plant.num_actuators(model_indices[0])
        x_desired = builder.AddSystem(Multiplexer([nu, nu]))
        builder.Connect(
            x_desired.get_output_port(),
            self.plant.get_desired_state_input_port(model_indices[0]),
        )
        builder.ExportInput(x_desired.get_input_port(0), "q_des")
        builder.ExportInput(x_desired.get_input_port(1), "v_des")

        # Export the plant's state output ports to the diagram's output ports.
        nq = self.plant.num_positions(model_indices[0])
        nv = self.plant.num_velocities(model_indices[0])
        x_hat = builder.AddSystem(Demultiplexer([nq, nv]))
        builder.Connect(
            self.plant.get_state_output_port(model_indices[0]),
            x_hat.get_input_port(),
        )
        builder.ExportOutput(x_hat.get_output_port(0), "q_hat")
        builder.ExportOutput(x_hat.get_output_port(1), "v_hat")

        builder.BuildInto(self)

    def add_custom_elements(self):
        """Override this method to add extra simulated objects."""
        pass

