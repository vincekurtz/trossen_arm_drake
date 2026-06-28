from collections.abc import Callable

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ApplyVisualizationConfig,
    CameraInfo,
    ClippingRange,
    ColorRenderCamera,
    Demultiplexer,
    DepthRange,
    DepthRenderCamera,
    Diagram,
    DiagramBuilder,
    MakeRenderEngineVtk,
    MultibodyPlant,
    Multiplexer,
    Parser,
    RenderCameraCore,
    RenderEngineVtkParams,
    RgbdSensor,
    RgbdSensorDiscrete,
    RigidTransform,
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
        - {name}_camera.rgb_image: RGB image from the named RGBD camera.
        - {name}_camera.depth_image: depth image from the named RGBD camera.

    where {name} is one of "top", "bottom", "left", or "right".

    By default, the scene consists of two follower arms and a table. Pass an
    add_custom_elements(plant) function to the constructor to add additional
    objects to the scene.
    """

    def __init__(
        self,
        add_custom_elements: Callable[[MultibodyPlant], None] = None
    ):
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
        if add_custom_elements is not None:
            add_custom_elements(self.plant)
        self.plant.Finalize()

        # Enable hydroelastic contact.
        scene_graph_config = SceneGraphConfig()
        scene_graph_config.default_proximity_properties.compliance_type = (
            "compliant"
        )
        scene_graph.set_config(scene_graph_config)

        # Set up a renderer for the cameras
        renderer_name = "renderer"
        scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

        # Intrinsics roughly matching the RealSense D405 color stream.
        intrinsics = CameraInfo(
            width=640, height=480, fov_y=np.radians(58.0)
        )
        camera_core = RenderCameraCore(
            renderer_name,
            intrinsics,
            ClippingRange(0.01, 10.0),
            RigidTransform(),
        )
        color_camera = ColorRenderCamera(camera_core, show_window=False)
        depth_camera = DepthRenderCamera(
            camera_core, DepthRange(0.07, 5.0)
        )

        # Add an RGBD camera at each of the optical frames defined in the URDF.
        camera_optical_frames = {
            "top": "cam_high_color_optical_frame",
            "bottom": "cam_low_color_optical_frame",
            "left": "follower_left_camera_color_optical_frame",
            "right": "follower_right_camera_color_optical_frame",
        }
        for name, frame_name in camera_optical_frames.items():
            optical_frame = self.plant.GetBodyByName(frame_name)
            camera = builder.AddSystem(
                RgbdSensorDiscrete(
                    RgbdSensor(
                        parent_id=self.plant.GetBodyFrameIdOrThrow(
                            optical_frame.index()
                        ),
                        X_PB=RigidTransform(),
                        color_camera=color_camera,
                        depth_camera=depth_camera,
                    ),
                    period=0.1,  # 10 Hz refresh rate
                )
            )
            builder.Connect(
                scene_graph.get_query_output_port(),
                camera.query_object_input_port(),
            )
            builder.ExportOutput(
                camera.color_image_output_port(), f"{name}_camera.rgb_image"
            )
            builder.ExportOutput(
                camera.depth_image_32F_output_port(),
                f"{name}_camera.depth_image",
            )

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
