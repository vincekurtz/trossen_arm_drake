import matplotlib.pyplot as plt
from pydrake.all import (
    AbstractValue,
    ImageDepth32F,
    ImageRgba8U,
    LeafSystem,
)


class CameraViewer(LeafSystem):
    """A system that displays a live view of one or more RGBD cameras.

    For each camera name, two abstract input ports are created:
        - {name}.rgb_image:   an ImageRgba8U color image.
        - {name}.depth_image: an ImageDepth32F depth image.

    The images are drawn with matplotlib at a fixed refresh rate, one row of
    (color, depth) subplots per camera.
    """

    def __init__(self, camera_names: list[str], period: float = 0.1):
        LeafSystem.__init__(self)

        self._camera_names = list(camera_names)

        # Declare an rgb + depth input port for each camera. Port names match
        # the SimulationStation output port naming convention.
        for name in self._camera_names:
            self.DeclareAbstractInputPort(
                f"{name}.rgb_image", AbstractValue.Make(ImageRgba8U())
            )
            self.DeclareAbstractInputPort(
                f"{name}.depth_image", AbstractValue.Make(ImageDepth32F())
            )

        # Set up a matplotlib figure with one row per camera and two columns
        # (color, depth). The image artists are created lazily on first draw.
        plt.ion()
        n = len(self._camera_names)
        self._fig, axes = plt.subplots(
            n, 2, squeeze=False, figsize=(8, 4 * n)
        )
        self._axes = axes
        self._color_artists = [None] * n
        self._depth_artists = [None] * n
        for row, name in enumerate(self._camera_names):
            self._axes[row, 0].set_title(f"{name} (color)")
            self._axes[row, 1].set_title(f"{name} (depth)")
            for col in range(2):
                self._axes[row, col].set_xticks([])
                self._axes[row, col].set_yticks([])
        self._fig.tight_layout()

        self.DeclarePeriodicPublishEvent(period, 0.0, self._draw)

    def _draw(self, context):
        for row, name in enumerate(self._camera_names):
            rgb = self.GetInputPort(f"{name}.rgb_image").Eval(context).data
            depth = (
                self.GetInputPort(f"{name}.depth_image")
                .Eval(context)
                .data.squeeze(axis=2)
            )

            if self._color_artists[row] is None:
                self._color_artists[row] = self._axes[row, 0].imshow(rgb)
                self._depth_artists[row] = self._axes[row, 1].imshow(
                    depth, cmap="turbo"
                )
            else:
                self._color_artists[row].set_data(rgb)
                self._depth_artists[row].set_data(depth)

            # Rescale the depth colormap to the finite values in this frame.
            finite = depth[depth < float("inf")]
            if finite.size > 0:
                self._depth_artists[row].set_clim(
                    vmin=finite.min(), vmax=finite.max()
                )

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
