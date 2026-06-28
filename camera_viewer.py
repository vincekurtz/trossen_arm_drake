import numpy as np
import pyqtgraph as pg
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

    The images are drawn at a fixed refresh rate, one row of (color, depth)
    image panels per camera, in a single resizable window.
    """

    def __init__(self, camera_names: list[str], period: float = 0.1):
        LeafSystem.__init__(self)

        self._camera_names = list(camera_names)

        # Display image data as (row, col) so HxW(xC) np arrays appear upright.
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Declare an rgb + depth input port for each camera.
        for name in self._camera_names:
            self.DeclareAbstractInputPort(
                f"{name}.rgb_image", AbstractValue.Make(ImageRgba8U())
            )
            self.DeclareAbstractInputPort(
                f"{name}.depth_image", AbstractValue.Make(ImageDepth32F())
            )

        # Reuse an existing QApplication if one is already running (e.g. when
        # several viewers are constructed), otherwise create one.
        self._app = pg.mkQApp("Camera Viewer")

        # One row per camera, two columns (color, depth). Each cell is a
        # ViewBox with a locked aspect ratio and an ImageItem we update in
        # place. invertY puts row 0 at the top, matching image convention.
        self._win = pg.GraphicsLayoutWidget()
        self._win.setWindowTitle("Camera Viewer")
        self._color_items = []
        self._depth_items = []
        depth_cmap = pg.colormap.get("turbo")
        for row, name in enumerate(self._camera_names):
            # Each camera occupies two grid rows: a title row and an image row.
            title_row, image_row = 2 * row, 2 * row + 1
            for col, label in enumerate(("color", "depth")):
                self._win.addLabel(f"{name} ({label})", row=title_row, col=col)
                vb = self._win.addViewBox(
                    row=image_row, col=col, lockAspect=True
                )
                vb.invertY(True)
                vb.setMouseEnabled(False, False)
                item = pg.ImageItem(axisOrder="row-major")
                vb.addItem(item)
                if col == 0:
                    self._color_items.append(item)
                else:
                    item.setColorMap(depth_cmap)
                    self._depth_items.append(item)

        self._win.show()
        self._win.resize(800, 400 * max(len(self._camera_names), 1))

        self.DeclarePeriodicPublishEvent(period, 0.0, self._draw)

    def _draw(self, context):
        for row, name in enumerate(self._camera_names):
            rgb = self.GetInputPort(f"{name}.rgb_image").Eval(context).data
            depth = (
                self.GetInputPort(f"{name}.depth_image")
                .Eval(context)
                .data.squeeze(axis=2)
            )

            # RGBA uint8 is displayed directly; no levels/LUT needed.
            self._color_items[row].setImage(rgb, autoLevels=False)

            # Rescale the depth colormap to the finite values in this frame.
            finite = depth[np.isfinite(depth)]
            if finite.size > 0:
                levels = (float(finite.min()), float(finite.max()))
            else:
                levels = (0.0, 1.0)
            self._depth_items[row].setImage(depth, levels=levels)

        # Pump the Qt event loop so the window repaints and stays responsive.
        self._app.processEvents()
