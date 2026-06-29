##
#
# Basic loading and reading utilities for 3Dconnexion SpaceMouse devices.
#
##

import time

import easyhid
import pyspacemouse


def list_spacemouse_paths() -> list[str]:
    """Return the unique /dev/hidraw* paths of connected SpaceMice."""
    supported = {
        (spec.vendor_id, spec.product_id)
        for spec in pyspacemouse.get_device_specs().values()
    }
    paths = []
    for dev in easyhid.Enumeration().find():
        if (dev.vendor_id, dev.product_id) not in supported:
            continue
        path = dev.path.decode() if isinstance(dev.path, bytes) else dev.path
        if path not in paths:
            paths.append(path)
    return paths


def test_spacemice():
    """Open every connected SpaceMouse and continuously print its readings.

    Useful for verifying the pyspacemouse install/permissions and checking
    which physical device is enumerated first (device 0 -> left arm).
    """
    paths = list_spacemouse_paths()
    print(f"Found {len(paths)} physical SpaceMouse device(s): {paths}")
    if not paths:
        print("No spacemice found!")

    input("Press [ENTER] to continue with the raw readings test...")

    devices = [pyspacemouse.open_by_path(p) for p in paths]
    print("Move the devices; press Ctrl-C to stop.\n")
    last_print = 0.0
    try:
        while True:
            # Drain pending HID reports every loop so we always hold the latest
            # state. read() consumes a single report, and the device emits them
            # at ~100+ Hz while moving, so this tight loop (vs. one read per
            # print) is what keeps the readings from lagging behind.
            states = [dev.read() for dev in devices]

            now = time.monotonic()
            if now - last_print >= 0.1:  # print at ~10 Hz
                last_print = now
                for i, s in enumerate(states):
                    print(
                        f"dev{i} [{paths[i]}]  "
                        f"x={s.x:+.2f} y={s.y:+.2f} z={s.z:+.2f}  "
                        f"r={s.roll:+.2f} p={s.pitch:+.2f} yaw={s.yaw:+.2f}  "
                        f"buttons={s.buttons}"
                    )
                print("---")
            time.sleep(0.001)
    except KeyboardInterrupt:
        for dev in devices:
            dev.close()


if __name__ == "__main__":
    test_spacemice()
