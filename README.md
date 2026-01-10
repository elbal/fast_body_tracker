# Fast body tracker
This library provides an efficient Python implementation of the
[Azure Kinect Sensor SDK (K4A)](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)
and of the
[Azure Kinect Body Tracking SDK (K4ABT)](https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/index.html).
It intends to:
- Run as fast as possible.
- Support **multiple cameras**.
- Support **multiple threads**.
- Support **multiple bodies**.

For every frame it can, for example:
- Collect BGRA images from multiple depth cameras.
- Compute the joint positions.
- Plot those joint positions as overlays on the BGRA images.
- Save the joints positions in a HDF5 file.
- Save the BGRA images as video frames.
Without lagging or losing data.

It is fully compatible with the
[Orbbec SDK K4A Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/tree/v2-main),
which I personally use for my multi-camera Orbbec Femto Bolt setup.

This library is a fork of
[pyKinectAzure](https://github.com/ibaiGorordo/pyKinectAzure). The original
library is very robust but does not support multiple cameras or multiple
threads; this limits its performance in applications like real-time ergonomic
evaluation.

## Prerequisites
### Azure Kinect
- [Azure Kinect Sensor SDK (K4A)](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)
  should be installed to acquire data from the Azure Kinect.
- [Azure Kinect Body Tracking SDK (K4ABT)](https://www.microsoft.com/en-us/download/details.aspx?id=104221)
  should be installed to track the joints positions.

### Orbbec Femto Bolt/Mega prerequisites
- [Azure Kinect Sensor SDK (K4A)](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)
  should be installed to allow Python to use the original Azure Kinect
  library.
- [Azure Kinect Body Tracking SDK (K4ABT)](https://www.microsoft.com/en-us/download/details.aspx?id=104221)
  should be installed to track the joints positions.
- [Open Source Orbbec SDK](https://github.com/orbbec/OrbbecSDK_v2)
  should be installed to control the Orbbec depth cameras.
- [Orbbec SDK K4A Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/tree/v2-main)
  should be installed as a compatibility layer between the Orbbec SDK and the
  Azure Kinect library.

> note: I suggest updating the
> [Femto Bolt/Mega firmware](https://github.com/orbbec/OrbbecFirmware) to the
> latest version.

## Installation
```commandline
pip install fast_body_tracker
```

### Tested setup
The library was tested on Windows 11 with multiple Orbbec Femto Bolt cameras
used simultaneously. The cameras firmware was updated to v1.1.2, the Open
Source Orbbec SDK to v2.5.5, the Orbbec SDK K4A Wrapper to v2.0.11, the Azure
Kinect Sensor SDK to 1.4.2, and the Azure Kinect Body Tracking SDK to v1.1.2.

The setup should also work on Linux machines, while it does not work on Mac due
to the unavailability of the Azure Kinect Body Tracking SDK for that OS.

> note: when you are working on Linux, please ensure that the user has
> permission to the USB devices, or always execute under the root permission by
> adding `sudo` ahead. Additional information on the use on Linux can be found
> [here](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md).

## Contribution
Feel free to send pull requests, bug reports are also appreciated.