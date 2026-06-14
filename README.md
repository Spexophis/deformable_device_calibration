# deformable_device_calibration

A wavefront sensing (WFS) system for calibrating and controlling deformable mirrors (DMs).

---

## Overview

This package implements a complete wavefront sensing and correction pipeline:

1. **Senses wavefront aberrations** via two independent methods:
   - **Shack-Hartmann WFS** — measures local slopes across a lenslet array
   - **Interferometry WFS** — extracts wrapped phase from interference patterns via FFT-based carrier tracking
2. **Reconstructs and decomposes** the wavefront into up to Zernike polynomial modes
3. **Controls a deformable mirror** to correct aberrations in closed-loop
4. **Runs automated calibration routines** — influence function measurement, sensorless modal optimization, and integrator-based closed-loop flattening

---
### Main Operations

| Operation | Description |
|-----------|-------------|
| WFS Imaging | Live camera feed with slope/phase overlay |
| Wavefront Reconstruction | Continuous Shack-Hartmann or interferometry reconstruction |
| Influence Function | Poke each actuator, collect images, compute the influence matrix |
| WFS Iteration (sensorless) | Modal sweep over Zernike modes with metric-based minimization |
| Closed-loop Flattening | Integrator-based correction: `v = v_old + gain × Δphase` |
| DM Flatten | Multi-frame interferometry averaging with piston-drift correction |

---

## Key Algorithms

### Shack-Hartmann WFS (`computations/shwfs_reconstruction.py`)
- 28×28 lenslet array, 44 px pitch, 32 px half-subimage
- Spot localization: FFT cross-correlation or iterative weighted center-of-gravity (IWCOG)
- Hudgins 2D integrator for wavefront reconstruction from gradients

reference: https://github.com/Knerlab/SIM_Control_Software

### Interferometry WFS (`computations/interferometry_reconstruction.py`, `actuator_response.py`)
- FFT-based carrier order detection (Gaussian fit with power-centroid fallback)
- Coherent multi-frame averaging with piston-drift correction
- Phase unwrapping via `skimage.restoration.unwrap_phase`
- Least-squares tilt removal

reference: https://aomicroscopy.org/interferometric-calibration-of-a-deformable-mirror

---

## Configuration

The application requires a JSON config file with the following structure:

```json
{
  "Adaptive Optics": {
    "Deformable Mirror": {
      "ALPAO": {
        "Serial": "0108xxxx",
        "Model": "97",
        "Phase Control Matrix": "path/to/phase_ctrl.tif",
        "Zonal Control Matrix": "path/to/zonal_ctrl.tif",
        "Modal Control Matrix": "path/to/modal_ctrl.tif",
        "Influence Function Images": "path/to/influence.tif",
        "Mirror Flat": "path/to/flat.tif",
        "Initial Flat": "path/to/initial_flat.tif",
        "Calibration File Folder": "path/to/calibration/"
      }
    },
    "Light Sources": {
      "Lasers": {
        "Cobolt": {
          "LaserName": { "Serial": "xxxx" }
        }
      }
    }
  }
}
```

---

## Project Structure

```
deformable_device_calibration/
├── main.py                            # Entry point
├── pyproject.toml
├── LICENSE
└── src/deformable_device_calibration/
    ├── main.py                        # AppWrapper (PyQt6 main loop)
    ├── executor.py                    # Command routing & signal handling
    ├── run_threads.py                 # Background threads (WFS, camera, tasks)
    ├── logger.py                      # Logging configuration
    │
    ├── devices/                       # Hardware abstraction layer
    │   ├── device.py                  # DeviceManager (coordinator)
    │   ├── alpao_dm.py                # ALPAO deformable mirror driver
    │   ├── flir_cmos.py               # FLIR camera driver (PySpin)
    │   ├── cobolt_laser.py            # Cobolt laser driver
    │   └── mock_cam.py                # Mock camera for offline use
    │
    ├── computations/                  # Wavefront reconstruction
    │   ├── computator.py             # ComputationManager
    │   ├── shwfs_reconstruction.py    # Shack-Hartmann WFS
    │   ├── interferometry_reconstruction.py  # Interferometry WFS
    │   ├── actuator_response.py       # Carrier tracking & phase extraction
    │   └── dynamic_controller.py     # Closed-loop controller
    │
    ├── gui/                           # PyQt6 interface
    │   ├── main_window.py
    │   ├── controller_panel.py
    │   ├── ao_panel.py
    │   ├── viewer_window.py
    │   ├── gl_viewer.py               # Optional OpenGL visualization
    │   └── custom_widgets.py
    │
    └── utilities/                     # Core algorithms
        ├── zernike_generator.py       # Zernike polynomials (Noll indexing)
        └── image_processor.py         # Centroid detection, SVD, image metrics
```
---

## License

MIT License — see [LICENSE](LICENSE) for details.
