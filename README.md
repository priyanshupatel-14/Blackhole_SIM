# Schwarzschild Black Hole Simulation

A real-time Python visualization of a Schwarzschild black hole with an accretion disk, gravitational lensing, relativistic Doppler beaming, a lensed starfield, and a central black-hole shadow.

The simulation is rendered with **Pygame**, while **NumPy** is used for the particle calculations and coordinate transformations.

## Features

- Real-time 800×800 Pygame visualization
- 60 FPS target
- 60,000 accretion-disk particles
- Procedurally generated 2,500-star background
- Approximate gravitational lensing / Einstein-ring displacement
- Black-hole shadow and photon-sphere masking
- Inclined accretion disk viewed at approximately 82°
- Doppler-beaming color and brightness effects
- Temperature-based disk coloring from blue/white inner regions to orange/red outer regions
- Particle glow rendering through Pygame's pixel surface access

## Requirements

- Python 3.x
- Pygame
- NumPy
- A desktop environment capable of opening a Pygame window

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Simulation

Run:

```bash
python blackhole_sim.py
```

A Pygame window will open and display the simulation.

Close the window normally to stop the program.

## How It Works

### Starfield

The program creates a background field of approximately 2,500 stars. Their apparent positions are modified using an approximate point-mass gravitational-lensing calculation. Stars inside the configured shadow radius are hidden to represent the black-hole shadow. fileciteturn3file0L23-L56

### Accretion Disk

The accretion disk is represented by 60,000 particles distributed from 3 Schwarzschild radii to 15 Schwarzschild radii. The particle distribution is weighted toward the inner region, producing a denser appearance closer to the black hole. fileciteturn3file0L59-L70

The particle colors vary with radius to represent a hotter inner disk and cooler outer regions. fileciteturn3file0L73-L98

### Doppler Beaming

The simulation modifies particle brightness and color based on the projected velocity of the disk material. The approaching side is boosted toward brighter/bluer colors while the receding side becomes dimmer/redder. fileciteturn3file0L101-L116

### Gravitational Lensing

The disk is projected using an inclined 3D coordinate system and then displaced using an approximate Einstein-ring mapping. Particles behind the black hole receive the stronger lensing transformation. fileciteturn3file0L139-L177

### Rendering Order

The frame is rendered in layers:

1. Background starfield
2. Back portion of the lensed accretion disk
3. Black-hole shadow
4. Front portion of the accretion disk
5. FPS and simulation labels

This ordering creates the visual effect of the disk passing behind and in front of the black hole. fileciteturn3file0L189-L225

## Main Configuration

The current simulation starts with:

```python
width=800
height=800
fps=60
```

It uses:

```python
Schwarzschild radius = 40.0
Inclination = 82°
Particles = 60000
```

These values are defined in the `BlackHoleApp` initialization. fileciteturn3file0L6-L20

## Project Structure

```text
project/
├── blackhole_sim.py
├── requirements.txt
└── README.md
```

## Notes

This is a **visual simulation using approximations**, not a full general-relativistic ray-tracing engine. The code explicitly uses approximate point-mass lensing and simplified orbital/Doppler calculations. fileciteturn3file0L37-L39 fileciteturn3file0L70-L70

The simulation updates particle positions continuously and targets the configured FPS using Pygame's clock. fileciteturn3file0L210-L232
