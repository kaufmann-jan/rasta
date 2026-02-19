# rasta — Technical Specification

## Purpose
`rasta` computes **linear short-term and long-term ship response statistics** from complex RAOs and wave spectra.  
It also provides preprocessing utilities to derive additional responses (accelerations, point motions, relative motions) from 6DoF RAOs.

The package uses a **single canonical in-memory representation** to guarantee correctness and vectorized computation.

---

## Canonical Data Model

### Core container
All hydrodynamic response data is stored as:

> `xarray.Dataset` with variable `rao: complex128`

This is the only valid internal representation.

---

### Required coordinates
| name | type | unit | description |
|-----|----|----|----|
| `freq` | float | rad/s | strictly increasing |
| `dir` | float | deg | 180=head sea, 0=following |
| `resp` | string | — | response channel |

---

### Optional coordinates (SI only)
`speed (m/s)`, `depth (m)`, `draft`, `trim`, `lc`, etc.  
All optional axes must be 1-D labeled coordinates.

---

### Required dataset attributes

freq_unit = "rad/s"
dir_unit = "deg"
dir_convention = "180=head, 0=following"
coord_sys = "body-fixed RH"
axis_x = "forward"
axis_y = "port"
axis_z = "up"
rotation_convention = "RH about +x/+y/+z for roll/pitch/yaw"
rao_definition = "complex response per unit wave amplitude"
angle_unit = "rad"


Direction normalized to `[0,360)`.

---

## Response Naming

### Motions
`surge, sway, heave, roll, pitch, yaw`  
(rotations stored in radians)

### Accelerations
`<motion>_acc`

### Loads
`Fx, Fy, Fz, Mx, My, Mz`

Names must never encode units, speed, direction, or points.

---

## Wrapper Class
A thin wrapper `RAOSet` enforces schema validity.

Responsibilities:
- validate dataset at construction
- expose:
  - `.rao`
  - `.amp = abs(rao)`
  - `.phase_deg = angle(rao)`
- selection helpers (`sel_dir`, `sel_speed`, …)
- conversion helpers (amp/phase import)

All package functions accept and return `RAOSet`.

---

## Schema Validation Rules
Validation occurs once at module boundaries (I/O or preprocessing output).

Reject dataset if:
- missing `rao`
- non-complex dtype
- missing dims `freq, dir, resp`
- coordinates not 1-D
- freq not float or not monotonic
- dir not float
- required attrs missing or wrong

Normalization:
- sort `freq`
- wrap `dir` to `[0,360)`

After validation, internal code assumes correctness.

---

## Coordinate & Motion Conventions

Right-handed body-fixed frame:

| axis | direction |
|----|----|
| x | forward |
| y | port |
| z | up |

Rotations:
- roll about +x
- pitch about +y
- yaw about +z

---

## Frequency-Domain Kinematics
For ω = `freq` (rad/s)

Displacement RAO: `X(ω)`

Velocity:

i ω X


Acceleration:

-ω² X


---

## Point Motion Linearization
For point position `r = (x,y,z)` relative CG:

Displacement:

u_point = u + θ × r


Acceleration:

a_point = a + α × r


(centrifugal term ignored — linear theory)

---

## Architecture

Modules:

rasta.io → readers produce canonical Dataset
rasta.preprocessing → derived RAOs
rasta.spectra → wave spectra
rasta.stats.shortterm
rasta.stats.longterm
rasta.validation → schema enforcement
rasta.rao → RAOSet wrapper


Rules:
- pandas only for import/export
- internal math uses xarray/NumPy
- never rely on axis order
- always rely on labeled dimensions

---

## Core Principle
> Validate once at boundaries.  
> Assume correctness everywhere else.

All algorithms must operate purely vectorized over any additional dimensions (speed, depth, loadcase, scatter bins).

---

This file acts as the **authoritative contract** for contributors and automated code generation.
