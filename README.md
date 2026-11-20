# rasta

`rasta` computes linear short-term and long-term ship response statistics from complex RAOs and wave spectra.

The internal canonical representation is:

- `xarray.Dataset`
- data variable `rao` with complex dtype
- required dimensions/coordinates: `freq`, `dir`, `resp`

See `SPECS.md` for the full contract.

## Reconstructing a CFD wave realization

`reconstruct_response` combines the original component amplitudes and random
phases with complex RAOs to reconstruct first-order motion, acceleration, or
load time histories. RAOs are interpolated in real and imaginary parts at the
intrinsic component frequencies and at the requested heading. At forward speed,
the time dependence uses `omega_e = omega - k*speed*cos(direction)`, with
0 degrees following and 180 degrees head seas. Stored component wave numbers
are validated at the stated water depth and used directly.

For example, for HydroStar RAOs calculated at 30 m water depth:

```python
import numpy as np
import matplotlib.pyplot as plt

from rasta import generate_wave_components, reconstruct_response
from rasta.io import read_hydrostar_raos

rs = read_hydrostar_raos(["heave.rao", "pitch.rao"])
components = generate_wave_components(
    hs=1.0,
    tp=6.0,
    spectrum_model="jonswap",
    n_components=50,
    discretization="equal_energy",
    depth=30.0,
    seed=42,
)
time = np.arange(0.0, 660.0, 0.1)

response = reconstruct_response(
    rs,
    components,
    time,
    resp=["heave", "pitch"],
    direction=180.0,
    frequency_domain="encounter",
)
z_linear = response.sel(resp="heave")
plt.plot(z_linear.time, z_linear, label="HydroStar linear reconstruction")
# With your CFD results loaded as rbd:
# plt.plot(rbd["time"], rbd["z"], label="OpenFOAM CFD")
plt.xlabel("Time [s]")
plt.ylabel("Heave [m]")
plt.legend()
plt.grid(True)
plt.show()
```

Use exactly these wave components in the CFD wave generator. Select scalar
speed and depth coordinates if the RAOs contain several operating conditions.
The component and RAO water depths must agree. The default frequency
extrapolation raises if a component lies outside the RAO table. Supply a wider
RAO table, restrict the generated frequency range, or explicitly choose
`extrapolation="zero"` to omit those components' response contributions.
Heading interpolation stays within the available range; no heading symmetry
or interpolation across the heading-grid boundary is assumed.

Time-domain agreement also requires verifying all of the following:

- The RAO harmonic convention. For `phase_convention="negative_time"`, the
  response is `Re(sum(a*H*exp(1j*(phase-omega_e*time))))`. The `positive_time`
  convention uses a plus sign in the exponent. Supply
  `rao_phase_convention="negative_time"` or `"positive_time"` after checking
  the solver convention. If it differs from the requested time convention,
  the function conjugates the RAOs without changing the wave phases. For
  example, a verified positive-time RAO used in a negative-time reconstruction
  requires `rao_phase_convention="positive_time"`. The same source convention
  can be recorded in `rs.dataset.attrs["rao_phase_convention"]`. If unspecified,
  a warning is issued and the RAOs are used as supplied. HydroStar's harmonic
  convention is not inferred from its source name.
- The spatial phase reference. Generated phases refer to the wave origin
  `(0, 0)`, while HydroStar can specify another incident-wave reference point.
  Unknown or differing references produce a warning. Establish their alignment
  in your simulation setup before using `phase_reference_confirmed=True`;
  that argument acknowledges alignment and does not apply a spatial correction.
  Component attributes `x_wave_ref` and `y_wave_ref` can describe a phase origin
  established externally. They do not translate the generated phases.
- The CFD time origin and wave ramp-up. Time values are used exactly as supplied;
  unknown CFD offsets and ramp-up cannot be recovered from the RAOs.

The output has `time` and `resp` dimensions plus any sectional dimensions (such
as `x`). A single response retains its length-one `resp` dimension. Reliable
response units can be supplied through a `response_unit(resp)` coordinate or a
`response_units` mapping in RAO dataset attributes. A single-channel HydroStar
unit is also supported, accounting for the reader's conversion of rotations to
radians. A global units attribute is omitted for mixed or unknown units.

The reconstructed signal represents the oscillatory first-order response,
including any finite-record mean from the supplied components. It includes no
nonlinear mean offset, filtering, or normalization. Signed or nonmonotonic
encounter frequencies can be summed directly without the spectral-branch
restrictions required by encounter-frequency statistics.

For distinct, nonzero encounter frequencies, its theoretical discrete variance
is `0.5 * sum(amplitude**2 * abs(H)**2)`. A finite record (especially 600 seconds)
can differ because of beating and interference. The continuous variance
`integral(abs(H)**2 * S_wave, omega)` can also differ slightly because of wave
component discretization.
