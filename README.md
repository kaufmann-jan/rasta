# rasta

`rasta` computes linear short-term and long-term ship response statistics from complex RAOs and wave spectra.

The internal canonical representation is:

- `xarray.Dataset`
- data variable `rao` with complex dtype
- required dimensions/coordinates: `freq`, `dir`, `resp`

See `SPECS.md` for the full contract.
