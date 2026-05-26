# Driven dynamics with LOREM via i-PI

This example shows how to run an applied electric field simulation using a LOREM BEC model and i-PI's driven dynamics (Electric Dipole Approximation).

Unlike the standard i-PI efield example which reads fixed Born Effective Charges from a file, LOREM computes BECs on-the-fly from its learned model (`<bec mode="driver"/>`).

## Prerequisites

- Trained `LoremBEC` checkpoint (with `model/model.yaml`, `model/baseline.yaml`, `model/model.msgpack`)
- i-PI installed (`pip install ipi`)
- LOREM i-PI driver installed: `lorem-install-ipi-driver`

## Files

- `input.xml` — i-PI configuration for driven dynamics (EDA-NVE). Adapt the E-field parameters (`amp`, `freq`, `peak`, `sigma`), cell size, and `start.xyz` to your system.
- `start.xyz` — Starting structure in atomic units. Replace with your system's geometry.
- `run.sh` — Launch script. Set `MODEL_PATH` to your trained checkpoint folder.

## Running

```bash
MODEL_PATH="/path/to/checkpoint"

# Install LOREM driver into i-PI (idempotent)
lorem-install-ipi-driver

# Start i-PI server
i-pi input.xml > i-pi.out &
sleep 5

# Start LOREM driver
i-pi-driver-py -a lorem -u -m lorem \
    -o model_path=${MODEL_PATH},template=start.xyz \
    > driver.out &

wait
```

See `run.sh` for a self-contained reference script.

i-PI outputs will be written to `i-pi.*` files. Key outputs:
- `i-pi.properties.out` — energy, kinetic energy, E-field strength over time
- `i-pi.positions_*` — nuclear trajectories
- `i-pi.bec{x,y,z}_*` — BEC tensor components over time

## Adapting to your system

1. Replace `start.xyz` with your starting geometry (atomic units).
2. Set the cell size in `input.xml` (large for isolated systems, physical for periodic).
3. Set `pbc='True'` in `<ffsocket>` if your system is periodic.
4. Adjust E-field parameters to match your target excitation.
5. Point `MODEL_PATH` in `run.sh` to your trained LoremBEC checkpoint.
