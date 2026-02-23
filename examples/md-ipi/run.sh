#!/bin/bash
# Driven dynamics with LOREM BEC model via i-PI.
#
# Prerequisites:
#   - i-PI installed (`pip install ipi`)
#   - LOREM driver installed (`lorem-install-ipi-driver`)
#   - Trained LoremBEC checkpoint
#
# Adapt MODEL_PATH and start.xyz / input.xml to your system.

MODEL_PATH="/path/to/checkpoint"

# Install LOREM driver into i-PI (idempotent)
lorem-install-ipi-driver

# Start i-PI server
i-pi input.xml > i-pi.out &
echo "i-PI started"

# Wait for socket to be ready
sleep 5

# Start LOREM driver
i-pi-driver -a lorem -u -m lorem \
    -o model_path=${MODEL_PATH},template=start.xyz \
    > driver.out &
echo "LOREM driver started"

wait
echo "Simulation complete"
