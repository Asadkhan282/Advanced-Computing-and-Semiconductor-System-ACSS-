# MemCore

**Experimental memory-management and scheduling simulator for AI inference workloads.**

MemCore explores memory placement and eviction policies between modeled HBM and host memory.

## Research objectives
- Memory allocation, access, and eviction.
- Deadline-aware and prediction-aware scheduling.
- Adaptive reuse prediction and prediction uncertainty.
- Evaluation across synthetic workload families.

## Policies
- FIFO
- LRU
- SLO
- SLO_MARGIN
- ADAPTIVE_SLO
- ADAPTIVE_MARGIN

## Workload coverage
- Uniform
- Zipf
- Bursty
- Recurring
- Prediction mismatch
- Capacity pressure

## Repository structure
- benchmarks/: Research experiments
- memcore/: Simulator and memory manager
- tests/: Python tests
- workloads/: Synthetic workload definitions
- rtl/: Early RTL workspace
- vivado/: Early Vivado workspace

## Run tests
python -m unittest discover -s tests -v

## Limitations
- Deterministic synthetic software simulator.
- Not a physical HBM model.
- Not a measured real-GPU inference benchmark.
- No FPGA implementation or hardware validation is claimed.
- RTL and Vivado directories are early workspaces.

## Development direction
Improve prediction-aware memory management under inaccurate predictions and changing workload behavior.

## License
No open-source license has been added yet.
