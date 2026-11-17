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

## Experimental results

A results report is available at **[results/benchmark_results.md](results/benchmark_results.md)**. It documents the reported multi-workload robustness leaders and earlier deterministic trace experiments, including experiment conditions and limitations.

**Important:** The published results are deterministic synthetic simulator outputs. They are not physical HBM measurements or real-GPU inference benchmarks. The multi-workload report contains selected leading results, not the full per-policy result matrix; raw CSV/JSON exports should be added when available.

## Repository structure

- benchmarks/: Research experiments
- memcore/: Simulator and memory manager
- tests/: Python tests
- workloads/: Synthetic workload definitions
- rtl/: Early RTL workspace
- vivado/: Early Vivado workspace
- results/: Research benchmark reports

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Reproduce robustness experiments

```bash
python -m benchmarks.generate_robustness_workloads
python -m benchmarks.compare_workloads
```

## Limitations

- Deterministic synthetic software simulator.
- Not a physical HBM model.
- Not a measured real-GPU inference benchmark.
- No FPGA implementation or hardware validation is claimed.
- RTL and Vivado directories are early workspaces.

## Development direction

Improve prediction-aware memory management under inaccurate predictions and changing workload behavior, then validate against real GPU traces before making hardware-performance claims.

## License

No open-source license has been added yet.
