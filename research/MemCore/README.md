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
- PREQUENTIAL_SLO_MARGIN (experimental)

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

## MemCore v0.1 Experimental

MemCore includes an opt-in prequential evidence-gated
SLO-margin policy for synthetic memory-management research.

### Prequential prediction monitoring

The simulator records a historical reuse baseline when a
prediction is registered, before its future reuse outcome
is observed.

Completed predictions are evaluated using Brier scores.

The experimental policy uses:

- Failure window: 20 completed observations.
- Failure threshold: excess Brier > +0.02.
- Recovery window: 12 completed observations.
- Recovery threshold: excess Brier <= -0.02.
- Fallback: observed global reuse prevalence.
- Eviction: fixed SLO-margin victim selection.

The default memory-management policy remains unchanged.

### Research validation

Validation completed November 20, 2026:

- 78 Python unit tests passed.
- 1,920 held-out policy executions completed.
- 240/240 aggregate and phase-level equivalence checks passed.
- 240/240 event-level research/implementation traces matched.

The event-level comparison covered tracked HBM and host
ordering, eviction decisions, memory counters, and prediction
histories on the tested workloads.

### Held-out synthetic results

The following are HBM hit rates, not real GPU benchmarks.

| Informed metadata scenario | RAW_FIXED | PREQ_20_FAST12 |
| --- | ---: | ---: |
| Clean | 45.51% | 47.81% |
| Late inversion | 42.25% | 44.98% |
| Temporary inversion | 43.77% | 45.27% |

For clean workloads with default metadata, both policies
achieved 46.13%.

Some recovery-phase workloads still regress against the
raw baseline. The policy remains experimental.

### Reproduce validation

Run commands from research/MemCore:

```powershell
python -m unittest discover -s tests -p "test_*.py" -q
python -m benchmarks.validate_prequential_guard
python -m benchmarks.validate_production_prequential_equivalence
python -m benchmarks.verify_prequential_event_equivalence
```

Benchmark scripts use deterministic synthetic workloads.
Their seed ranges and experimental conditions are defined
within each benchmark module.

### Hardware status

The RTL and Vivado directories contain early hardware
research material. No FPGA implementation, ASIC fabrication,
or hardware performance validation is claimed.

Real LLM inference traces, GPU runtime integration,
latency measurements, and energy measurements remain
future research milestones.

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
