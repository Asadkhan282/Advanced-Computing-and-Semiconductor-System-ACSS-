# MemCore Experimental Results

**Project:** ACSS MemCore  
**Result snapshot:** 16 November 2026 (multi-workload robustness run)  
**Earlier deterministic trace experiments:** 9 November 2026

> **Scope and interpretation:** These are results from MemCore's deterministic synthetic software simulator. They are not measurements from physical HBM, a real GPU inference workload, an FPGA, or a fabricated memory controller. Hit rates describe the simulator's modeled HBM behavior under the stated synthetic traces.

## 1. Multi-workload robustness experiment

The robustness workload generator produced 12 workload files: six workload families, each with an informed-metadata and a default-metadata variant. The comparison harness evaluated six policies over 240 access requests per workload.

### Policies evaluated

- FIFO
- LRU
- SLO
- SLO_MARGIN
- ADAPTIVE_SLO
- ADAPTIVE_MARGIN

### Workload families

- Uniform
- Zipf
- Bursty
- Recurring
- Prediction mismatch
- Capacity pressure

### Confirmed leading results from the reported run

| Workload file | Leading policy/policies | HBM hit rate |
|---|---|---:|
| `bursty_default_metadata.json` | LRU | 77.50% |
| `recurring_default_metadata.json` | LRU | 62.08% |
| `uniform_informed.json` | LRU | 30.42% |
| `prediction_mismatch_default.json` | SLO_MARGIN | 54.58% |
| `prediction_mismatch_informed.json` | ADAPTIVE_MARGIN | 53.75% |
| `capacity_pressure_informed.json` | SLO, ADAPTIVE_SLO, ADAPTIVE_MARGIN (tie) | 17.92% |
| `capacity_pressure_default.json` | All six policies (tie) | 0.00% |

These are selected leaders, not a complete per-policy result matrix. The original console output / machine-readable result files should be added alongside this report when exported from the local experiment run.

### Initial observations

1. No single policy leads every workload family.
2. LRU led the reported bursty, recurring, and uniform-informed cases.
3. SLO_MARGIN led the reported default-metadata prediction-mismatch case.
4. ADAPTIVE_MARGIN led the reported informed-metadata prediction-mismatch case.
5. The informed capacity-pressure case produced a three-policy tie, while the default-metadata capacity-pressure case produced zero HBM hits for all six policies in this run.
6. The results motivate evaluating prediction confidence and fallback behavior rather than assuming adaptive predictions always improve placement or eviction decisions.

## 2. Earlier deterministic trace experiments

These smaller experiments compare FIFO, LRU, SLO, and SLO_MARGIN only. They are separate from the 12-file robustness experiment above and should not be combined into one aggregate score.

### 2.1 Host-pressure trace

**Configuration:** 10 events, 4 allocations, 6 accesses; modeled HBM capacity 1,024 bytes and host-memory capacity 512 bytes.

| Metric | FIFO | LRU | SLO | SLO_MARGIN |
|---|---:|---:|---:|---:|
| HBM hit rate | 33.33% | 33.33% | 50.00% | 50.00% |
| Cached hit rate | 83.33% | 83.33% | 83.33% | 83.33% |
| HBM hits | 2 | 2 | 3 | 3 |
| Host hits | 3 | 3 | 2 | 2 |
| Cold misses | 1 | 1 | 1 | 1 |
| HBM evictions | 5 | 5 | 4 | 4 |
| Host evictions | 1 | 1 | 1 | 1 |
| Bytes moved to host | 2,560 | 2,560 | 2,048 | 2,048 |
| Bytes restored | 1,536 | 1,536 | 1,024 | 1,024 |
| Bytes dropped | 512 | 512 | 512 | 512 |

In this specific trace, SLO and SLO_MARGIN achieved a higher HBM hit rate and fewer HBM evictions than FIFO and LRU. This is a trace-specific observation, not evidence of general superiority.

### 2.2 Prediction-mismatch trace

**Configuration:** 9 events, 4 allocations, 5 accesses; deterministic synthetic trace; simulated time 0.0 ms.

| Metric | FIFO | LRU | SLO | SLO_MARGIN |
|---|---:|---:|---:|---:|
| HBM hit rate | 80.00% | 40.00% | 40.00% | 40.00% |
| Cached hit rate | 80.00% | 100.00% | 80.00% | 80.00% |
| HBM hits | 4 | 2 | 2 | 2 |
| Host hits | 0 | 3 | 2 | 2 |
| Cold misses | 1 | 0 | 1 | 1 |
| HBM evictions | 2 | 5 | 4 | 4 |
| Host evictions | 1 | 1 | 1 | 1 |
| Bytes moved to host | 1,024 | 2,560 | 2,048 | 2,048 |
| Bytes restored | 0 | 1,536 | 1,024 | 1,024 |
| Bytes dropped | 512 | 512 | 512 | 512 |

This trace illustrates that metadata-driven policies can underperform a simpler baseline when predictions do not match subsequent accesses. FIFO achieved the highest HBM hit rate in this particular trace, while LRU achieved a 100% cached hit rate by counting both HBM and host hits. These metrics are distinct and should not be conflated.

## 3. Reproducibility

Run commands from the MemCore project root:

```bash
python -m unittest discover -s tests -v
python -m benchmarks.generate_robustness_workloads
python -m benchmarks.compare_workloads
```

The benchmark scripts and workload definitions are in the adjacent `benchmarks/` and `workloads/` directories. Exact output files depend on the local run and should be committed when generated.

## 4. Limitations and next validation steps

- The simulator uses deterministic synthetic workloads and modeled memory capacities.
- HBM hit rate is a simulator metric; it is not measured GPU bandwidth, latency, energy, or application throughput.
- No physical HBM device, real GPU inference trace, or FPGA prototype is represented by these results.
- The selected multi-workload leaders above are a summary of the reported console run, not the complete 12-workload × 6-policy matrix.
- Next validation steps: export full per-policy results to CSV/JSON; repeat experiments across multiple random seeds; report uncertainty and prediction-quality metrics; then validate memory-transfer behavior using real GPU profiling before making hardware-performance claims.

## 5. Reporting standard

When adding future results, include the exact command, code revision, workload file, seed, memory capacities, access count, policy configuration, and raw CSV/JSON output. Keep synthetic simulator results, GPU measurements, and FPGA measurements in separate sections.
