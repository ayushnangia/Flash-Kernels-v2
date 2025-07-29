# Omni-Perf-Bench Specification

## 1. Motivation

Writing high-performance GPU kernels is still expert-intensive; even state-of-the-art LLMs match PyTorch performance on <20% of KernelBench tasks. Hardware diversity (different GPU architectures/vendors) further complicates manual optimization. Existing Triton operator generation also shows substantial headroom: professional human Triton code often attains low GPU efficiency, underscoring the difficulty.

**Goal**: Provide a performance-engineer-oriented benchmark that:
- (a) Stresses end-to-end kernel quality across languages/DSLs
- (b) Reports microarchitectural efficiency, cost and portability
- (c) Adheres to reproducibility principles similar to MLPerf

## 2. Limitations of Existing Benchmarks

### KernelBench
- Supplies 250 modern PyTorch workloads with evaluation metric combining correctness and speedup threshold
- Does not expose rich microarchitectural statistics (occupancy, memory behavior, roofline placement)
- Focuses on single GPU context despite acknowledging hardware proliferation

### TritonBench
- Introduces TritonBench-G (184 curated human operators) and TritonBench-T (PyTorch-aligned fused operators)
- Language-restricted to Triton DSL only
- Community requests support for other DSLs like ThunderKittens and TileLang

### General Benchmarking Guidelines
- Industry benchmarks (e.g., MLPerf) emphasize fairness, reproducibility, and standardized reporting
- Current kernel-generation benchmarks do not yet fully adopt these structured submission/verification practices

## 3. Design Principles

1. **Performance-Engineer Fidelity**: Expose micro metrics (warp occupancy, memory throughput, arithmetic intensity, roofline position) via Nsight Compute/similar profilers

2. **Cross-Language/Cross-DSL**: Include CUDA C++, Triton, ThunderKittens, TileLang and others

3. **Hardware Portability**: Run identical tasks across multiple GPU vendors/architectures to quantify portability gaps

4. **Cost & Energy Awareness**: Report time-to-solution and $/speedup using cloud pricing methodology

5. **MLPerf-Style Reproducibility**: Standard harness, environment capture, and run rules (fixed driver versions, pinned clocks, power management settings)

## 4. Workload Suite

### Coverage
- Merge operator-level, fusion patterns, and end-to-end micro-models
- Build on KernelBench's breadth and TritonBench's real-world operators/fusions
- Support dynamic shapes and mixed precision

### Selection Process
1. Start from open workloads (KernelBench/TritonBench licenses)
2. Add tasks targeting optimization pain points (attention variants, normalization, reduction-heavy ops)
3. Annotate each workload with metadata: tensor shapes, memory footprint, theoretical arithmetic intensity

## 5. Evaluation Metrics

| Category | Metric | Rationale |
|----------|--------|-----------|
| **Correctness** | Call & Execution Accuracy | Proven effectiveness in TritonBench |
| **Functional Speed** | Speedup vs. Baseline | Extends KernelBench metric |
| **Microarchitecture** | Occupancy, SM efficiency, memory throughput, warp stall reasons | Nsight Compute diagnosis |
| **Roofline Position** | FLOP/s vs Arithmetic Intensity | Guides bottleneck classification |
| **GPU Efficiency** | Per-device efficiency metric | Generalized beyond Triton |
| **Portability Score** | Performance variance across devices | Responds to hardware diversity |
| **Cost/TCO** | $ per speedup/inference hour | Mirrors industry analyses |
| **Reproducibility** | Determinism checks (variance <ε) | Aligns with MLPerf fairness goals |

## 6. Measurement Methodology

- **Harness**: Docker images pin OS, CUDA/ROCm drivers, compiler versions
- **Profiling**: Automated Nsight Compute CLI collection with fixed metric set
- **Noise Reduction**: Disable DVFS/set persistence mode; exclusive GPU; fixed power mode
- **Validation**: Independent replay script verifies kernel hashes and reproduces metrics

## 7. Scoring & Leaderboard

### Engineering Score (Weighted Composite)
- 40% functional speedup
- 30% micro-efficiency normalized to roofline ceilings
- 20% portability
- 10% reproducibility

### Economic Score
- Normalizes speedup by cost ($/hour)
- Surfaces financially efficient generation strategies

Both include confidence intervals; submissions must attach profiling artifacts for audit.

## 8. Baselines

Include:
- Hand-written expert kernels (from TritonBench-G/GitHub)
- PyTorch eager/torch.compile
- Vendor libraries (cuBLAS/cuDNN)
- Prominent kernel-generation LLMs

## 9. Reproducibility & Release

Following MLPerf norms:
- Open-source code
- Dataset scripts
- Environment manifest
- Continuous integration tests
- Community feedback channels for DSL additions

## 10. Implementation Status

This document describes the target state of Omni-Perf-Bench. The implementation is being done incrementally on top of the existing Flash-Kernels-v2 infrastructure.

### Current Phase: Phase 1 - Enhanced Metrics Collection
- Adding microarchitectural metrics collection
- Integrating Nsight Compute profiling
- Implementing roofline analysis

### Upcoming Phases:
- Phase 2: Multi-DSL Support
- Phase 3: Reproducibility & Environment Control
- Phase 4: Enhanced Scoring System
- Phase 5: Advanced Visualization
- Phase 6: CI/CD and Automation

See the implementation plan in the repository for detailed timelines and milestones.