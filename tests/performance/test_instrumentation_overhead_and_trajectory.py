from __future__ import annotations

import itertools
import json
import os
import platform
import statistics
import time
import warnings

import pytest
import torch

from observable_library.filter.identity import identity_observables
from observable_library.generator.introspector import introspect_parameters
from observable_library.runtime.runtime import Runtime
from observable_library.tensor.source import HookSource

STEPS = 20
OBSERVABLE_REDUCTIONS = ["sum", "mean", "l2_norm", "numel"]
MEASUREMENT_BATCHES = 3
REPETITIONS_PER_BATCH = 4
ABSOLUTE_ADVISORY_TARGET_SECONDS = 0.002


@pytest.mark.reproducibility
def test_instrumentation_does_not_change_loss_trajectory():
    baseline_losses, _elapsed = _run_training(instrumented=False)
    instrumented_losses, _elapsed = _run_training(instrumented=True)

    assert baseline_losses == instrumented_losses


@pytest.mark.performance
@pytest.mark.advisory
def test_framework_overhead_is_monitored_against_advisory_target():
    operations = {
        "baseline": lambda: _run_training(instrumented=False)[1],
        "instrumented": lambda: _run_training(instrumented=True)[1],
        "compute": _run_standalone_observable_compute,
    }
    operation_orders = list(itertools.permutations(operations))

    for operation_order in operation_orders:
        for operation_name in operation_order:
            operations[operation_name]()

    batches = []
    for _batch in range(MEASUREMENT_BATCHES):
        samples = {operation_name: [] for operation_name in operations}
        paired_trials = []
        load_before = _load_average()
        wall_started = time.perf_counter()
        process_started = time.process_time()
        for _repetition in range(REPETITIONS_PER_BATCH):
            for operation_order in operation_orders:
                trial = {}
                for operation_name in operation_order:
                    value = operations[operation_name]()
                    samples[operation_name].append(value)
                    trial[operation_name] = value
                paired_trials.append(trial)
        wall_time = time.perf_counter() - wall_started
        process_time = time.process_time() - process_started
        batches.append(
            {
                "samples": samples,
                "paired_trials": paired_trials,
                "wall_time": wall_time,
                "process_time": process_time,
                "load_before": load_before,
                "load_after": _load_average(),
            }
        )

    baseline_time = statistics.median(
        statistics.median(batch["samples"]["baseline"]) for batch in batches
    )
    instrumented_time = statistics.median(
        statistics.median(batch["samples"]["instrumented"]) for batch in batches
    )
    compute_time = statistics.median(
        statistics.median(batch["samples"]["compute"]) for batch in batches
    )

    extra_time = instrumented_time - baseline_time
    framework_time = max(0.0, extra_time - compute_time)
    advisory_limit = max(
        0.25 * compute_time,
        0.02 * baseline_time,
        ABSOLUTE_ADVISORY_TARGET_SECONDS,
    )

    within_advisory_target = framework_time <= advisory_limit
    diagnostic = _timing_diagnostic(
        batches=batches,
        baseline_time=baseline_time,
        instrumented_time=instrumented_time,
        compute_time=compute_time,
        extra_time=extra_time,
        framework_time=framework_time,
        advisory_limit=advisory_limit,
        within_advisory_target=within_advisory_target,
    )
    print(
        "INSTRUMENTATION_TIMING_DIAGNOSTIC="
        f"{json.dumps(diagnostic, separators=(',', ':'))}"
    )
    if not within_advisory_target:
        warnings.warn(
            "framework overhead exceeded its advisory target; "
            "see INSTRUMENTATION_TIMING_DIAGNOSTIC for timing and runner details",
            RuntimeWarning,
            stacklevel=1,
        )


def _timing_diagnostic(
    *,
    batches: list[dict[str, object]],
    baseline_time: float,
    instrumented_time: float,
    compute_time: float,
    extra_time: float,
    framework_time: float,
    advisory_limit: float,
    within_advisory_target: bool,
) -> dict[str, object]:
    return {
        "summary_seconds": {
            "T_baseline": baseline_time,
            "T_instrumented": instrumented_time,
            "T_compute": compute_time,
            "T_extra": extra_time,
            "T_framework": framework_time,
            "advisory_limit": advisory_limit,
            "absolute_advisory_target": ABSOLUTE_ADVISORY_TARGET_SECONDS,
        },
        "within_advisory_target": within_advisory_target,
        "batches": [_batch_diagnostic(batch) for batch in batches],
        "runner": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "affinity_cpu_count": _affinity_cpu_count(),
            "torch_intra_op_threads": torch.get_num_threads(),
            "torch_inter_op_threads": torch.get_num_interop_threads(),
            "runner_environment": {
                name: os.environ.get(name)
                for name in (
                    "CI",
                    "GITHUB_ACTIONS",
                    "RUNNER_OS",
                    "RUNNER_ARCH",
                    "RUNNER_NAME",
                )
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
    }


def _batch_diagnostic(batch: dict[str, object]) -> dict[str, object]:
    samples = batch["samples"]
    paired_trials = batch["paired_trials"]
    assert isinstance(samples, dict)
    assert isinstance(paired_trials, list)

    paired_extra = []
    paired_framework_signed = []
    for trial in paired_trials:
        assert isinstance(trial, dict)
        extra = trial["instrumented"] - trial["baseline"]
        paired_extra.append(extra)
        paired_framework_signed.append(extra - trial["compute"])

    wall_time = batch["wall_time"]
    process_time = batch["process_time"]
    assert isinstance(wall_time, float)
    assert isinstance(process_time, float)
    return {
        "components_seconds": {
            name: _sample_summary(values) for name, values in samples.items()
        },
        "paired_seconds": {
            "extra": _sample_summary(paired_extra),
            "framework_signed": _sample_summary(paired_framework_signed),
        },
        "wall_time": wall_time,
        "process_time": process_time,
        "process_wall_ratio": process_time / wall_time if wall_time else None,
        "load_before": batch["load_before"],
        "load_after": batch["load_after"],
    }


def _sample_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _affinity_cpu_count() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    return len(get_affinity(0)) if get_affinity is not None else None


def _load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def _run_training(instrumented: bool) -> tuple[list[float], float]:
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    model = _make_model()
    inputs, targets = _make_data()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    hook_source = HookSource(model) if instrumented else None
    metadata = introspect_parameters(model)
    observables = identity_observables(
        metadata,
        reductions=OBSERVABLE_REDUCTIONS,
    )

    if hook_source is not None:
        tensors = {
            item["source"]: parameter
            for item, parameter in zip(metadata, model.parameters())
        }
        runtime = Runtime(observables=observables, tensors=tensors)
        hook_source.attach()

    losses: list[float] = []
    started = time.perf_counter()
    try:
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output, targets)
            if hook_source is not None:
                hook_source.record_loss(loss, step)
            loss.backward()
            if hook_source is not None:
                hook_source.get("activation.0", step)
                hook_source.get("grad.0.weight", step)
                hook_source.get("loss", step)
                runtime.observe(step=step)
            optimizer.step()
            losses.append(float(loss.detach()))
        elapsed = time.perf_counter() - started
    finally:
        if hook_source is not None:
            hook_source.detach()
    return losses, elapsed


def _run_standalone_observable_compute() -> float:
    torch.manual_seed(7)
    model = _make_model()
    observables = identity_observables(
        introspect_parameters(model),
        reductions=OBSERVABLE_REDUCTIONS,
    )
    metadata = introspect_parameters(model)
    tensors = {
        item["source"]: parameter
        for item, parameter in zip(metadata, model.parameters())
    }
    runtime = Runtime(observables=observables, tensors=tensors)
    started = time.perf_counter()
    for step in range(STEPS):
        runtime.observe(step=step)
    return time.perf_counter() - started


def _make_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.linspace(-1.0, 1.0, steps=80).reshape(20, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    return inputs, targets
