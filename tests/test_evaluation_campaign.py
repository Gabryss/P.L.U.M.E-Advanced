import json
import sys
from pathlib import Path

import numpy as np
import psutil
import pytest

from plume_advanced.evaluation.experiments.host_ablation import _summary
from plume_advanced.evaluation.experiments.scalability import monitor_worker
from plume_advanced.evaluation.morphology_bootstrap import PreparedDistance, morphology_intervals
from plume_advanced.evaluation.runner import ResultStore, run_case, run_cases
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.evaluation.statistics import normalized_wasserstein_distance


def test_cached_weighted_distance_matches_explicit_cluster_replication():
    rng = np.random.default_rng(21)
    reference = np.array([0., 0., 1., 3., 4., 7., 9.])
    generated = np.array([0., 2., 2., 5., 8.])
    prepared = PreparedDistance(reference, generated)
    for _ in range(100):
        ref_counts = rng.integers(0, 4, len(reference))
        gen_counts = rng.integers(0, 4, len(generated))
        if not ref_counts.any() or not gen_counts.any():
            continue
        expected = normalized_wasserstein_distance(np.repeat(reference, ref_counts), np.repeat(generated, gen_counts))
        assert prepared.evaluate(ref_counts, gen_counts) == pytest.approx(expected, rel=1e-12)


def test_joint_bootstrap_uses_same_cluster_draws_for_aggregate():
    reference = [{"reference_cave_id": str(i//2), "x": x, "y": 2*x} for i,x in enumerate([0.,1.,3.,4.])]
    generated = [{"generated_world_id": str(i//2), "x": x, "y": 2*x} for i,x in enumerate([0.,2.,5.,8.])]
    kwargs = dict(iterations=100, confidence_level=.95, seed=4)
    first = morphology_intervals(reference, generated, ["x","y"], {"x","y"}, **kwargs)
    assert first == morphology_intervals(reference, generated, ["x","y"], {"x","y"}, **kwargs)
    assert first['aggregate'] == first['x'] == first['y']


def test_host_effects_use_valid_seed_pairs_and_per_seed_differences():
    def row(seed, condition, beta):
        return dict(seed=seed,condition_id=condition,status="complete",cyclomatic_number=beta,centerline_displacement_mean_m=0. if condition=="full" else 3.)
    rows = [row(1,"full",100),row(2,"full",0),row(3,"full",1000),
            row(1,"no_slope",102),row(2,"no_slope",100),row(4,"no_slope",9999)]
    paired = _summary(rows,("full","no_slope"))["paired_effects"]["no_slope"]
    assert paired["valid_pairs"] == 2
    assert paired["effects"]["cyclomatic_number"]["median_delta"] == 51


def test_parallel_cases_preserve_failures_and_resume_without_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("PLUME_EVALUATION_WORKERS", "2")
    store = ResultStore(tmp_path, "parallel")
    tasks = []
    for seed in range(4):
        template = ExperimentResult("parallel", f"case-{seed}", "test", seed, "complete")
        def operation(seed=seed):
            if seed == 2:
                raise ValueError("expected failure")
            return {"square":seed**2}
        tasks.append((template,operation))
    run_cases(store,tasks)
    assert [r['status'] for r in store.rows()] == ['complete','complete','failed','complete']
    original = store.case_path('case-0').read_bytes()
    run_cases(store,tasks[:1])
    assert original == store.case_path('case-0').read_bytes()


def test_nonfinite_metrics_become_a_failed_case(tmp_path):
    store = ResultStore(tmp_path, 'nonfinite')
    template = ExperimentResult('nonfinite','bad','test',1,'complete')
    result = run_case(store,template,lambda:{'value':float('nan')})
    assert result.status == 'failed'
    json.loads(store.case_path('bad').read_text())


def test_resource_guard_records_memory_failure(tmp_path: Path):
    command = [sys.executable,'-c','import time; a=bytearray(30*1024**2); time.sleep(5)']
    with pytest.raises(RuntimeError) as error:
        monitor_worker(command,tmp_path,3.,.02,psutil)
    assert error.value.metrics['failure_kind'] == 'memory_limit'
    assert error.value.metrics['peak_rss_gib'] > .02
    assert error.value.metrics['wall_time_s'] < 3.


def test_resource_guard_records_timeout(tmp_path: Path):
    with pytest.raises(TimeoutError) as error:
        monitor_worker([sys.executable,'-c','import time; time.sleep(5)'],tmp_path,.1,1.,psutil)
    assert error.value.metrics['failure_kind'] == 'timeout'
