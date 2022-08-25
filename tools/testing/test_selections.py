import os
import subprocess

from typing import Any, Callable, Dict, List, Optional

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests


class TestsToRun:
    def __init__(self, num_procs: int = 3) -> None:
        self.large_tests: List[str] = []
        self.must_serial: List[str] = []
        self.other_tests: List[List[str]] = [[] for _ in range(num_procs)]
        self.total_time = 0.0

    def update_total_time(self, job_times: Dict[str, float]) -> None:
        total = sum(job_times[test] for test in self.must_serial)
        total += max(
            (sum(job_times[test] for test in proc) for proc in self.other_tests)
        )
        self.total_time = total

    def add_to_other_tests(self, test: str, job_times: Dict[str, float]) -> None:
        min_proc = min(self.other_tests, key=lambda x: sum(job_times[t] for t in x))
        min_proc.append(test)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "large_tests": self.large_tests,
            "must_serial": self.must_serial,
            "other_tests": self.other_tests,
            "total_time": self.total_time,
        }

    def get_all_tests(self) -> List[str]:
        return (
            self.large_tests
            + self.must_serial
            + [test for proc in self.other_tests for test in proc]
        )


def calculate_shards(
    num_shards: int,
    tests: List[str],
    job_times: Dict[str, float],
    must_serial: Optional[Callable[[str], bool]] = None,
) -> List[List[str]]:
    must_serial = must_serial if must_serial is not None else lambda x: True

    filtered_job_times: Dict[str, float] = dict()
    unknown_tests: List[str] = []
    for test in tests:
        if test in job_times:
            filtered_job_times[test] = job_times[test]
        else:
            unknown_tests.append(test)

    sorted_jobs = sorted(
        filtered_job_times, key=lambda j: filtered_job_times[j], reverse=True
    )
    sharded_jobs: List[TestsToRun] = [TestsToRun() for _ in range(num_shards)]

    for test in sorted_jobs:
        time = filtered_job_times[test]
        min_job = min(sharded_jobs, key=lambda tests_to_run: tests_to_run.total_time)
        if time > 3600:  # if test takes longer than an hour
            for job in sharded_jobs:
                job.large_tests.append(test)
        elif must_serial(test):
            min_job.total_time += time
            min_job.must_serial.append(test)
            min_job.update_total_time(job_times)
        else:
            min_job.add_to_other_tests(test, job_times)
            min_job.update_total_time(job_times)

            # Round robin the unknown jobs starting with the smallest shard
    index = min(range(num_shards), key=lambda i: sharded_jobs[i].total_time)
    for test in unknown_tests:
        sharded_jobs[index].must_serial.append(test)
        index = (index + 1) % num_shards

    return [job.get_all_tests() for job in sharded_jobs]


def _query_changed_test_files() -> List[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'master')}"
    cmd = ["git", "diff", "--name-only", default_branch, "HEAD"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    return lines


def get_reordered_tests(tests: List[str]) -> List[str]:
    """Get the reordered test filename list based on github PR history or git changed file."""
    prioritized_tests: List[str] = []
    if len(prioritized_tests) == 0:
        try:
            changed_files = _query_changed_test_files()
        except Exception:
            # If unable to get changed files from git, quit without doing any sorting
            return tests

        prefix = f"test{os.path.sep}"
        prioritized_tests = [
            f for f in changed_files if f.startswith(prefix) and f.endswith(".py")
        ]
        prioritized_tests = [f[len(prefix) :] for f in prioritized_tests]
        prioritized_tests = [f[: -len(".py")] for f in prioritized_tests]
        print("Prioritized test from test file changes.")

    bring_to_front = []
    the_rest = []

    for test in tests:
        if test in prioritized_tests:
            bring_to_front.append(test)
        else:
            the_rest.append(test)
    if len(tests) == len(bring_to_front) + len(the_rest):
        print(
            f"reordering tests for PR:\n"
            f"prioritized: {bring_to_front}\nthe rest: {the_rest}\n"
        )
        return bring_to_front + the_rest
    else:
        print(
            f"Something went wrong in CI reordering, expecting total of {len(tests)}:\n"
            f"but found prioritized: {len(bring_to_front)}\nthe rest: {len(the_rest)}\n"
        )
        return tests


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
