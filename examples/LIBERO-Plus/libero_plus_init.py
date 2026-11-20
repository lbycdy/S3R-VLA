# LIBERO-Plus task selection by perturbation dimension. Replace LIBERO-Plus/libero/libero/benchmark/__init__.py with this
# file (task_classification.json stays in the same directory).
# Compared with the replacement file distributed with VLA-JEPA it
#   - converts the 1-based ids of task_classification.json to 0-based task indices and checks every task name (used
#     directly, the 1-based ids select the next task, and the last task of each suite is dropped: 10,026 of 10,030);
#   - loads the init states with torch.load(path), which the torch 1.11 of the LIBERO environment supports
#     (torch.load(..., weights_only=False) raises a TypeError there);
#   - keeps the classification order as task order 0 (the shuffled orders 1-19 use their own seeded generator).
# The init states of the selected tasks are the same as with the VLA-JEPA file.
import abc
import glob
import json
import os
import random
import torch
import re

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
import libero.libero.envs.bddl_utils as BDDLUtils

BENCHMARK_MAPPING = {}
LIBERO_PLUS_SUITE_ORDER = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
TASK_CLASSIFICATION_PATH = os.path.join(
    os.path.dirname(__file__),
    "task_classification.json",
)


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(suite_name, x):
    if "_language_" not in x:
        if x[0].isupper():  # LIBERO-100
            if "SCENE10" in x:
                language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
            else:
                language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
        else:
            language = " ".join(x.split("_"))
        en = language.find(".bddl")
        return language[:en]
    else:
        if "_view_" in x:
            bddl_file_path = os.path.join(
                get_libero_path("bddl_files"),
                suite_name,
                x.split("_view_")[0]+'.bddl',
            )
        else:
            bddl_file_path = os.path.join(
                get_libero_path("bddl_files"),
                suite_name,
                x,
            )
        # print("bddl_file_path:", bddl_file_path)
        problem_info = BDDLUtils.get_problem_info(bddl_file_path)
        return problem_info["language_instruction"]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(libero_suite, task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")

def _classification_id_offset(suite_name, records):
    """Return the offset needed to convert classification IDs to Python indices."""
    raw_ids = [int(record["id"]) for record in records]
    num_tasks = len(task_maps[suite_name])

    is_zero_based = all(0 <= raw_id < num_tasks for raw_id in raw_ids)
    is_one_based = all(1 <= raw_id <= num_tasks for raw_id in raw_ids)

    if 0 in raw_ids and is_zero_based:
        return 0
    if num_tasks in raw_ids and is_one_based:
        return 1
    if is_zero_based and not is_one_based:
        return 0
    if is_one_based and not is_zero_based:
        return 1

    raise ValueError(
        f"Cannot determine task ID convention for {suite_name}: "
        f"range=[{min(raw_ids)}, {max(raw_ids)}], num_tasks={num_tasks}"
    )


def get_ids_by_category(category_value):
    """
    Retrieve the list of matching IDs from all suites based on the given category value.

    Args:
        category_value (str): The category value to query, e.g., "Background Textures".
        Valid category values include:
            - Background Textures
            - Camera Viewpoints
            - Language Instructions
            - Light Conditions
            - Objects Layout
            - Robot Initial States
            - Sensor Noise

    Returns:
        dict: A dictionary in the format {suite_name: [id1, id2, ...]}.
    """
    with open(TASK_CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}

    for suite_name in LIBERO_PLUS_SUITE_ORDER:
        records = data.get(suite_name, [])
        if not records:
            continue
        if suite_name not in task_maps:
            raise KeyError(f"Unknown LIBERO suite in classification: {suite_name}")

        id_offset = _classification_id_offset(suite_name, records)
        task_names = list(task_maps[suite_name].keys())
        matching_ids = []

        for record in records:
            if record.get("category") != category_value:
                continue

            raw_id = int(record["id"])
            source_task_id = raw_id - id_offset
            if not 0 <= source_task_id < len(task_names):
                raise IndexError(
                    f"Invalid classification ID for {suite_name}: raw_id={raw_id}, "
                    f"offset={id_offset}, num_tasks={len(task_names)}"
                )

            expected_name = record.get("name")
            actual_name = task_names[source_task_id]
            if expected_name is not None and expected_name != actual_name:
                raise ValueError(
                    f"Classification/task-map mismatch for {suite_name} raw_id={raw_id}: "
                    f"classification={expected_name!r}, task_map={actual_name!r}"
                )

            matching_ids.append(source_task_id)

        if matching_ids:
            # Order 0 is the stable classification order used for evaluation.
            # Other orders remain available, but use independent lists and a
            # deterministic local RNG instead of repeatedly aliasing/shuffling
            # the same list object.
            orders = [matching_ids.copy()]
            rng = random.Random(f"libero-plus:{category_value}:{suite_name}")
            for _ in range(19):
                order = matching_ids.copy()
                rng.shuffle(order)
                orders.append(order)
            result[suite_name] = orders

    return result

"""
suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
task_num = [2402, 2518, 2591, 2519, 90]
task_order_dict = dict()

for idx in range(5):
    task_orders = [list(range(0,task_num[idx]))]
    for _ in range(19):
        order = list(range(0,task_num[idx]))
        random.shuffle(order)
        task_orders.append(order)
    task_order_dict[suite_order[idx]] = task_orders
"""

class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        self.task_embs = None
        self.task_order_index = task_order_index
        print("-------LIBERO-PLUS-CATEGORY:", category_value)
        self.task_order_dict = get_ids_by_category(category_value)

    def _make_benchmark(self):
        self.source_suites = []
        self.source_task_ids = []

        if self.name != "libero_mix":
            tasks = list(task_maps[self.name].values())
            indices = self.task_order_dict[self.name][self.task_order_index]
            print(f"[info] using task orders {indices}")
            invalid = [index for index in indices if not 0 <= index < len(tasks)]
            if invalid:
                raise IndexError(
                    f"Invalid task indices for {self.name}: {invalid[:10]} "
                    f"(num_tasks={len(tasks)})"
                )
            self.tasks = [tasks[index] for index in indices]
            self.source_suites = [self.name] * len(indices)
            self.source_task_ids = list(indices)
            self.n_tasks = len(self.tasks)
        else:
            self.tasks = []
            for each_suite in LIBERO_PLUS_SUITE_ORDER:
                if each_suite not in self.task_order_dict:
                    continue
                tasks = list(task_maps[each_suite].values())
                indices = self.task_order_dict[each_suite][self.task_order_index]
                invalid = [index for index in indices if not 0 <= index < len(tasks)]
                if invalid:
                    raise IndexError(
                        f"Invalid task indices for {each_suite}: {invalid[:10]} "
                        f"(num_tasks={len(tasks)})"
                    )
                self.tasks.extend(tasks[index] for index in indices)
                self.source_suites.extend([each_suite] * len(indices))
                self.source_task_ids.extend(indices)
            self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states_ori(self, i):
        if "_table_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_table_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        elif "_tb_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_tb_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        elif "_view_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_view_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        else:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file,
            )

        init_states = torch.load(init_states_path)
        return init_states
    
    def get_task_init_states_path(self, i):
        """Resolve the exact init-state file used by a perturbation task."""
        task = self.tasks[i]
        filename = task.init_states_file
        init_states_root = get_libero_path("init_states")

        if "_language_" in filename:
            filename = (
                filename.split("_language_")[0]
                + "."
                + filename.split(".")[-1]
            )
            return os.path.join(init_states_root, task.problem_folder, filename)
        if "_view_" in filename:
            filename = (
                filename.split("_view_")[0]
                + "."
                + filename.split(".")[-1]
            )
            return os.path.join(init_states_root, task.problem_folder, filename)
        if re.search(r"_table_\d+(?=\.)", filename):
            filename = re.sub(r"_table_\d+", "", filename)
            return os.path.join(init_states_root, task.problem_folder, filename)
        if re.search(r"_tb_\d+(?=\.)", filename):
            filename = re.sub(r"_tb_\d+", "", filename)
            return os.path.join(init_states_root, task.problem_folder, filename)
        if re.search(r"_light_\d+(?=\.)", filename):
            filename = (
                filename.split("_light_")[0]
                + "."
                + filename.split(".")[-1]
            )
            return os.path.join(init_states_root, task.problem_folder, filename)
        if "_add_" in filename or "_level" in filename:
            return os.path.join(
                init_states_root,
                "libero_newobj",
                task.problem_folder,
                filename,
            )
        return os.path.join(init_states_root, task.problem_folder, filename)

    def get_task_init_states(self, i):
        init_states_path = self.get_task_init_states_path(i)
        init_states = torch.load(init_states_path)
        if (
            "_add_" in self.tasks[i].init_states_file
            or "_level" in self.tasks[i].init_states_file
        ):
            init_states = init_states.reshape(1, -1)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        super().__init__(task_order_index=task_order_index, category_value=category_value)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        super().__init__(task_order_index=task_order_index, category_value=category_value)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        super().__init__(task_order_index=task_order_index, category_value=category_value)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        super().__init__(task_order_index=task_order_index, category_value=category_value)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        super().__init__(task_order_index=task_order_index, category_value=category_value)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        super().__init__(task_order_index=task_order_index, category_value=category_value)
        self.name = "libero_100"
        self._make_benchmark()

@register_benchmark
class LIBERO_MIX(Benchmark):
    def __init__(self, task_order_index=0, category_value="Camera Viewpoints"):
        super().__init__(task_order_index=task_order_index, category_value=category_value)
        self.name = "libero_mix"
        self._make_benchmark()
