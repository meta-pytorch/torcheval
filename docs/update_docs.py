#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import os
from pathlib import Path


def get_submodule_vars(filename) -> tuple[str, list[str]]:
    """This function reads the init files which are inside of the metrics/ and functional/ subdirectories.
    Each subdirectory within these two folders are associated with a domain of metrics, e.g. classification or regression.
    This function assumes that each of the init files has two variables defined.

    1. __all__ (List[str]): A list of all the metrics in that sudirectory, in alphabetical order.
    The order of this list defines the order the metrics will show up in the docs.

    2. __doc_name__ (str): The heading title for that domain of metrics.
    This string is pulled as used in the docs as the header for that domain of metrics
    (e.g. /window has __doc_name__ = 'Windowed Metrics')

    returns:
        doc_name (str): defined above
        all_modules (List[str]): defined above
    """
    with open(filename) as file:
        tree = ast.parse(file.read())

    doc_name = None
    all_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    if target.id == "__doc_name__":
                        doc_name = node.value.value
                    elif target.id == "__all__":
                        all_modules = [e.value for e in node.value.elts]

    return doc_name, all_modules


def getMetricsRST(eval_dir: Path) -> str:
    """Constructs the rst page for the Class based metrics. This function reads the metrics/ subdirectory
    and automatically populates the docs.

    returns:
    torcheval.metrics.rst (str)"""
    metrics_dir = eval_dir / "metrics"

    header = """Metrics
=============

.. automodule:: torcheval.metrics


"""
    ignore_folders = ["functional"]
    metric_code = ""
    for filename in sorted(os.listdir(metrics_dir)):
        if filename not in ignore_folders:
            init_path = os.path.join(metrics_dir, filename, "__init__.py")
            if os.path.exists(init_path):
                doc_name, all_metrics = get_submodule_vars(init_path)
                metric_code += f"""{doc_name}
-------------------------------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

"""
                for metric in all_metrics:
                    metric_code += f"   {metric}\n"
                metric_code += "\n"

    return header + metric_code


def getFunctionalMetricRST(eval_dir: Path) -> str:
    """Constructs the rst page for the functional metrics. This function reads the metrics/functional/ subdirectory
    and automatically populates the docs.

    returns:
    torcheval.metrics.rst (str)"""
    metrics_dir = eval_dir / "metrics" / "functional"

    header = """Functional Metrics
==================

.. automodule:: torcheval.metrics.functional

"""
    metric_code = ""
    for filename in sorted(os.listdir(metrics_dir)):
        init_path = os.path.join(metrics_dir, filename, "__init__.py")
        if os.path.exists(init_path):
            doc_name, all_metrics = get_submodule_vars(init_path)
            metric_code += f"""{doc_name}
-------------------------------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

"""
            for metric in all_metrics:
                metric_code += f"   {metric}\n"
            metric_code += "\n"

    return header + metric_code


def main() -> None:
    """This code reads through the code tree and automatically populates the torcheval API docs for the
    functional and class based metrics."""
    doc_dir = Path(os.path.abspath(__file__)).parent / "source"
    eval_dir = Path(os.path.abspath(__file__)).parent.parent / "torcheval"

    # ======================
    # torcheval.metrics.rst
    # ======================

    with open(doc_dir / "torcheval.metrics.rst", "w") as f:
        f.write(getMetricsRST(eval_dir))
    with open(doc_dir / "torcheval.metrics.functional.rst", "w") as f:
        f.write(getFunctionalMetricRST(eval_dir))


if __name__ == "__main__":
    main()
