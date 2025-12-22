# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

from datetime import date

from setuptools import find_packages, setup
from torcheval import __version__


def current_path(file_name: str) -> str:
    return os.path.abspath(os.path.join(__file__, os.path.pardir, file_name))


def read_requirements(file_name: str) -> list[str]:
    with open(current_path(file_name), encoding="utf8") as f:
        return f.read().strip().split()


def get_nightly_version() -> str:
    return date.today().strftime("%Y.%m.%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torcheval setup")
    parser.add_argument(
        "--nightly",
        dest="nightly",
        action="store_true",
        help="enable settings for nightly package build",
    )
    parser.set_defaults(nightly=False)
    return parser.parse_known_args()


if __name__ == "__main__":
    with open(current_path("README.md"), encoding="utf8") as f:
        readme = f.read()

    custom_args, setup_args = parse_args()
    package_name = "torcheval" if not custom_args.nightly else "torcheval-nightly"
    version = __version__ if not custom_args.nightly else get_nightly_version()
    print(f"using package_name={package_name}, version={version}")

    sys.argv = [sys.argv[0]] + setup_args

    setup(
        name=package_name,
        version=version,
        author="torcheval team",
        author_email="yicongd@fb.com",
        description="A library for providing a simple interface to create new metrics and an easy-to-use toolkit for metric computations and checkpointing.",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/torcheval",
        license="BSD-3",
        keywords=["pytorch", "evaluation", "metrics"],
        python_requires=">=3.7",
        install_requires=read_requirements("requirements.txt"),
        packages=find_packages(),
        package_data={"torcheval": ["py.typed"]},
        zip_safe=True,
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        extras_require={
            "audio": read_requirements("audio-requirements.txt"),
            "dev": read_requirements("dev-requirements.txt"),
            "image": read_requirements("image-requirements.txt"),
        },
    )
