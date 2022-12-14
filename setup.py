import os
import platform
import sys
from distutils import sysconfig
from distutils.command import build
from distutils.command.build_ext import build_ext
from distutils.spawn import spawn

from setuptools import Extension, find_packages, setup

_version_module = None
try:
    from packaging import version as _version_module
except ImportError:
    try:
        from setuptools._vendor.packaging import version as _version_module
    except ImportError:
        pass


min_python_version = "3.8"
max_python_version = "3.12"


def _guard_py_ver():
    if _version_module is None:
        return

    parse = _version_module.parse

    min_py = parse(min_python_version)
    max_py = parse(max_python_version)
    cur_py = parse('.'.join(map(str, sys.version_info[:3])))

    if not min_py <= cur_py < max_py:
        msg = ('Cannot install on Python version {}; only versions >={}, <{} '
                'are supported.')
        raise RuntimeError(msg.format(cur_py, min_py, max_py))


_guard_py_ver()


def get_ext_modules():
    """
    Return a list of Extension instances for the setup() call.
    """

    if os.getenv("HYDROGYM_DISABLE_FIREDRAKE"):
        print("Firedrake engine disabled")
    else:
        # Search for Firedrake to see if it is already available
        firedrake_root = os.getenv("FIREDRAKEROOT")

        if firedrake_root:
            print("Using Firedrake from:", firedrake_root)
            # TODO(ludgerpaehler): This path here needs to filled with content
            #      and to actually be tested by using a Firedrake from a
            #      different virtualenv.
        else:
            print("Firedrake not found")

    ext_modules = []

    return ext_modules

build_requires = []
install_requires = [
    'jupyterlab',
    'ipykernel',
    'notebook',
    'ipywidgets',
    'mpltools',
    'nbformat',
    'nbconvert',
    'dmsuite',
    'gym',
    'memory_profiler',
    'gmsh',
    'modred',
    'seaborn',
    'control',
    '--no-binary=h5py h5py',
    'ray[rllib]',
    'torch',
    'evotorch',
    'tensorboard',
]

metadata = dict(
    name="hydrogym",
    description="A Reinforcement Learning Environment for Fluid Dynamics",
    version="0.1",
    classifiers=[
          "Development Status :: 2 - Pre-Alpha",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Operating System :: MacOS",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Atmospheric Science",
          "Topic :: Scientific/Engineering :: Hydrology",
          "Topic :: Scientific/Engineering :: Physics",
    ],
    setup_requires=build_requires,
    install_requires=install_requires,
    python_requires=">={}".format(min_python_version),
    license="MIT",
)

with open('README.md') as f:
    metadata["long_description"] = f.read()

setup(**metadata)
