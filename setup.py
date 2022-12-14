import os
import platform
import sys
from distutils import sysconfig
from distutils.command import build
from distutils.command.build_ext import build_ext
from distutils.spawn import spawn

from setuptools import Extension, find_packages, setup
import versioneer

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


