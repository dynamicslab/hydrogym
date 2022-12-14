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

