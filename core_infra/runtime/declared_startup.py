"""Canonical Component/Preparation startup API, executed by Rust core.

The Python module binds model functions and transports. It contains no alternate
startup scheduler; the previous implementation is retained only in CPU tests.
"""
try:
    from .native_startup import DeclaredStartup
    from .startup_spec import Preparation
except ImportError:
    from native_startup import DeclaredStartup
    from startup_spec import Preparation

__all__ = ['DeclaredStartup', 'Preparation']
