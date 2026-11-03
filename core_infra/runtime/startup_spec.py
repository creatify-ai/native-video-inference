"""Immutable preparation declarations shared by every native executor binding."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Preparation:
    name: str
    before: tuple
    run: object
    validate: object
    inputs: tuple = ()
    prepare: object = None
    validate_prepared: object = None
    weights: tuple = ()
    at_use: object = False
    after: tuple = ()
    serial_group: object = None
    initialize: object = None
    validate_initialize: object = None
