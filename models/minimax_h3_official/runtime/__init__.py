"""Official H3 bindings to the shared Component/Preparation startup protocol."""
from core_infra.runtime import ProcessGroup, ReaderBudget
from .framework import FrameworkH3Runtime


class H3Runtime(FrameworkH3Runtime):
    """Public import; Rust core owns startup scheduling and before-use gates."""
