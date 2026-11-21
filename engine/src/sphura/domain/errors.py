class SphuraError(Exception):
    """Base for expected, user-facing failures."""


class NotFoundError(SphuraError):
    pass


class InvalidMediaError(SphuraError):
    pass


class UnsupportedContainerError(InvalidMediaError):
    pass


class JobCancelledError(SphuraError):
    pass


class ToolchainError(SphuraError):
    """FFmpeg/FFprobe missing, broken, or a command failed."""
