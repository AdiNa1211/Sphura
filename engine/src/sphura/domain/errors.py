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


class ConflictError(SphuraError):
    """The thing was changed elsewhere since it was read (e.g. another editor tab)."""


class ToolchainError(SphuraError):
    """FFmpeg/FFprobe missing, broken, or a command failed."""
