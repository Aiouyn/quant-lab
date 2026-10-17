"""Operational interruptions are distinct from disproven research."""


class ResearchStopped(RuntimeError):
    pass


class ResearchTimeLimit(ResearchStopped):
    """A supervisor deadline, distinct from an operator stop or failed hypothesis."""
    pass
