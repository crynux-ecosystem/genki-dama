from enum import IntEnum

class CompetitionId(IntEnum):
    """Unique identifiers for each competition."""

    ChIPTUNE_MUSIC_MODEL = 1

    # Overwrite the default __repr__, which doesn't work with
    # bt.logging for some unknown reason.
    def __repr__(self) -> str:
        return f"{self.value}"
