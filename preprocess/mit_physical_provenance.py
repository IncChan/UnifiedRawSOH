"""Canonical physical-cell provenance for the MIT/A123 cycling data.

The three public MIT batch files contain 140 *source records*, not 140
independent cells.  ``LoadData.m`` in the original release appends five
second-batch records to first-batch cells and removes the duplicate
second-batch entries.  The published 124-cell cohort additionally applies the
author's documented batch-3 and batch-1 curation.

This module centralises that mapping so raw extraction, statistical-feature
extraction, manifests, and splits cannot silently disagree about identity.
It intentionally contains no I/O besides pure metadata construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


BATCH1_DATE = "2017-05-12"
BATCH2_DATE = "2017-06-30"
BATCH3_DATE = "2018-04-12"
BATCH_DATES: Tuple[str, str, str] = (BATCH1_DATE, BATCH2_DATE, BATCH3_DATE)

# Exact continuation mapping from the original LoadData.m:
#   batch1(1:5) <- batch2([8, 9, 10, 16, 17])
CONTINUATION_SOURCE_BY_BATCH1_CELL: Mapping[int, int] = {
    1: 8,
    2: 9,
    3: 10,
    4: 16,
    5: 17,
}
CONTINUATION_BATCH2_CELLS = frozenset(CONTINUATION_SOURCE_BY_BATCH1_CELL.values())

# To reproduce the published 124 cells, LoadData.m removes these batch-1
# records after continuation concatenation.  The MATLAB comment calls this
# optional because another modelling objective could retain them; it is
# required for the original 124-cell paper cohort.
PAPER124_BATCH1_REMOVED_CELLS = frozenset({9, 11, 13, 14, 23})

# These are the original batch-3 source-cell numbers after faithfully applying
# LoadData.m's sequential channel/end-cap/noisy-cell removals:
#   remove 38 (channel 46), then endcap > .885 (24, 33), then positions
#   [3, 40, 41] in the surviving vector (original 3, 43, 44).
PAPER124_BATCH3_REMOVED_CELLS = frozenset({3, 24, 33, 38, 43, 44})

# This is an extraction-level label QC, not an official LoadData.m cell
# exclusion.  It must be applied identically to raw and feature products so a
# known impossible capacity spike cannot reach just the raw path.
KNOWN_INVALID_SOURCE_CYCLES: Mapping[Tuple[str, int, int], str] = {
    (
        BATCH1_DATE,
        19,
        39,
    ): (
        "source capacity spike: QDischarge=2.8840845 Ah, "
        "derived SOH=2.621895 (nominal capacity=1.1 Ah)"
    )
}

LOAD_DATA_URL = (
    "https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-"
    "before-capacity-degradation/blob/master/LoadData.m"
)
PAPER_URL = (
    "https://petermattia.com/papers/Severson%2C%20Attia%20et%20al.%20-"
    "%202019%20-%20Data-driven%20prediction%20of%20battery%20cycle%20life%20"
    "before%20capacity%20degradation.pdf"
)


@dataclass(frozen=True)
class SourceSegment:
    """One source-file cell that contributes to a physical lifetime."""

    batch_date: str
    cell: int

    @property
    def source_file_identity(self) -> str:
        return f"{self.batch_date}_battery-{self.cell}"

    def as_dict(self) -> Dict[str, object]:
        return {
            "batch_date": self.batch_date,
            "cell": int(self.cell),
            "source_file_identity": self.source_file_identity,
        }


@dataclass(frozen=True)
class PhysicalCell:
    """A canonical MIT physical-cell identity and its ordered source pieces."""

    physical_cell_id: str
    physical_index: int
    paper_batch: str
    primary_batch_date: str
    source_segments: Tuple[SourceSegment, ...]
    cohort: str
    curation_note: str

    @property
    def output_stem(self) -> str:
        """Stable filename used by both raw and feature canonical products."""

        return f"MIT_{self.primary_batch_date}_physical-{self.physical_index:03d}"

    def as_dict(self) -> Dict[str, object]:
        return {
            "physical_cell_id": self.physical_cell_id,
            "physical_index": int(self.physical_index),
            "paper_batch": self.paper_batch,
            "primary_batch_date": self.primary_batch_date,
            "output_stem": self.output_stem,
            "cohort": self.cohort,
            "curation_note": self.curation_note,
            "source_segments": [segment.as_dict() for segment in self.source_segments],
        }


def source_segments_for_batch1_cell(cell: int) -> Tuple[SourceSegment, ...]:
    """Return the ordered raw source segments for a batch-1 physical cell."""

    if not 1 <= int(cell) <= 46:
        raise ValueError(f"batch-1 cell must be in [1, 46], got {cell}")
    pieces = [SourceSegment(BATCH1_DATE, int(cell))]
    continuation = CONTINUATION_SOURCE_BY_BATCH1_CELL.get(int(cell))
    if continuation is not None:
        pieces.append(SourceSegment(BATCH2_DATE, continuation))
    return tuple(pieces)


def _append_cells(
    output: List[PhysicalCell],
    paper_batch: str,
    primary_batch_date: str,
    items: Iterable[Tuple[Tuple[SourceSegment, ...], str]],
    cohort: str,
) -> None:
    for segments, note in items:
        index = len(output) + 1
        output.append(
            PhysicalCell(
                physical_cell_id=f"mit_p{index:03d}",
                physical_index=index,
                paper_batch=paper_batch,
                primary_batch_date=primary_batch_date,
                source_segments=segments,
                cohort=cohort,
                curation_note=note,
            )
        )


def build_physical_cells(cohort: str = "paper124") -> List[PhysicalCell]:
    """Build the canonical physical-cell inventory.

    ``source135`` only joins the five continuation records.  ``paper124`` is
    the release used for the MIT paper: it joins continuations and applies the
    original author curation.  IDs are stable within each cohort and are
    deliberately unrelated to source-file suffixes.
    """

    cohort = str(cohort)
    if cohort not in {"source135", "paper124"}:
        raise ValueError("cohort must be 'source135' or 'paper124'")

    output: List[PhysicalCell] = []

    batch1_removed = PAPER124_BATCH1_REMOVED_CELLS if cohort == "paper124" else frozenset()
    batch1_items = []
    for cell in range(1, 47):
        if cell in batch1_removed:
            continue
        segments = source_segments_for_batch1_cell(cell)
        if len(segments) == 2:
            note = (
                "LoadData.m continuation: append "
                f"{segments[1].source_file_identity} after {segments[0].source_file_identity}"
            )
        else:
            note = "single source record"
        batch1_items.append((segments, note))
    _append_cells(output, "batch1", BATCH1_DATE, batch1_items, cohort)

    # The five batch-2 records above are no longer independent cells after
    # continuation append; all remaining batch-2 source cells are physical
    # cells in their own right.
    batch2_items = [
        ((SourceSegment(BATCH2_DATE, cell),), "single source record")
        for cell in range(1, 49)
        if cell not in CONTINUATION_BATCH2_CELLS
    ]
    _append_cells(output, "batch2", BATCH2_DATE, batch2_items, cohort)

    batch3_removed = PAPER124_BATCH3_REMOVED_CELLS if cohort == "paper124" else frozenset()
    batch3_items = [
        ((SourceSegment(BATCH3_DATE, cell),), "single source record")
        for cell in range(1, 47)
        if cell not in batch3_removed
    ]
    _append_cells(output, "batch3", BATCH3_DATE, batch3_items, cohort)

    expected = 124 if cohort == "paper124" else 135
    if len(output) != expected:
        raise AssertionError(
            f"{cohort} inventory should contain {expected} physical cells, got {len(output)}"
        )
    validate_physical_cells(output, expected_count=expected)
    return output


def validate_physical_cells(
    physical_cells: Sequence[PhysicalCell], expected_count: int | None = None
) -> None:
    """Fail loudly if an identity/provenance mapping is internally invalid."""

    cells = list(physical_cells)
    ids = [cell.physical_cell_id for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("physical_cell_id values are not unique")
    indices = [cell.physical_index for cell in cells]
    if indices != list(range(1, len(cells) + 1)):
        raise ValueError("physical indices must be contiguous and ordered from one")
    if expected_count is not None and len(cells) != int(expected_count):
        raise ValueError(
            f"expected {expected_count} physical cells, got {len(cells)}"
        )

    source_owner: Dict[Tuple[str, int], str] = {}
    for physical in cells:
        if not physical.source_segments:
            raise ValueError(f"{physical.physical_cell_id} has no source segments")
        for segment in physical.source_segments:
            key = (segment.batch_date, int(segment.cell))
            previous = source_owner.setdefault(key, physical.physical_cell_id)
            if previous != physical.physical_cell_id:
                raise ValueError(
                    f"source record {key} belongs to both {previous} and "
                    f"{physical.physical_cell_id}"
                )

    # All five public continuation source records must be attached to their
    # batch-1 physical parent, never listed as an independent batch-2 cell.
    for batch1_cell, batch2_cell in CONTINUATION_SOURCE_BY_BATCH1_CELL.items():
        owner = source_owner.get((BATCH2_DATE, batch2_cell))
        if owner is None:
            continue  # expected only for a deliberately reduced subset
        cell = next(item for item in cells if item.physical_cell_id == owner)
        expected_parent = (BATCH1_DATE, batch1_cell)
        if expected_parent not in {
            (piece.batch_date, piece.cell) for piece in cell.source_segments
        }:
            raise ValueError(
                f"continuation {BATCH2_DATE} cell {batch2_cell} is not joined to "
                f"batch-1 cell {batch1_cell}"
            )


def source_cycle_is_known_invalid(batch_date: str, cell: int, source_cycle: int) -> str | None:
    """Return the documented QC reason for a source cycle, if any."""

    return KNOWN_INVALID_SOURCE_CYCLES.get((str(batch_date), int(cell), int(source_cycle)))


def physical_test_batteries(
    physical_cells: Sequence[PhysicalCell], modulus: int = 5, remainder: int = 0
) -> List[str]:
    """Return deterministic test cells based on canonical physical index.

    This is intentionally *not* the historical source-file suffix rule.  It
    produces an auditably stable physical-cell split and cannot place segments
    of one continuation cell in separate roles.
    """

    modulus = int(modulus)
    remainder = int(remainder)
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return [
        cell.physical_cell_id
        for cell in physical_cells
        if cell.physical_index % modulus == remainder
    ]


def physical_cell_by_source(
    physical_cells: Sequence[PhysicalCell], batch_date: str, cell: int
) -> PhysicalCell | None:
    """Find the physical identity that owns one source record."""

    key = (str(batch_date), int(cell))
    for physical in physical_cells:
        if key in {(item.batch_date, item.cell) for item in physical.source_segments}:
            return physical
    return None
