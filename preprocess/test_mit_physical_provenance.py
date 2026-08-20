#!/usr/bin/env python3
"""Fast contract tests for the MIT physical-cell mapping."""

from __future__ import annotations

import unittest

from mit_physical_provenance import (
    BATCH1_DATE,
    BATCH2_DATE,
    BATCH3_DATE,
    CONTINUATION_SOURCE_BY_BATCH1_CELL,
    PAPER124_BATCH1_REMOVED_CELLS,
    PAPER124_BATCH3_REMOVED_CELLS,
    build_physical_cells,
    physical_cell_by_source,
    physical_test_batteries,
)


class MITPhysicalProvenanceTest(unittest.TestCase):
    def test_source_inventory_joins_five_continuations(self):
        cells = build_physical_cells("source135")
        self.assertEqual(len(cells), 135)
        for batch1_cell, batch2_cell in CONTINUATION_SOURCE_BY_BATCH1_CELL.items():
            physical = physical_cell_by_source(cells, BATCH1_DATE, batch1_cell)
            self.assertIsNotNone(physical)
            self.assertEqual(
                [(item.batch_date, item.cell) for item in physical.source_segments],
                [(BATCH1_DATE, batch1_cell), (BATCH2_DATE, batch2_cell)],
            )
            self.assertIs(
                physical_cell_by_source(cells, BATCH2_DATE, batch2_cell), physical
            )

    def test_paper124_curation_and_source_uniqueness(self):
        cells = build_physical_cells("paper124")
        self.assertEqual(len(cells), 124)
        for cell in PAPER124_BATCH1_REMOVED_CELLS:
            self.assertIsNone(physical_cell_by_source(cells, BATCH1_DATE, cell))
        for cell in PAPER124_BATCH3_REMOVED_CELLS:
            self.assertIsNone(physical_cell_by_source(cells, BATCH3_DATE, cell))
        source_records = [
            (segment.batch_date, segment.cell)
            for physical in cells
            for segment in physical.source_segments
        ]
        self.assertEqual(len(source_records), len(set(source_records)))

    def test_physical_modulo_split_is_balanced_and_uses_physical_ids(self):
        cells = build_physical_cells("paper124")
        test_ids = physical_test_batteries(cells)
        self.assertEqual(len(test_ids), 24)
        self.assertEqual(test_ids[0], "mit_p005")
        self.assertEqual(test_ids[-1], "mit_p120")
        self.assertTrue(all(item.startswith("mit_p") for item in test_ids))


if __name__ == "__main__":
    unittest.main()
