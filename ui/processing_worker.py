
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.flattening import flatten_mesh
from src.plane_fitting import fit_plane_pca


class FlatteningWorker(QThread):
    # stage, cell, current, total, status
    progress = Signal(str, object, int, int, str)

    # Flattened vertices, fitted planes, statistics
    processing_finished = Signal(object, object, object)

    # Error message if processing fails
    processing_failed = Signal(str)

    def __init__(self, vertices, cells, threshold):
        super().__init__()

        self.vertices = np.array(vertices, copy=True)
        self.cells = cells
        self.threshold = threshold

    def run(self):
        try:
            new_vertices, cell_planes, statistics = flatten_mesh(
                self.vertices,
                self.cells,
                fit_plane_pca,
                self.threshold,
                progress_callback=self.report_progress,
            )

            self.processing_finished.emit(
                new_vertices,
                cell_planes,
                statistics,
            )

        except Exception:
            self.processing_failed.emit(traceback.format_exc())

    def report_progress(self, stage, cell, current, total, status):
        self.progress.emit(
            stage,
            cell,
            current,
            total,
            status,
        )
