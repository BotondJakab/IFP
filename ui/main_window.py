
from pathlib import Path

import pyvista as pv

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QFormLayout,
    QStatusBar,
)
from pyvistaqt import QtInteractor
import numpy as np
from src.rasterization import create_grid


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gravestone Surface Flattening")
        self.resize(1200, 750)

        self.mesh = None
        self.grid_actor = None

        # Main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # 3D viewer
        self.viewer = QtInteractor(central_widget)
        self.viewer.set_background("#252a34")
        main_layout.addWidget(self.viewer.interactor, 4)

        # Controls panel
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        title = QLabel("Surface Flattening")
        title.setStyleSheet(
            "font-size: 20px; font-weight: bold;"
        )
        controls_layout.addWidget(title)

        # File selection
        self.file_label = QLabel("No mesh loaded")
        self.file_label.setWordWrap(True)

        self.load_button = QPushButton("Open PLY File")
        self.load_button.clicked.connect(self.load_mesh)

        controls_layout.addWidget(self.file_label)
        controls_layout.addWidget(self.load_button)

        # Parameters
        parameter_box = QGroupBox("Parameters")
        parameter_layout = QFormLayout(parameter_box)

        self.grid_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_slider.setRange(5, 150)
        self.grid_slider.setValue(60)

        self.grid_label = QLabel("60")

        grid_row = QWidget()
        grid_layout = QHBoxLayout(grid_row)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(self.grid_slider)
        grid_layout.addWidget(self.grid_label)

        parameter_layout.addRow("Grid size", grid_row)

        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 20.0)
        self.threshold_input.setSingleStep(0.1)
        self.threshold_input.setValue(1.0)

        parameter_layout.addRow(
            "Flattening threshold",
            self.threshold_input,
        )

        controls_layout.addWidget(parameter_box)

        self.grid_slider.valueChanged.connect(self.update_grid_overlay)

        # Processing button: connected later
        self.run_button = QPushButton("Run Flattening")
        self.run_button.setEnabled(False)
        controls_layout.addWidget(self.run_button)

        controls_layout.addStretch()

        main_layout.addWidget(controls, 1)

        self.setCentralWidget(central_widget)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def load_mesh(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a 3D mesh",
            "",
            "PLY files (*.ply);;All files (*)",
        )

        if not path:
            return

        try:
            mesh = pv.read(path)

            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()

            if mesh.n_points == 0:
                raise ValueError(
                    "The selected file contains no points."
                )


            # Keep the full-resolution mesh for our algorithm
            self.mesh = mesh

            # Print available data arrays so we can identify vertex colors
            print("Point data arrays:", list(mesh.point_data.keys()))

            # Make a lighter copy for interactive display
            preview_mesh = mesh

            if mesh.n_cells > 200_000:
                try:
                    preview_mesh = mesh.decimate_pro(
                        0.75,
                        preserve_topology=True,
                    )
                    print(
                        f"Preview simplified: {mesh.n_cells:,} -> "
                        f"{preview_mesh.n_cells:,} cells"
                    )
                except Exception as error:
                    print("Preview simplification failed:", error)
                    preview_mesh = mesh

            self.viewer.clear()

            # Detect common RGB/RGBA vertex-color arrays
            color_name = next(
                (
                    name
                    for name in ("RGB", "RGBA", "Colors", "colors")
                    if name in preview_mesh.point_data
                    and preview_mesh.point_data[name].ndim == 2
                    and preview_mesh.point_data[name].shape[1] in (3, 4)
                ),
                None,
            )

            if color_name is not None:
                self.viewer.add_mesh(
                    preview_mesh,
                    scalars=color_name,
                    rgb=True,
                    smooth_shading=True,
                    show_edges=False,
                )
            else:
                self.viewer.add_mesh(
                    preview_mesh,
                    color="lightgray",
                    smooth_shading=True,
                    show_edges=False,
                )

            self.viewer.reset_camera()
            self.viewer.view_xy()
            self.update_grid_overlay()

            self.file_label.setText(Path(path).name)
            self.run_button.setEnabled(True)

            self.statusBar().showMessage(
                f"Loaded {mesh.n_points:,} points"
            )

        except Exception as error:
            self.statusBar().showMessage(
                f"Could not load mesh: {error}"
            )

    
    def update_grid_overlay(self, *_):
        if self.mesh is None:
            return

        cell_size = self.grid_slider.value()
        vertices = np.asarray(self.mesh.points)

        if len(vertices) == 0:
            return

        grid = create_grid(vertices, cell_size)

        x_min = grid["x_min"]
        y_min = grid["y_min"]
        nx = grid["nx"]
        ny = grid["ny"]

        # Place the grid just above the mesh's highest point
        z = vertices[:, 2].max() + 0.01

        points = []
        lines = []

        # Vertical grid lines
        for i in range(nx + 1):
            x = x_min + i * cell_size
            start = len(points)
            points.extend([
                [x, y_min, z],
                [x, y_min + ny * cell_size, z],
            ])
            lines.extend([2, start, start + 1])

        # Horizontal grid lines
        for j in range(ny + 1):
            y = y_min + j * cell_size
            start = len(points)
            points.extend([
                [x_min, y, z],
                [x_min + nx * cell_size, y, z],
            ])
            lines.extend([2, start, start + 1])

        grid_mesh = pv.PolyData()
        grid_mesh.points = np.asarray(points, dtype=float)
        grid_mesh.lines = np.asarray(lines, dtype=np.int64)

        # Remove the previous grid without clearing the gravestone
        if hasattr(self, "grid_actor") and self.grid_actor is not None:
            self.viewer.remove_actor(self.grid_actor)

        self.grid_actor = self.viewer.add_mesh(
            grid_mesh,
            color="white",
            line_width=2,
            name="grid_overlay",
        )

        self.grid_label.setText(str(cell_size))
        self.viewer.render()

