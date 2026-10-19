
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
    QProgressBar
)
from pyvistaqt import QtInteractor
import numpy as np
from src.rasterization import create_grid, assign_vertices
from ui.processing_worker import FlatteningWorker


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Surface Flattening")
        self.resize(1200, 750)

        self.mesh = None
        self.grid_actor = None
        self.mesh_light = None

        self.active_cell_actor = None
        self.processing_grid = None

        self.flattened_mesh = None
        self.cell_planes = None
        self.flattening_statistics = None
        self.worker = None

        self.display_mode = "vertex_colors"
        self.display_mesh_actor = None
        self.displayed_mesh = None

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

        #Display mode button
        self.display_mode_button = QPushButton("Show Gray")
        self.display_mode_button.setCheckable(True)
        self.display_mode_button.setChecked(True)
        self.display_mode_button.toggled.connect(
            self.toggle_display_mode
        )
        self.display_mode_button.setVisible(False)

        controls_layout.addWidget(self.display_mode_button)

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

        # Processing button
        self.run_button = QPushButton("Run Flattening")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.start_flattening)
        controls_layout.addWidget(self.run_button)

        # Processing progress
        self.progress_label = QLabel("Ready to flatten")
        controls_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        controls_layout.addWidget(self.progress_bar)

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

            # Keep the full-resolution mesh for processing.
            self.mesh = mesh

            # Create a lighter copy for interactive display if needed.
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

            # Reset the display mode to vertex colors.
            self.display_mode = "vertex_colors"

            self.display_mode_button.blockSignals(True)
            self.display_mode_button.setChecked(True)
            self.display_mode_button.setText("Show Gray")
            self.display_mode_button.blockSignals(False)
            self.display_mode_button.setVisible(True)

            # Clear the OLD scene before displaying the NEW mesh.
            self.viewer.clear()
            self.grid_actor = None
            self.active_cell_actor = None
            self.display_mesh_actor = None
            self.displayed_mesh = None

            # Display the new mesh with the shared lighting setup.
            self.display_mesh(
                preview_mesh,
                reset_camera=True,
            )

            # Print available arrays for troubleshooting.
            print("Point data arrays:", list(mesh.point_data.keys()))

            # Set the camera and redraw the grid.
            self.viewer.view_xy()
            self.update_grid_overlay()
            self.viewer.render()

            self.file_label.setText(Path(path).name)
            self.run_button.setEnabled(True)

            self.statusBar().showMessage(
                f"Loaded {mesh.n_points:,} points"
            )

        except Exception as error:
            self.statusBar().showMessage(
                f"Could not load mesh: {error}"
            )
            print("Mesh loading failed:", error)

    def display_mesh(self, mesh, reset_camera=False):
        self.displayed_mesh = mesh

        # Find the original vertex-color array, if available.
        color_name = next(
            (
                name
                for name in ("RGB", "RGBA", "Colors", "colors")
                if name in mesh.point_data
                and mesh.point_data[name].ndim == 2
                and mesh.point_data[name].shape[1] in (3, 4)
            ),
            None,
        )

        # Apply the selected appearance.
        if self.display_mode == "vertex_colors" and color_name is not None:
            self.display_mesh_actor = self.viewer.add_mesh(
                mesh,
                scalars=color_name,
                rgb=True,
                smooth_shading=False,
                lighting=True,
                ambient=0.15,
                diffuse=1.0,
                specular=0.0,
                show_edges=False,
                reset_camera=reset_camera,
                name="main_mesh",
            )
        else:
            self.display_mesh_actor = self.viewer.add_mesh(
                mesh,
                color="#c0c0c0",
                smooth_shading=False,
                lighting=True,
                ambient=0.15,
                diffuse=1.0,
                specular=0.0,
                show_edges=False,
                reset_camera=reset_camera,
                name="main_mesh",
            )

        # Add the directional light that reveals surface details.
        # Reset the scene lights so they never accumulate.
        self.viewer.renderer.remove_all_lights()

        self.mesh_light = pv.Light(
            position=(1, -1, 1),
            focal_point=(0, 0, 0),
            light_type="scene light",
        )
        self.mesh_light.intensity = 0.9

        self.viewer.add_light(self.mesh_light)

        self.viewer.render()

    def toggle_display_mode(self, checked):
        self.display_mode = (
            "vertex_colors" if checked else "gray"
        )

        self.display_mode_button.setText(
            "Show Gray" if checked else "Show Vertex Colors"
        )

        if self.displayed_mesh is None:
            return

        if self.display_mesh_actor is not None:
            self.viewer.remove_actor(self.display_mesh_actor)
            self.display_mesh_actor = None

        self.display_mesh(
            self.displayed_mesh,
            reset_camera=False,
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

    def start_flattening(self):
        if self.mesh is None:
            return

        if self.worker is not None and self.worker.isRunning():
            return

        try:
            vertices = np.asarray(self.mesh.points).copy()

            grid = create_grid(
                vertices,
                self.grid_slider.value(),
            )

            cells = assign_vertices(vertices, grid)

            self.processing_grid = grid

            if not cells:
                raise ValueError("No occupied grid cells were found.")

            threshold = self.threshold_input.value()

            # Show the original mesh in gray while processing.
            self.viewer.clear()
            self.grid_actor = None
            self.active_cell_actor = None

            preview_mesh = self.mesh
            self.display_mode = "gray"
            self.display_mesh(preview_mesh, reset_camera=False)
            
            self.update_grid_overlay()
            self.viewer.render()

            # Disable controls while processing
            self.run_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.grid_slider.setEnabled(False)
            self.threshold_input.setEnabled(False)

            self.progress_bar.setValue(0)
            self.progress_label.setText("Starting flattening...")

            self.worker = FlatteningWorker(
                vertices,
                cells,
                threshold,
            )

            self.worker.progress.connect(self.handle_flattening_progress)
            self.worker.processing_finished.connect(
                self.handle_flattening_finished
            )
            self.worker.processing_failed.connect(
                self.handle_flattening_failed
            )

            self.worker.start()

        except Exception as error:
            self.progress_label.setText("Could not start flattening")
            self.statusBar().showMessage(str(error))


    def handle_flattening_progress(
        self, stage, cell, current, total, status
    ):
        stage_names = {
            "fitting_planes": "Fitting planes",
            "assigning_planes": "Assigning planes",
            "flattening": "Flattening cells",
        }

        stage_name = stage_names.get(stage, stage)

        if status == "stage_started":
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"{stage_name}...")

        elif status == "cell_started":
            self.progress_label.setText(
                f"{stage_name}: cell {cell} ({current}/{total})"
            )

            self.highlight_active_cell(cell)

        elif status == "cell_completed":
            percentage = int(current / total * 100) if total else 100
            self.progress_bar.setValue(percentage)

        elif status == "stage_completed":
            self.progress_bar.setValue(100)
            self.progress_label.setText(f"{stage_name} complete")

    def highlight_active_cell(self, cell):
        if self.mesh is None or self.processing_grid is None:
            return

        # Remove the previous highlight.
        if self.active_cell_actor is not None:
            self.viewer.remove_actor(self.active_cell_actor)
            self.active_cell_actor = None

        cx, cy = cell
        grid = self.processing_grid
        cell_size = grid["cell_size"]

        # Calculate this cell's center in the XY grid.
        x = grid["x_min"] + (cx + 0.5) * cell_size
        y = grid["y_min"] + (cy + 0.5) * cell_size
        z = float(np.max(self.mesh.points[:, 2])) + 0.03

        # Draw a translucent highlight over the active cell.
        highlight = pv.Plane(
            center=(x, y, z),
            direction=(0, 0, 1),
            i_size=cell_size,
            j_size=cell_size,
        )

        self.active_cell_actor = self.viewer.add_mesh(
            highlight,
            color="#ffd54f",
            opacity=0.25,
            show_edges=True,
            edge_color="#ff9800",
            line_width=3,
            lighting=False,
            name="active_cell",
        )

        self.viewer.render()

    def handle_flattening_finished(
        self, vertices, cell_planes, statistics
    ):
        # Preserve the original mesh and its topology.
        result_mesh = self.mesh.copy(deep=True)
        result_mesh.points = vertices

        self.flattened_mesh = result_mesh
        self.cell_planes = cell_planes
        self.flattening_statistics = statistics

        # Display the flattened result.
        self.viewer.clear()
        self.grid_actor = None
        self.active_cell_actor = None

        preview_mesh = result_mesh
        
        preview_mesh = result_mesh.copy(deep=True)

        # Remove stale normals from the display copy only.
        if "Normals" in preview_mesh.point_data:
            del preview_mesh.point_data["Normals"]

        if "Normals" in preview_mesh.cell_data:
            del preview_mesh.cell_data["Normals"]

        # Recalculate normals for the updated geometry.
        preview_mesh = preview_mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            consistent_normals=True,
            auto_orient_normals=False,
            inplace=False,
        )
        if result_mesh.n_cells > 200_000:
            try:
                preview_mesh = result_mesh.decimate_pro(
                    0.75,
                    preserve_topology=True,
                )
            except Exception as error:
                print("Preview simplification failed:", error)

        """color_name = next(
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
        """
        self.display_mesh(preview_mesh, reset_camera=False)

        self.update_grid_overlay()

        self.progress_bar.setValue(100)
        self.progress_label.setText("Flattening complete")

        self.statusBar().showMessage(
            f"Flattened {statistics['processed_cells']} cells; "
            f"{statistics['flattened_vertices']:,} vertices changed."
        )

        self.run_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.grid_slider.setEnabled(True)
        self.threshold_input.setEnabled(True)


    def handle_flattening_failed(self, error_message):
        print(error_message)

        self.progress_label.setText("Flattening failed — see terminal")
        self.statusBar().showMessage("Flattening failed. See terminal.")

        self.run_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.grid_slider.setEnabled(True)
        self.threshold_input.setEnabled(True)
