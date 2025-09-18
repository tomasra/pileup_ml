import os
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import open3d as o3d

from pileup_ml.detectors.pixels import PixelModule, PixelDetector, RowColMapping
from pileup_ml.events import PixelDigiEvent


def create_rectangle_wireframe(width, height, thickness=0.0, color=[1, 0, 0], alpha=0.5):
    """
    Create a rectangle (wireframe) in the XY-plane centered at origin in local coordinates.
    If thickness > 0, returns a thin box wireframe.
    """
    if thickness > 0:
        box = o3d.geometry.TriangleMesh.create_box(width=width,
                                                   height=height,
                                                   depth=thickness)
        box.translate([-width/2, -height/2, -thickness/2])
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(box)
    else:
        # flat rectangle in XY-plane
        pts = np.array([
            [-width/2, -height/2, 0],
            [ width/2, -height/2, 0],
            [ width/2,  height/2, 0],
            [-width/2,  height/2, 0],
        ])
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
    line_set.paint_uniform_color(color)
    return line_set


def place_module(module: PixelModule, thickness=0.0, color=[0, 1, 0]):
    rect = create_rectangle_wireframe(1.86,
                                      6.66,
                                      thickness,
                                      color)
    # Apply rotation & translation
    rect.rotate(module.rotation.T, center=(0, 0, 0))
    rect.translate(module.position)
    return rect


def visualize_point_cloud(points: np.ndarray,
                          colors: Optional[np.ndarray] = None,
                          scalar_values: Optional[np.ndarray] = None,
                          point_size: int = 5,
                          show: bool = False) -> o3d.geometry.PointCloud:
    """
    Create an Open3D PointCloud from Nx3 numpy points and return the geometry object.

    - points: (N,3) float array
    - colors: optional (N,3) array in [0,1] or [0,255]
    - scalar_values: optional (N,) numeric array to be mapped to a colormap (uses matplotlib if available)
    - point_size: integer (used only if show=True)
    - show: if True, opens an Open3D window to display the point cloud

    Returns:
        o3d.geometry.PointCloud
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # determine colors
    if colors is not None:
        cols = np.asarray(colors)
        if cols.ndim != 2 or cols.shape[0] != points.shape[0] or cols.shape[1] != 3:
            raise ValueError("colors must be shape (N, 3)")
        # normalize if 0-255
        if cols.dtype.kind in ("u", "i") or cols.max() > 1.0:
            cols = cols.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    elif scalar_values is not None:
        try:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            sv = np.asarray(scalar_values)
            norm = mcolors.Normalize(vmin=sv.min(), vmax=sv.max(), clip=True)
            cmap = cm.get_cmap("viridis")
            cols = cmap(norm(sv))[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        except Exception:
            gray = np.full((points.shape[0], 3), 0.5)
            pcd.colors = o3d.utility.Vector3dVector(gray)

    return pcd


def visualize_event(event: PixelDigiEvent):
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("CMS Pixel Tracker", 1280, 768)
    vis.show_settings = True  # left-hand tree view

    # Make detector module lines thinner
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 2

    # Add modules grouped by layer
    for i, m in enumerate(event.detector.bpix_modules + event.detector.fpix_modules):
        geom = place_module(m, thickness=0.0, color=[0.3, 0.3, 0.3])
        # name = f"{m.layer}/module_{i}"
        name = f"module_{i}"
        vis.add_geometry(name, geom, material=mat)

    colors = event.adcs() / 255.0

    pcd = visualize_point_cloud(event.to_global_coords(), scalar_values=colors)
    vis.add_geometry("Event 0 hits", pcd)

    # vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Visualize pixel detector hits from a ROOT file")
    argparser.add_argument("root_file", type=str, help="Path to the ROOT file with pixel digis")
    argparser.add_argument("--event_index", type=int, default=0, help="Index of the event to visualize (0-based)")
    args = argparser.parse_args()

    # Detector description
    pixel_modules = PixelModule.read_json(
        Path(os.environ["DETID_INFO_DIR"]) / Path("detids_bpix.json"),
        Path(os.environ["DETID_INFO_DIR"]) / Path("detids_fpix.json"))
    rowcol_mapping = RowColMapping.read_csv(
        Path(os.environ["DETID_INFO_DIR"]) / Path("rowcol_to_local.csv"))
    pix_det = PixelDetector(pixel_modules, rowcol_mapping)

    # Events
    pixel_digi_events = PixelDigiEvent.read_root(args.root_file, pix_det)
    visualize_event(pixel_digi_events[args.event_index])
