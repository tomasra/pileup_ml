import json
from typing_extensions import Self, Any

import numpy as np
import pandas as pd
from pydantic import ConfigDict, BaseModel, ConfigDict, field_validator
from pydantic.dataclasses import dataclass


class RowColMapping:
    # BPIX/FPIX module dimensions
    N_ROWS = 160
    N_COLS = 416

    def __init__(self, coord_map: np.ndarray):
        self.coord_map = coord_map

    def to_local_coords(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return self.coord_map[rows, cols]

    @classmethod
    def read_csv(cls, path: str):
        local_coords_csv = pd.read_csv(path)
        local_coords = local_coords_csv[['local_x', 'local_y']] \
            .to_numpy() \
            .reshape((cls.N_ROWS, cls.N_COLS, 2))
        # Add Z coordinate with zeros
        coord_map = np.concatenate((local_coords, np.zeros((cls.N_ROWS, cls.N_COLS, 1))), axis=2)
        return cls(coord_map)


class PixelModule(BaseModel):
    det_id: int
    position: np.ndarray
    rotation: np.ndarray
    rows: int = 160
    cols: int = 416

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("position", mode="before")
    @classmethod
    def parse_position(cls, v: Any) -> np.ndarray:
        arr = np.array(v, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return arr

    @field_validator("rotation", mode="before")
    @classmethod
    def parse_rotation(cls, v: Any) -> np.ndarray:
        arr = np.array(v, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(f"Expected shape (3, 3), got {arr.shape}")
        return arr

    @staticmethod
    def read_json(bpix_path: str, fpix_path: str) -> list[Self]:
        with open(bpix_path, 'r') as f:
            detids_bpix = json.load(f)
        with open(fpix_path, 'r') as f:
            detids_fpix = json.load(f)

        bpix_modules = [BPIX_Module(**obj) for obj in detids_bpix]
        fpix_modules = [FPIX_Module(**obj) for obj in detids_fpix]
        return bpix_modules + fpix_modules

    def to_global_coords(self, local_coords: np.ndarray) -> np.ndarray:
        """Apply linear transformation to module's coordinate system
        """
        return np.dot(local_coords, self.rotation) + self.position


class BPIX_Module(PixelModule):
    layer: int
    ladder: int
    module: int
    subdet: str = "BPIX"


class FPIX_Module(PixelModule):
    disk: int
    blade: int
    panel: int
    module: int
    side: int
    subdet: str = "FPIX"


class PixelDetector:
    modules: list[PixelModule]
    rowcol_mapping: RowColMapping

    def __init__(self, modules: list[PixelModule], rowcol_mapping: RowColMapping):
        self.modules = modules
        self.rowcol_mapping = rowcol_mapping
        # For fast access by det_id
        self._detid_modules = {m.det_id: m for m in modules}

    def __getitem__(self, det_id: int) -> PixelModule:
        return self._detid_modules[det_id]

    @property
    def bpix_modules(self):
        return [m for m in self.modules if m.subdet == "BPIX"]

    @property
    def fpix_modules(self):
        return [m for m in self.modules if m.subdet == "FPIX"]
