from typing_extensions import Self

import uproot
import numpy as np
from tqdm import tqdm

from pileup_ml.detectors.pixels import PixelDetector, RowColMapping


class PixelDigiEvent:
    # TODO: refactor this to store and access row/col/adc in a more efficient way
    """Represents digi-level pixel detector hits of a single event
    """
    def __init__(self, id_: int, det_id_hits: dict[int, dict]):
        self.det_id_hits = det_id_hits

    @property
    def det_ids(self):
        return self.det_id_hits.keys()

    def to_global_coords(self, pix_det: PixelDetector):
        """Iterate through pixel detector modules and convert
        their digitized hits in local module coordinates to global.
        """
        global_coords_det_ids = []
        for det_id in self.det_ids:
            pixel_module = pix_det[det_id]
            local_coords = self._to_local_coords(det_id, pix_det.rowcol_mapping)
            global_coords_det_id = pixel_module.to_global_coords(local_coords)
            global_coords_det_ids.append(global_coords_det_id)

        global_coords = np.concatenate(global_coords_det_ids)
        return global_coords


    @staticmethod
    def read_root(path: str, branch="analyzer/digiTree") -> list[Self]:
        file = uproot.open(path)
        df = file[branch].arrays(library="pd")

        events = []
        grouped = df.groupby("event")

        for event_id, event_df in tqdm(grouped):
            det_dict = {}
            for det_id, det_df in event_df.groupby("detId"):
                det_dict[int(det_id)] = {
                    "row": det_df["row"].to_numpy().astype(np.uint16),
                    "col": det_df["col"].to_numpy().astype(np.uint16),
                    "adc": det_df["adc"].to_numpy().astype(np.uint8),
                }
            events.append(PixelDigiEvent(event_id, det_dict))
        return events

    def _to_local_coords(self, det_id, rowcol_mapping: RowColMapping) -> np.ndarray:
        """Convert row/col hits to local coordinates of a pixel module
        """
        det_rows = self.det_id_hits[det_id]["row"]
        det_cols = self.det_id_hits[det_id]["col"]
        local_coords = rowcol_mapping.to_local_coords(det_rows, det_cols)
        return local_coords
