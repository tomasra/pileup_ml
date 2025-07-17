from typing_extensions import Self

import uproot
import numpy as np
from tqdm import tqdm

from pileup_ml.detectors.pixels import PixelModule, PixelDetector, RowColMapping


class PixelDigiEvent:
    # TODO: refactor this to store and access row/col/adc in a more efficient way
    """Represents digi-level pixel detector hits of a single event
    """
    def __init__(self,
                 id_: int,
                 det_id_hits: dict[int, dict],
                 detector: PixelDetector):
        self.id_ = id_
        self.det_id_hits = det_id_hits
        self.detector = detector

    def __len__(self):
        """Number of pixel hits in the event
        """
        return sum(len(hits["row"]) for hits in self.det_id_hits.values())

    @property
    def det_ids(self):
        return self.det_id_hits.keys()

    def to_global_coords(self) -> np.ndarray:
        """Iterate through pixel detector modules and convert
        their digitized hits in local module coordinates to global.
        """
        global_coords_det_ids = []
        for det_id in self.det_ids:
            pixel_module = self.detector[det_id]
            local_coords = self._to_local_coords(det_id, self.detector.rowcol_mapping)
            global_coords_det_id = pixel_module.to_global_coords(local_coords)
            global_coords_det_ids.append(global_coords_det_id)

        global_coords = np.concatenate(global_coords_det_ids)
        return global_coords

    def to_images(self) -> list[tuple[PixelModule, np.ndarray]]:
        """Represent digitized hits as 2D images for each pixel module.
        ADC values are used as pixel intensities and normalized to [0, 1] range.
        """
        images = []
        for det_id in self.det_ids:
            module = self.detector[det_id]
            img = np.zeros((module.rows, module.cols), dtype=np.float32)
            rows = self.det_id_hits[det_id]['row']
            cols = self.det_id_hits[det_id]['col']
            adcs = self.det_id_hits[det_id]['adc']
            img[rows, cols] = adcs
            img /= 255.0  # if ADC max value is 255
            images.append((module, img))
        return images

    @staticmethod
    def read_root(path: str, detector: PixelDetector, branch="analyzer/digiTree") -> list[Self]:
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
            events.append(PixelDigiEvent(event_id, det_dict, detector))
        return events

    def _to_local_coords(self, det_id) -> np.ndarray:
        """Convert row/col hits to local coordinates of a pixel module
        """
        rowcol_mapping = self.detector.rowcol_mapping
        det_rows = self.det_id_hits[det_id]["row"]
        det_cols = self.det_id_hits[det_id]["col"]
        local_coords = rowcol_mapping.to_local_coords(det_rows, det_cols)
        return local_coords
