from dataclasses import dataclass
from typing_extensions import Self

import cv2
import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm

from pileup_ml.detectors.pixels import PixelModule, PixelDetector, RowColMapping


@dataclass
class PixelModuleHits:
    """Represents pixel hits in a single pixel detector module.
    """
    detector: PixelDetector
    module: PixelModule
    rows: np.ndarray
    cols: np.ndarray
    adcs: np.ndarray

    def __len__(self):
        """Number of pixel hits in the module
        """
        return len(self.adcs)

    @property
    def clusters(self) -> list["PixelModuleHitCluster"]:
        """Find connected pixel hit clusters in the module using 8-connectivity.
        Uses OpenCV's connectedComponents function.

        Returns:
            list[PixelModuleHitCluster]: detected pixel hit clusters
        """
        hit_map = self.to_image(normalize=False)
        mask = hit_map > 0
        n_clusters, cluster_indices = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

        clusters = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_rows, cluster_cols = np.where(cluster_indices == cluster_id)
            cluster_adcs = hit_map[np.where(cluster_indices == cluster_id)]
            cluster = PixelModuleHitCluster(
                detector=self.detector,
                module=self.module,
                rows=cluster_rows,
                cols=cluster_cols,
                adcs=cluster_adcs
            )
            clusters.append(cluster)
        return clusters

    def to_image(self, normalize=True) -> np.ndarray:
        """Represent digitized hits as a 2D image for the pixel module.
        ADC values are used as pixel intensities and normalized to [0, 1] range,
        if `normalize` is True.
        """
        img = np.zeros((self.module.rows, self.module.cols),
                       dtype=np.float32 if normalize else np.uint8)
        img[self.rows, self.cols] = self.adcs
        if normalize:
            img /= 255.0
        return img

    def to_local_points(self) -> np.ndarray:
        """Represent digitized hits of the pixel module as 3D points
        in local module coordinates (x/y/adc).
        """
        local_coords = self.detector.rowcol_mapping.to_local_coords(self.rows, self.cols)
        local_coords = local_coords[:, :2]  # Z coordinate is always zero, drop it
        local_coords = np.concatenate((local_coords, self.adcs[:, np.newaxis]), axis=1)
        return local_coords


@dataclass
class PixelModuleHitCluster(PixelModuleHits):
    # TODO: add methods for cluster properties (size, total adc, centroid, etc)
    pass


class PixelDigiEvent:
    """Represents digi-level pixel detector hits of a single event
    """
    def __init__(self,
                 id_: int,
                 hits: list[PixelModuleHits],
                 detector: PixelDetector):
        self.id_ = id_
        self.hits = hits
        self.detector = detector

    def __len__(self):
        """Number of pixel hits in the event
        """
        return sum([len(mh) for mh in self.hits])

    def __getitem__(self, module: PixelModule) -> PixelModuleHits | None:
        """Get pixel hits for a specific pixel detector module."""
        try:
            return next(h for h in self.hits if h.module == module)
        except StopIteration:
            return None

    def to_global_coords(self) -> np.ndarray:
        """Iterate through pixel detector modules and convert
        their digitized hits in local module coordinates to global.
        """
        global_coords_det_ids = []
        for mh in self.hits:
            local_coords = self.detector.rowcol_mapping.to_local_coords(
                mh.rows, mh.cols)
            global_coords_det_id = mh.module.to_global_coords(local_coords)
            global_coords_det_ids.append(global_coords_det_id)

        global_coords = np.concatenate(global_coords_det_ids)
        return global_coords

    def to_images(self) -> list[tuple[PixelModule, np.ndarray]]:
        """Represent digitized hits as 2D images for each pixel module.
        ADC values are used as pixel intensities and normalized to [0, 1] range.
        """
        return [(mh.module, mh.to_image()) for mh in self.hits]

    def adcs(self) -> np.ndarray:
        """Return all ADC values of the event as a single array.
        """
        return np.concatenate([mh.adcs for mh in self.hits])

    @staticmethod
    def read_root(path: str, detector: PixelDetector, branch="analyzer/digiTree") -> list['PixelDigiEvent']:
        file = uproot.open(path)
        # ROOT file contents as awkward array
        ak_events = file[branch].arrays()
        event_lengths = ak.run_lengths(ak_events['event'])
        pixel_digis = []

        # Event row start/end indices
        event_start_idx = 0
        for event_length in tqdm(event_lengths):
            event_end_idx = event_start_idx + event_length

            ak_event = ak.unzip(ak_events[event_start_idx:event_end_idx])
            event_id = ak_event[0][0]  # All events in a group have the same ID
            np_event_detids = ak_event[1].to_numpy()
            np_event_rows = ak_event[2].to_numpy().astype(np.uint16)
            np_event_cols = ak_event[3].to_numpy().astype(np.uint16)
            np_event_adcs = ak_event[4].to_numpy().astype(np.uint8)

            module_hits_all = []

            # Group rows by detID (pixel detector module)
            detid_lengths = ak.run_lengths(ak_event[1])
            detid_start_idx = 0
            for detid_length in detid_lengths:
                detid = np_event_detids[detid_start_idx]
                detid_end_idx = detid_start_idx + detid_length

                # Create PixelModuleHits for each module
                module = detector[detid]
                module_hits = PixelModuleHits(detector=detector,
                                              module=module,
                                              rows=np_event_rows[detid_start_idx:detid_end_idx],
                                              cols=np_event_cols[detid_start_idx:detid_end_idx],
                                              adcs=np_event_adcs[detid_start_idx:detid_end_idx])
                module_hits_all.append(module_hits)
                detid_start_idx = detid_end_idx

            event_start_idx = event_end_idx
            pixel_digi = PixelDigiEvent(id_=event_id,
                                        hits=module_hits_all,
                                        detector=detector)
            pixel_digis.append(pixel_digi)

        return pixel_digis
