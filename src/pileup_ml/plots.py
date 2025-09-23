import numpy as np
import matplotlib.pyplot as plt

from pileup_ml.events import PixelDigiEvent
from pileup_ml.detectors.pixels import PixelModule, RowColMapping


def plot_pixel_module(event: PixelDigiEvent, module: PixelModule):
    hits = event[module]
    plt.imshow(hits.to_image(), cmap='gist_yarg')
