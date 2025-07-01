import numpy as np
import torch
import h5py
import uproot
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_pixel_digis(root_path, branch="analyzer/digiTree"):
    file = uproot.open(root_path)
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
        events.append((int(event_id), det_dict))

    return events


def read_pointcloud_and_adc(h5_path, sample_frac=None):
    points_list = []
    adcs_list = []
    with h5py.File(h5_path, 'r') as f:
        event_keys = sorted(f.keys(), key=lambda k: int(k.split('_')[-1]))
        for key in event_keys:
            data = f[key]['pixel_digis'][:]
            xyz = np.stack([data['x'], data['y'], data['z']], axis=-1)
            adc = np.array(data['adc'])
            
            if sample_frac is not None:
                idx = np.random.choice(len(xyz), int(len(xyz) * sample_frac), replace=False)
                xyz = xyz[idx]
                adc = adc[idx]
                
            xyz = torch.tensor(xyz, dtype=torch.float32).to(device)
            adc = torch.tensor(adc, dtype=torch.float32).to(device)

            points_list.append(xyz)
            adcs_list.append(adc)
    return points_list, adcs_list
