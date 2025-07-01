import torch


def chamfer_distance(p1, p2):
    """
    Computes symmetric Chamfer Distance between two point clouds.
    p1, p2: [B, N, 3] tensors
    """
    B, N, _ = p1.size()
    _, M, _ = p2.size()

    # Compute pairwise distance [B, N, M]
    dist = torch.cdist(p1, p2, p=2)  # Euclidean distance

    # For each point in p1, find the nearest in p2
    min_dist_p1 = dist.min(dim=2)[0]  # [B, N]
    min_dist_p2 = dist.min(dim=1)[0]  # [B, M]

    # Average over points
    loss = min_dist_p1.mean(dim=1) + min_dist_p2.mean(dim=1)  # [B]
    return loss.mean()
