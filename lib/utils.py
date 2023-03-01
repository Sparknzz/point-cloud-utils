import numpy as np

def get_bfcxcy(Q):
    b = 1.0/Q[3,2]
    f = Q[2,3]
    cx = -Q[0,3]
    cy = -Q[1,3]
    return b,f,cx,cy

def fetch_coordinate_values(image, xs, ys):
    # opencv remap or grid sample
    import torch
    import torch.nn.functional as F

    height, width = image.shape[:2]
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
    xs = ((xs / (width - 1)) - 0.5) * 2
    ys = ((ys / (height - 1)) - 0.5) * 2
    
    xys = torch.stack([xs.unsqueeze(1), ys.unsqueeze(1)], 2).unsqueeze(0)
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
    rgb = F.grid_sample(image, xys, padding_mode='zeros', align_corners=True)
    rgb = rgb.squeeze(0).permute(1,2,0).numpy().reshape(-1,3)
    rgb = np.ascontiguousarray(rgb)
    
    return rgb


def get_corr_voxel(point, voxels, voxel_size, min_bound, max_bound, min_num_pts=4):
    from scipy import stats

    bin_ = np.arange(min_bound, max_bound, voxel_size)
    counts, edges, binnumbers = stats.binned_statistic_dd(
        point,
        values=None,
        statistic="count",
        bins=bin_,
        range=None,
        expand_binnumbers=False
    )

    ub = np.unique(binnumbers)
    pts_ds = []
    scores_ds = []
    
    for b in ub:
        if len(np.where(binnumbers == b)[0]) >= min_num_pts:
            pts_ds.append(pts[np.where(binnumbers == b)[0]].mean(axis=0))
            scores_ds.append(scores[np.where(binnumbers == b)[0]].mean())
    
    pts_ds = np.vstack(pts_ds)

    scores_ds = np.vstack(scores_ds).reshape(-1)
    
    return voxel, id_