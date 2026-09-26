import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def physical_structure_target(volume, spacing, anchors, frame='voxel_xyz', sigma_mm=None, rho_mm=None, energy_threshold=0.):
    spacing = np.asarray(spacing, dtype=float)
    sigma_mm = float(np.max(spacing)) if sigma_mm is None else float(sigma_mm)
    rho_mm = 2 * sigma_mm if rho_mm is None else float(rho_mm)
    if sigma_mm <= 0 or rho_mm <= 0 or np.any(spacing <= 0):
        raise ValueError('Structure smoothing scales and spacing must be positive mm values')
    smoothed = gaussian_filter(volume.astype(np.float32), sigma_mm / spacing)
    gradient = np.stack(np.gradient(smoothed, *spacing), -1)
    if frame == 'scanner':
        gradient *= np.array([1., 1., -1.])
    elif frame != 'voxel_xyz':
        raise ValueError(f'Unsupported structure frame: {frame}')
    structure = np.empty((*volume.shape, 3, 3), dtype=np.float32)
    for row in range(3):
        for column in range(3):
            structure[..., row, column] = gaussian_filter(
                gradient[..., row] * gradient[..., column], rho_mm / spacing)
    coordinates = (anchors * (np.asarray(volume.shape) - 1)).T
    sampled = np.stack([map_coordinates(structure[..., row, column], coordinates, order=1)
                        for row in range(3) for column in range(3)], -1).reshape(-1, 3, 3)
    eigenvalues = np.linalg.eigvalsh(sampled)
    trace = np.trace(sampled, axis1=-2, axis2=-1)
    confidence = (eigenvalues[:, -1] - eigenvalues[:, -2]) / (trace + 1e-8)
    confidence *= trace / (trace + 1e-5)
    confidence = np.where(trace >= energy_threshold, confidence, 0.).clip(0, 1)
    return sampled / (trace[:, None, None] + 1e-8), confidence.astype(np.float32)
