"""Fixed external coefficient only. Raw MMD, kernel and diagnostics are unchanged."""
MMD_WEIGHT=5.0
def combine(point_mse,raw_mmd):
    return point_mse+MMD_WEIGHT*raw_mmd
