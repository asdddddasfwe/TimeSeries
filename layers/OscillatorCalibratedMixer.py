"""Per-variable gains for the homogeneous and forced oscillator responses."""
import torch

from layers.OscillatorMixer import parallel_oscillator_scan


def calibrated_seasonal_forecast(encoder, x, history_gain, drive_gain, intervention="normal", history_stats=None, history_trend=None, encoded_history=None, scan_function=None):
    """Return [B,H_i,C] using the existing history encoder and decoder.

    At the STATE level, scaling initial state by alpha and future drive by beta
    equals alpha*history_response+beta*drive_response. Output superposition
    additionally requires a linear readout; nonlinear variants activate the
    already combined calibrated state.
    Historical drive is NOT scaled: calibration acts at the forecast boundary.
    This preserves two scans (history, future), without a third component scan.
    Gains are [C], broadcast across batch, time, modes and all model scales.
    """
    if x.ndim != 3 or x.shape[1] != encoder.length:
        raise ValueError("Expected seasonal input [B,T_i,C]")
    batch, length, channels = x.shape
    if history_gain.shape != (channels,) or drive_gain.shape != (channels,):
        raise ValueError("Expected one history/drive gain per input variable")
    if intervention not in ("normal", "zero_initial", "zero_drive", "no_memory"):
        raise ValueError("Unknown calibrated oscillator intervention")
    flat = x.transpose(1, 2).reshape(batch * channels, length)
    if encoded_history is None:
        history = (encoder.encode_history(flat) if history_trend is None else
                   encoder.encode_history(flat, history_trend.transpose(1, 2).reshape(batch * channels, length)))
    else:
        # Cross-scale mixing supplies already encoded [B*C,T_i,D] history.
        history = encoded_history
        if history.shape != (batch * channels, length, encoder.value_embedding.out_features):
            raise ValueError("Encoded history must have shape [B*C,T_i,D]")
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history)
    drive = (decoder.future_drive(future, history) if history_stats is None else
             decoder.future_drive(future, history, history_stats.reshape(batch * channels, -1)))
    drive = drive.reshape(batch * channels, encoder.horizon, decoder.modes, 2)
    # Flattened order is b0c0,b0c1,...,b1c0,b1c1,..., matching x above.
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    if intervention == "zero_initial":
        initial = torch.zeros_like(initial)
    elif intervention == "zero_drive":
        drive = torch.zeros_like(drive)
    elif intervention == "no_memory":
        transition = torch.zeros_like(transition)
    scan = parallel_oscillator_scan if scan_function is None else scan_function
    states = scan(
        transition[:, None].expand(-1, encoder.horizon, -1, -1), drive, initial)
    output = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype)).squeeze(-1)
    return output.reshape(batch, channels, encoder.horizon).transpose(1, 2)
