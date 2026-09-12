H3_PHASE_2_NOISE_LEVEL_START_DEFAULT = 0.9035
H3_AUDIO_REFINEMENT_SETTING = "audio_refinement"
H3_AUDIO_REFINEMENT_DENOISE = 0.5
H3_AUDIO_REFINEMENT_STEPS = 6

H3_MASK_MODE_SETTING = "h3_mask_mode"
H3_MASK_MODE_SHARED_TIMESTEP = "shared_timestep"
H3_MASK_MODE_GROUPED_ROWS = "grouped_rows"
H3_MASK_MODE_DEFAULT = H3_MASK_MODE_GROUPED_ROWS
H3_MASK_MODES = (H3_MASK_MODE_SHARED_TIMESTEP, H3_MASK_MODE_GROUPED_ROWS)

# Which frame of the still-image packet to keep. H3 is a video model, so a prompt that
# asks for a change ("she turns toward the candle") plays that change out ACROSS the
# packet: frame 0 still looks like the reference, the last frame is where the
# instruction has actually happened. "first" is maximum fidelity, "last" is an edit.
H3_STILL_FRAME_SETTING = "h3_still_frame"
H3_STILL_FRAME_FIRST = "first"
H3_STILL_FRAME_LAST = "last"
H3_STILL_FRAME_DEFAULT = H3_STILL_FRAME_LAST
H3_STILL_FRAMES = (H3_STILL_FRAME_FIRST, H3_STILL_FRAME_LAST)


def h3_still_frame_index(custom_settings):
    """Return the index into the decoded packet for still-image output."""
    mode = H3_STILL_FRAME_DEFAULT if custom_settings is None else custom_settings.get(H3_STILL_FRAME_SETTING, H3_STILL_FRAME_DEFAULT)
    if mode not in H3_STILL_FRAMES:
        raise ValueError(f"Unsupported MiniMax H3 still image frame {mode!r}")
    return 0 if mode == H3_STILL_FRAME_FIRST else -1


def h3_grouped_masking_enabled(custom_settings):
    mode = H3_MASK_MODE_DEFAULT if custom_settings is None else custom_settings.get(H3_MASK_MODE_SETTING, H3_MASK_MODE_DEFAULT)
    if mode not in H3_MASK_MODES:
        raise ValueError(f"Unsupported MiniMax H3 mask denoising mode {mode!r}")
    return mode == H3_MASK_MODE_GROUPED_ROWS
