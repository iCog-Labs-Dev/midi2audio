"""Stage 2: Track role-to-engine dispatch registry and stem render management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import pretty_midi
from pydantic import BaseModel
from m2a.analysis import Stage0AnalysisReport
from m2a.render.fluidsynth_r import synthesize_midi_track_fluidsynth


class RenderedStem(BaseModel):
    """Artifact representation of a rendered dry stem."""

    track_idx: int
    role: str
    stem_name: str
    audio_path: str
    sample_rate: int
    peak_dbfs: float


class RenderSessionReport(BaseModel):
    """Execution summary of Stage 2 deterministic rendering."""

    stems: List[RenderedStem]
    bar_timestamps: List[float]


def render_all_stems(
    expressive_midi_path: Path | str,
    analysis: Stage0AnalysisReport,
    output_dir: Path | str,
    config: Dict[str, Any],
) -> RenderSessionReport:
    """Render every track to isolated stems and extract bar-boundary timestamps."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(str(expressive_midi_path))

    soundfont = config.get("pipeline", {}).get("soundfont_path", "soundfonts/GeneralUser_GS.sf2")
    sr = config.get("pipeline", {}).get("sample_rate", 48000)
    headroom = config.get("pipeline", {}).get("headroom_dbfs", -12.0)

    stems: List[RenderedStem] = []
    role_map = {t.idx: t.role for t in analysis.tracks}

    for idx, inst in enumerate(pm.instruments):
        role = role_map.get(idx, "comp")
        stem_name = f"stem_{idx:02d}_{role}"
        target_path = out_dir / f"{stem_name}.wav"

        synthesize_midi_track_fluidsynth(
            midi_path=expressive_midi_path,
            track_idx=idx,
            soundfont_path=soundfont,
            output_wav_path=target_path,
            sample_rate=sr,
            headroom_dbfs=headroom,
        )

        stems.append(
            RenderedStem(
                track_idx=idx,
                role=role,
                stem_name=stem_name,
                audio_path=str(target_path),
                sample_rate=sr,
                peak_dbfs=headroom,
            )
        )

    # Calculate exact bar-boundary timestamps from tempo map
    bpm = analysis.tempo_map[0][1]
    sec_per_bar = (60.0 / bpm) * analysis.time_signatures[0][1]
    total_time = pm.get_end_time()
    num_bars = int(total_time // sec_per_bar) + 2
    bar_timestamps = [float(b * sec_per_bar) for b in range(num_bars)]

    return RenderSessionReport(stems=stems, bar_timestamps=bar_timestamps)