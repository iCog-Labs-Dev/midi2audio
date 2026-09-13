import numpy as np

# 1. Stage 1 Math
orig_start = 0.5000
bpm = 120.0
sec_per_16th = (60.0 / bpm) / 4.0  # 0.125 s
pos_16 = int(round(orig_start / sec_per_16th)) % 16  # 4
offset_ms = 7.5
expr_start = orig_start + (offset_ms / 1000.0)  # 0.5075 s

# 2. Stage 2 Math
sr = 48000
sample_idx = int(expr_start * sr)  # 24360

# 3. Stage 5 Scoring Check
ai_candidate_start = 0.5400  # Drifted +32.5 ms
drum_tolerance = 0.025      # 25 ms
deviation = abs(ai_candidate_start - expr_start)
passed_floor = deviation <= drum_tolerance

print(f"Metrical Slot      : {pos_16} (Beat 2 Downbeat)")
print(f"Expressive Onset   : {expr_start:.4f} s")
print(f"PCM Sample Index   : {sample_idx} @ 48 kHz")
print(f"AI Deviation       : {deviation * 1000:.1f} ms")
print(f"Passed Drum Floor? : {passed_floor} -> Composite Score = {1.0 if passed_floor else 0.0}")
