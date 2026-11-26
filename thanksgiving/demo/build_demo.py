#!/usr/bin/env python3
"""Thanksgiving hymn medley demo — pure-Python piano render.

Melodies (all public domain, verified):
  1. "We Gather Together" — tune KREMSER, 16th-c. Dutch melody (D major, 3/4).
     Four voices transcribed from the Open Hymnal Project ABC
     (openhymnal.org, "We Gather Together", public domain).
  2. "Come, Ye Thankful People, Come" — tune ST. GEORGE'S WINDSOR,
     George J. Elvey 1858; words Henry Alford 1844 (G major, 4/4).
     Melody from piano.org's bar-by-bar transcription.
  3. "Now Thank We All Our God" — tune NUN DANKET, Johann Crüger 1647;
     words Martin Rinkart 1636 (transposed here to G major for the coda).

Render: additive piano-ish synth (numpy) -> WAV -> ffmpeg MP3.
Target: ~85 seconds, gentle, uncluttered.
"""
import os
import subprocess
import wave

import numpy as np

SR = 44100
BEAT = 0.6  # 100 BPM
OUT_DIR = "/home/hatch/workspace/greetings/thanksgiving/demo"
os.makedirs(OUT_DIR, exist_ok=True)

NOTE_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
           "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
           "A#": 10, "Bb": 10, "B": 11}


def midi_of(name):
    # e.g. 'A4', 'F#4', 'C#5', 'Bb4'
    if "#" in name or "b" in name[1:]:
        pc_name = name[:2]
        octave = int(name[2:])
    else:
        pc_name = name[0]
        octave = int(name[1:])
    return 12 * (octave + 1) + NOTE_PC[pc_name]


def seq_total(seq):
    return sum(b for _, b in seq)


# ---------------------------------------------------------------- KREMSER (D major, 3/4)
# Four voices from the Open Hymnal Project ABC (L:1/4, Q:1/4=100).
# Final note of each voice held 3 beats instead of 2 (fermata feel).
KREM_S = [("A4", 1), ("A4", 1.5), ("B4", 0.5), ("A4", 1),
          ("F#4", 1), ("G4", 1), ("A4", 1),
          ("G4", 1.5), ("F#4", 0.5), ("E4", 1), ("F#4", 1), ("D4", 1),
          ("A4", 1), ("A4", 1.5), ("B4", 0.5), ("C#5", 1),
          ("D5", 1.5), ("E5", 0.5), ("C#5", 1),
          ("B4", 1.5), ("A4", 0.5), ("B4", 1), ("A4", 2),
          ("A4", 1), ("A4", 1.5), ("B4", 0.5), ("C#5", 1),
          ("D5", 1), ("A4", 1), ("A4", 1),
          ("A4", 1.5), ("B4", 0.5), ("G4", 0.5), ("A4", 0.5),
          ("F#4", 1), ("D4", 1),
          ("D4", 1), ("G4", 1.5), ("A4", 0.5), ("B4", 0.5), ("G4", 0.5),
          ("A4", 1.5), ("G4", 0.5), ("F#4", 0.5), ("F#4", 0.5),
          ("G4", 1), ("E4", 1.5), ("D4", 0.5), ("D4", 3)]
KREM_A = [("D4", 1), ("D4", 1.5), ("D4", 0.5), ("D4", 1),
          ("D4", 1), ("D4", 1), ("D4", 1),
          ("D4", 1.5), ("D4", 0.5), ("C#4", 1), ("D4", 1), ("D4", 1),
          ("F#4", 1), ("E4", 1.5), ("E4", 0.5), ("E4", 1),
          ("F#4", 1), ("E4", 1), ("E4", 1),
          ("F#4", 1.5), ("F#4", 0.5), ("G#4", 1), ("E4", 2),
          ("E4", 1), ("E4", 1.5), ("E4", 0.5), ("E4", 1),
          ("D4", 1), ("D4", 1), ("D4", 1),
          ("D4", 2), ("C#4", 0.5), ("C#4", 0.5), ("D4", 1), ("D4", 1),
          ("D4", 1), ("D4", 1.5), ("D4", 0.5), ("D4", 0.5), ("D4", 0.5),
          ("E4", 2), ("D4", 0.5), ("D4", 0.5),
          ("D4", 1), ("C#4", 1.5), ("D4", 0.5), ("D4", 3)]
KREM_T = [("F#3", 1), ("F#3", 1.5), ("G3", 0.5), ("F#3", 1),
          ("A3", 1), ("A3", 1), ("A3", 1),
          ("B3", 1.5), ("B3", 0.5), ("C4", 1), ("A3", 1), ("F#3", 1),
          ("A3", 1), ("A3", 1.5), ("A3", 0.5), ("A3", 1),
          ("A3", 1), ("G#3", 1), ("A3", 1),
          ("A3", 1.5), ("D4", 0.5), ("D4", 1), ("C4", 2),
          ("C4", 1), ("C4", 1.5), ("B3", 0.5), ("A3", 1),
          ("A3", 1), ("A3", 1), ("A3", 1),
          ("B3", 2), ("A3", 0.5), ("A3", 0.5), ("A3", 1), ("F#3", 1),
          ("F#3", 1), ("G3", 1.5), ("F#3", 0.5), ("G3", 0.5), ("G3", 0.5),
          ("E3", 1), ("A3", 1), ("A3", 0.5), ("A3", 0.5),
          ("G3", 1), ("G3", 1.5), ("F#3", 0.5), ("F#3", 3)]
KREM_B = [("D3", 1), ("D3", 1.5), ("D3", 0.5), ("D3", 1),
          ("D3", 1), ("E3", 1), ("F#3", 1),
          ("E3", 1.5), ("E3", 0.5), ("A3", 1), ("D3", 1), ("D3", 1),
          ("D3", 1), ("C3", 1.5), ("C3", 0.5), ("C3", 1),
          ("B2", 1), ("E3", 1), ("A2", 1),
          ("D3", 1.5), ("B2", 0.5), ("E3", 1), ("A2", 2),
          ("A3", 1), ("G3", 1.5), ("G3", 0.5), ("G3", 1),
          ("F#3", 1), ("F#3", 1), ("F#3", 1),
          ("E3", 2), ("A3", 0.5), ("A3", 0.5), ("D3", 1), ("D3", 1),
          ("D3", 1), ("B2", 1.5), ("A2", 0.5), ("G2", 0.5), ("B2", 0.5),
          ("C3", 2), ("D3", 0.5), ("D3", 0.5),
          ("E3", 1), ("A2", 1.5), ("D3", 0.5), ("D3", 3)]

# ------------------------------------------------- ST. GEORGE'S WINDSOR (G major, 4/4)
# Melody from piano.org bar-by-bar transcription (bars 1-18).
SGW = [("B4", 1), ("B4", 1), ("D5", 1), ("B4", 1), ("G4", 1), ("A4", 1), ("B4", 2),
       ("B4", 1), ("B4", 1), ("D5", 1), ("B4", 1), ("G4", 1), ("A4", 1), ("B4", 2),
       ("B4", 1), ("B4", 1), ("C5", 1), ("C5", 1), ("A4", 1), ("A4", 1), ("B4", 2),
       ("B4", 1), ("C#5", 1), ("D5", 1), ("G4", 1), ("F#4", 1), ("E4", 1), ("D4", 2),
       ("F#4", 1), ("F#4", 1), ("A4", 1), ("F#4", 1), ("G4", 1), ("A4", 1), ("B4", 2),
       ("B4", 1), ("B4", 1), ("D5", 1), ("B4", 1), ("C5", 1), ("D5", 1), ("E5", 2),
       ("E5", 1), ("E5", 1), ("C5", 1), ("A4", 1), ("D5", 1), ("D5", 1), ("B4", 2),
       ("C5", 1), ("E5", 1), ("D5", 1), ("G4", 1), ("B4", 1), ("A4", 1), ("G4", 2),
       ("G4", 4), ("G4", 4)]
# One chord per bar (18 bars): bass root + triad pad.
SGW_CHORDS = ["G", "G", "G", "G", "C", "Am", "D", "D",
              "D", "G", "G", "C", "C", "G", "G", "G", "G", "G"]
CHORD_TONES = {"G": (55, 59, 62), "C": (60, 64, 67),
               "Am": (57, 60, 64), "D": (62, 66, 69)}
CHORD_ROOT = {"G": 43, "C": 48, "Am": 45, "D": 50}

# ------------------------------------------------- NUN DANKET coda, transposed to G major
# Original (K:F): pickup c | c c d d | c A | Bb A G F | E F  (first phrase only)
NUN = [("D5", 0.5),
       ("D5", 1.5), ("D5", 0.5), ("E5", 1), ("E5", 1),
       ("D5", 3), ("B4", 1),
       ("C5", 1), ("B4", 1), ("A4", 1), ("G4", 1),
       ("F#4", 2), ("G4", 2)]
NUN_CHORDS = ["G", "G", "C", "G"]  # one per bar after the pickup


# ---------------------------------------------------------------- synth
def note_wave(midi, dur, vel):
    """Soft piano-ish tone: harmonics with decaying envelopes."""
    f = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    n = int((dur + 2.5) * SR)
    t = np.arange(n, dtype=np.float64) / SR
    y = np.zeros(n)
    for mult, amp in ((1.0, 1.0), (2.0, 0.5), (3.0, 0.28),
                      (4.0, 0.16), (5.0, 0.09), (6.0, 0.05)):
        tau = 1.7 / (1.0 + 0.4 * (mult - 1.0))
        y += amp * np.exp(-t / tau) * np.sin(2 * np.pi * f * mult * t)
    atk = int(0.008 * SR)
    y[:atk] *= np.linspace(0, 1, atk)
    return vel * y


def add_note(mix, start_beat, midi, beats, vel):
    start = int(start_beat * BEAT * SR)
    w = note_wave(midi, beats * BEAT, vel)
    end = start + len(w)
    if end > len(mix):
        mix.resize(end, refcheck=False)
    mix[start:end] += w


def schedule_seq(mix, start_beat, seq, vel):
    t = start_beat
    for name, beats in seq:
        add_note(mix, t, midi_of(name), beats, vel)
        t += beats
    return t


def main():
    # Validate voice totals.
    for label, seq in (("KREM_S", KREM_S), ("KREM_A", KREM_A),
                       ("KREM_T", KREM_T), ("KREM_B", KREM_B)):
        print(label, "total beats:", seq_total(seq))
    assert len({seq_total(s) for s in (KREM_S, KREM_A, KREM_T, KREM_B)}) == 1
    print("SGW total beats:", seq_total(SGW))
    print("NUN total beats:", seq_total(NUN))

    mix = np.zeros(int(4 * SR))
    t = 0.0
    # 1. Kremser, four voices (melody louder, inner voices soft).
    for seq, vel in ((KREM_S, 0.8), (KREM_A, 0.42),
                     (KREM_T, 0.42), (KREM_B, 0.5)):
        schedule_seq(mix, t, seq, vel)
    t += seq_total(KREM_S) + 1.0  # one-beat breath
    # 2. St. George's Windsor, melody + chord pads.
    bar_starts = []
    bt = t
    for i in range(18):
        bar_starts.append(bt)
        bt += 4
    schedule_seq(mix, t, SGW, 0.8)
    for bar, chord in enumerate(SGW_CHORDS):
        bs = bar_starts[bar]
        add_note(mix, bs, CHORD_ROOT[chord], 2.0, 0.45)
        for tone in CHORD_TONES[chord]:
            add_note(mix, bs, tone, 3.5, 0.30)
    t += seq_total(SGW) + 1.0
    # 3. Nun danket coda (pickup, then 4 bars).
    schedule_seq(mix, t, NUN, 0.8)
    for i, chord in enumerate(NUN_CHORDS):
        bs = t + 0.5 + i * 4
        add_note(mix, bs, CHORD_ROOT[chord], 2.0, 0.45)
        for tone in CHORD_TONES[chord]:
            add_note(mix, bs, tone, 3.5, 0.30)
    t += seq_total(NUN)
    total_dur = t + 2.0  # decay tail

    # Trim to total length, normalize, gentle stereo echo for air.
    n = int(total_dur * SR)
    mix = mix[:n]
    peak = np.max(np.abs(mix))
    mix = mix / peak * 0.89
    print("duration s: %.1f  peak: %.3f  rms: %.3f"
          % (len(mix) / SR, np.max(np.abs(mix)),
             float(np.sqrt(np.mean(mix ** 2)))))

    def delay(x, sec, gain):
        d = int(sec * SR)
        y = np.zeros_like(x)
        y[d:] = x[:-d] * gain
        return y

    left = mix + delay(mix, 0.13, 0.16) + delay(mix, 0.27, 0.08)
    right = mix + delay(mix, 0.17, 0.16) + delay(mix, 0.31, 0.08)
    stereo = np.stack([left, right], axis=1)
    stereo = stereo / np.max(np.abs(stereo)) * 0.89

    wav_path = os.path.join(OUT_DIR, "thanksgiving-melodies-demo.wav")
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    mp3_path = os.path.join(OUT_DIR, "thanksgiving-melodies-demo.mp3")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path,
                    "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path],
                   check=True)
    print("wrote", mp3_path, os.path.getsize(mp3_path) // 1024, "KB")


if __name__ == "__main__":
    main()
