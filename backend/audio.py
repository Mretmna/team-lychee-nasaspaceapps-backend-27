"""Render continuous image voices with STRAUSS; export bounded PCM with optional playback normalization."""
import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'jukebox-matplotlib'))

import numpy as np
from scipy.io import wavfile
from strauss.generator import Synthesizer
from strauss.score import Score
from strauss.sonification import Sonification
from strauss.sources import Objects
from .imaging import NOTES, COLUMNS

SAMPLE_RATE = 22050


# Frequency ratios are represented by STRAUSS oscillator detune percentages.
# Every voice's oscillator levels sum to one; color blending also sums to one.
# STRAUSS additive synthesis approximations, not recordings of real instruments.
# The original profile preserves the existing output. Each voice has at most
# three partials because the current generator uses three oscillator slots.
SOUND_TEMPLATES = {
    'original': {
        'label': 'Original colors',
        'description': 'The original rounded red bass, soft green tone, and blue bell.',
        'voices': {
            'neutral': {'instrument': 'neutral sine', 'octave': 0, 'partials': [(1., 1.)], 'attack': .02, 'sustain': 1.},
            'red': {'instrument': 'rounded bass', 'octave': -1, 'partials': [(1., .85), (2., .15)], 'attack': .02, 'sustain': 1.},
            'green': {'instrument': 'soft midrange', 'octave': 0, 'partials': [(1., .8), (2., .15), (3., .05)], 'attack': .02, 'sustain': 1.},
            'blue': {'instrument': 'bell-like tone', 'octave': 1, 'partials': [(1., .55), (2.76, .3), (5.4, .15)], 'attack': .02, 'sustain': 1.},
        },
    },
    'orchestral': {
        'label': 'Choir · Saxophone · Violin',
        'description': 'Synthesized approximations: red choir, green saxophone, blue violin.',
        'voices': {
            'neutral': {'instrument': 'soft organ', 'octave': 0, 'partials': [(1., .8), (2., .2)], 'attack': .08, 'sustain': 1.},
            'red': {'instrument': 'choir-like', 'octave': 0, 'partials': [(1., .55), (2., .3), (3., .15)], 'attack': .14, 'sustain': 1.},
            'green': {'instrument': 'saxophone-like', 'octave': 0, 'partials': [(1., .35), (2., .4), (3., .25)], 'attack': .04, 'sustain': 1.},
            'blue': {'instrument': 'violin-like', 'octave': 1, 'partials': [(1., .35), (2., .3), (3., .35)], 'attack': .09, 'sustain': 1.},
        },
    },
    'medieval': {
        'label': 'Medieval',
        'description': 'Red low horn, green lute-like tone, blue reed-like tone.',
        'voices': {
            'neutral': {'instrument': 'drone', 'octave': -1, 'partials': [(1., .85), (2., .15)], 'attack': .1, 'sustain': 1.},
            'red': {'instrument': 'low horn-like', 'octave': -1, 'partials': [(1., .7), (2., .2), (3., .1)], 'attack': .08, 'sustain': 1.},
            'green': {'instrument': 'lute-like', 'octave': 0, 'partials': [(1., .6), (2., .3), (3., .1)], 'attack': .015, 'sustain': 1.},
            'blue': {'instrument': 'reed-like', 'octave': 1, 'partials': [(1., .45), (2., .35), (3., .2)], 'attack': .06, 'sustain': 1.},
        },
    },
    'modern': {
        'label': 'Modern',
        'description': 'Red synth bass, green electric pad, blue glassy lead.',
        'voices': {
            'neutral': {'instrument': 'ambient pad', 'octave': 0, 'partials': [(1., .65), (2.01, .25), (3.02, .1)], 'attack': .12, 'sustain': 1.},
            'red': {'instrument': 'synth bass', 'octave': -1, 'partials': [(1., .5), (2., .35), (3., .15)], 'attack': .025, 'sustain': 1.},
            'green': {'instrument': 'electric pad', 'octave': 0, 'partials': [(1., .5), (1.01, .35), (2., .15)], 'attack': .12, 'sustain': 1.},
            'blue': {'instrument': 'glassy lead', 'octave': 1, 'partials': [(1., .45), (2.5, .3), (4.2, .25)], 'attack': .03, 'sustain': 1.},
        },
    },
    'heavenly': {
        'label': 'Heavenly',
        'description': 'Warm angelic choir, floating strings, and crystal chimes with a soft echo.',
        'voices': {
            'neutral': {'instrument': 'halo choir', 'octave': 0, 'partials': [(1., .6), (1.004, .25), (2., .15)], 'attack': .22, 'sustain': 1.},
            'red': {'instrument': 'angelic choir', 'octave': 0, 'partials': [(1., .55), (1.5, .25), (2., .2)], 'attack': .18, 'sustain': 1.},
            'green': {'instrument': 'floating string pad', 'octave': 1, 'partials': [(1., .55), (1.003, .25), (2., .2)], 'attack': .16, 'sustain': 1.},
            'blue': {'instrument': 'crystal chimes', 'octave': 1, 'partials': [(1., .5), (2.01, .3), (3.95, .2)], 'attack': .045, 'sustain': 1.},
        },
    },
}


def template_catalog():
    return [{'id': key, 'label': item['label'], 'description': item['description'],
             'instruments': {voice: preset['instrument'] for voice, preset in item['voices'].items()}}
            for key, item in SOUND_TEMPLATES.items()]


def render_audio(events, duration, normalize=True, template='original'):
    if template not in SOUND_TEMPLATES:
        raise ValueError(f'Unknown sound template: {template}')
    samples = np.zeros(round(duration * SAMPLE_RATE), dtype=np.float64)
    for voice, preset in SOUND_TEMPLATES[template]['voices'].items():
        layer = [event for event in events if event['voice'] == voice]
        if not layer:
            continue
        # One STRAUSS Object per image row. Its volume evolves smoothly as
        # the scan crosses columns; oscillators keep running throughout.
        volumes = np.zeros((12, COLUMNS + 1), dtype=np.float64)
        for event in layer:
            volumes[event['row'], event['column']] = event['volume']
        volumes[:, -1] = volumes[:, -2]
        active_rows = np.flatnonzero(np.any(volumes > 0, axis=1))
        keys = ['pitch', 'volume', 'time_evo']
        sources = Objects(keys)
        sources.fromdict({
            'pitch': [(11 - row + .5) / 12 for row in active_rows],
            'volume': [volumes[row] for row in active_rows],
            'time_evo': [np.linspace(0, 1, COLUMNS + 1) for _ in active_rows],
        })
        sources.apply_mapping_functions(map_lims={key: (0, 1) for key in keys})
        generator = Synthesizer(samprate=SAMPLE_RATE)
        oscillators = {f'osc{i + 1}': {'form': 'sine', 'level': 0., 'detune': 0., 'phase': 0.} for i in range(3)}
        for i, (ratio, amplitude) in enumerate(preset['partials']):
            oscillators[f'osc{i + 1}'].update(level=amplitude, detune=(ratio - 1) * 100)
        generator.modify_preset({
            'oscillators': oscillators,
            'note_length': duration,
            'volume_envelope': {'A': preset['attack'],
                                'D': .02, 'S': preset['sustain'],
                                'R': .02},
        }, clear_oscs=False)
        notes = [note[:-1] + str(int(note[-1]) + preset['octave']) for note in NOTES]
        score = Score([notes], duration, pitch_binning='uniform')
        sonification = Sonification(score, sources, generator, audio_setup='mono', samprate=SAMPLE_RATE)
        sonification.render()
        samples += np.asarray(sonification.out_channels['0'].values, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise RuntimeError('STRAUSS produced nonfinite audio')
    if template == 'heavenly':
        # Three quiet delays broaden the sustained voices without using samples
        # or changing the scan timing. The weights sum below one.
        dry = samples
        samples = dry * .7
        for seconds, level in ((.17, .14), (.34, .09), (.53, .05)):
            offset = round(seconds * SAMPLE_RATE)
            if offset < len(samples):
                samples[offset:] += dry[:-offset] * level
    # Fixed gain preserves brightness comparisons and keeps black images silent.
    samples = samples * .7
    if np.max(np.abs(samples), initial=0) > 1:
        raise RuntimeError('STRAUSS output exceeded the fixed audio headroom')
    peak = np.max(np.abs(samples), initial=0)
    if normalize and peak > 0:
        samples *= .8 / peak
    result = io.BytesIO()
    wavfile.write(result, SAMPLE_RATE, np.round(samples * 32767).astype(np.int16))
    return result.getvalue()
