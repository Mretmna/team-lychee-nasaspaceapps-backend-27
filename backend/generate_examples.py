"""Run: python -m backend.generate_examples (writes deterministic audio demos)."""
import io
import json
from pathlib import Path
import wave

import numpy as np
from .audio import render_audio, SOUND_TEMPLATES
from .app import SOURCE, ASSETS
from .imaging import decode_image, image_events


def main():
    destination = Path(__file__).parent / 'examples'
    destination.mkdir(exist_ok=True)
    rgb, _ = decode_image((ASSETS / 'nasa-ocean.jpg').read_bytes())
    events, mapping = image_events(rgb, 12)
    payload = render_audio(events, 12)
    (destination / 'nasa-ocean.wav').write_bytes(payload)
    for template in SOUND_TEMPLATES:
        (destination / f'nasa-{template}.wav').write_bytes(render_audio(events, 12, template=template))
    with wave.open(io.BytesIO(payload)) as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        record = {'durationSeconds': len(samples) / wav.getframerate(), 'sampleRate': wav.getframerate(),
                  'channels': wav.getnchannels(), 'bytes': len(payload),
                  'peakAbsolutePcm': int(np.max(np.abs(samples))),
                  'rmsPcm': float(np.sqrt(np.mean(samples.astype(float) ** 2))),
                  'sourceImageSha256': SOURCE['sha256'], 'mapping': mapping,
                  'normalization': 'peak-0.8'}
    (destination / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
    # One horizontal strip, equal luminance and geometry: isolate hue/voice.
    for name, color in [('red', [.05 / .2126, 0, 0]), ('green', [0, .05 / .7152, 0]), ('blue', [0, 0, .05 / .0722])]:
        grid = np.zeros((12, 48, 3))
        grid[6, :] = color
        events, _ = image_events(grid, 3)
        (destination / f'{name}.wav').write_bytes(render_audio(events, 3))
    print('Generated five NASA template examples and three equal-brightness color demonstrations.')


if __name__ == '__main__':
    main()
