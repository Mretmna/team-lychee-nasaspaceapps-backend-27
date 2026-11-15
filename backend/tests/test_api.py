import io
import wave
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app, render_lock
from backend.imaging import decode_image, image_events, grid_events, radial_grid, circle_grid, ImageError, MAX_BYTES

client = TestClient(app)


@pytest.mark.parametrize('method,url', [('post', '/sonify?duration_seconds=2'), ('get', '/legend/red')])
def test_render_failure_returns_safe_error_and_allows_retry(monkeypatch, method, url):
    from importlib import import_module

    module = import_module('backend.app')

    def fail_render(*args, **kwargs):
        raise RuntimeError('Internal rendering details')

    with monkeypatch.context() as patch:
        patch.setattr(module, 'render_audio', fail_render)
        response = client.request(method, url, headers={'Origin': 'http://localhost:3000'})
        assert response.status_code == 500
        assert response.json() == {'detail': 'The server could not finish processing. Try again.'}
        assert response.headers['access-control-allow-origin'] == 'http://localhost:3000'

    assert pcm(client.request(method, url)).size == 44100


def test_voice_selection_and_timeline():
    buffer = io.BytesIO()
    image = Image.new('RGB', (48, 24), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 24, 24))
    image.save(buffer, format='PNG')
    payload = {'file': ('map.png', buffer.getvalue(), 'image/png')}
    query = '?duration_seconds=2&scan_mode=linear&voices=red&normalize=false'
    samples = pcm(client.post('/sonify'+query, files=payload))
    assert np.max(np.abs(samples[:18000])) > 100
    assert np.max(np.abs(samples[30000:])) == 0
    meta = client.post('/analyze'+query, files=payload).json()['mapping']
    assert meta['enabledVoices'] == ['red']
    assert len(meta['voiceTimeline']) == 49
    assert meta['voiceTimeline'][0]['red'] > 0
    assert meta['voiceTimeline'][40]['blue'] > 0
    assert meta['voiceEventCounts']['blue'] == 0
    assert not pcm(client.post('/sonify?duration_seconds=2&voices=', files=payload)).any()
    assert client.post('/sonify?voices=purple', files=payload).status_code == 422


@pytest.mark.parametrize('template', ['original', 'orchestral', 'medieval', 'modern', 'heavenly'])
def test_legend_voices(template):
    from backend.audio import render_audio
    from backend.imaging import COLUMNS
    outputs = []
    for voice in ('red', 'green', 'blue', 'neutral'):
        response = client.get(f'/legend/{voice}?sound_template={template}')
        samples = pcm(response)
        assert len(samples) == 44100
        assert np.max(np.abs(samples)) > 1000
        events = [{'voice': voice, 'row': 5, 'column': column, 'volume': 1 / 12} for column in range(COLUMNS)]
        assert response.content == render_audio(events, 2, True, template)
        outputs.append(response.content)
    assert len(set(outputs)) == 4


def test_legend_validation_and_busy():
    assert client.get('/legend/purple').status_code == 422
    assert client.get('/legend/red?sound_template=unknown').status_code == 422
    with render_lock:
        assert client.get('/legend/red').status_code == 503


def picture(value=180, fmt='PNG', mode='RGB'):
    buf = io.BytesIO()
    Image.new(mode, (48, 24), value if mode == 'L' else (value,) * len(mode)).save(buf, format=fmt)
    return buf.getvalue()


def pcm(response):
    assert response.status_code == 200, response.text[:500] if response.status_code != 200 else ''
    assert response.headers['content-type'] == 'audio/wav'
    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')


@pytest.mark.parametrize('fmt', ['PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF'])
def test_formats(fmt):
    image, detected = decode_image(picture(fmt=fmt))
    assert image.shape == (24, 48, 3)
    assert detected == fmt


def test_mapping():
    image = np.zeros((12, 48, 3))
    image[0, 0] = 1
    image[-1, -1] = .5
    events, metadata = grid_events(image, 12)
    assert len(events) == 2
    assert events[0]['time'] == 0
    assert events[1]['time'] == 47 / 48
    assert events[0]['pitch'] > events[1]['pitch']
    assert events[0]['volume'] == pytest.approx(2 * events[1]['volume'])
    assert metadata['secondsPerScanPosition'] == .25


def test_alpha_and_gray():
    image, _ = decode_image(picture(0, mode='RGBA'))
    assert np.max(image) == 0
    image, _ = decode_image(picture(100, mode='L'))
    assert np.allclose(image, 100 / 255)


def test_black_returns_explanation_and_not_nasa():
    payload = {'file': ('map.png', picture(0), 'image/png')}
    r = client.post('/sonify?duration_seconds=2', files=payload)
    assert r.status_code == 422
    assert 'No audible image features' in r.json()['detail']
    info = client.post('/analyze', files=payload).json()
    assert info['source']['id'] == 'user-upload'
    assert info['mapping']['eventCount'] == 0


def test_real_strauss_render_changes_with_brightness():
    quiet = pcm(client.post('/sonify?duration_seconds=2&normalize=false', files={'file': ('map.png', picture(70))}))
    loud = pcm(client.post('/sonify?duration_seconds=2&normalize=false', files={'file': ('map.png', picture(210))}))
    assert len(loud) == 44100
    assert np.std(loud) > 2.5 * np.std(quiet) > 0
    assert np.max(np.abs(loud)) < 32767


def test_nasa_default():
    r = client.post('/sonify?duration_seconds=2')
    assert np.std(pcm(r)) > 0
    assert r.headers['x-source-id'] == 'nasa-ocean'
    assert r.headers['x-source-url'] == 'https://svs.gsfc.nasa.gov/3912/'
    metadata = client.post('/analyze').json()
    assert metadata['source']['credits']
    assert metadata['mapping']['eventCount'] > 0
    assert client.get('/sources/nasa-ocean/image').status_code == 200


@pytest.mark.parametrize('payload', [b'', b'not image', b'<svg></svg>'])
def test_invalid_upload_does_not_fall_back(payload):
    r = client.post('/sonify', files={'file': ('fake.jpg', payload, 'image/jpeg')})
    assert r.status_code in (415, 422)
    assert 'detail' in r.json()


@pytest.mark.parametrize('duration', ['0', '-1', '31', 'nan', 'inf'])
def test_bad_duration(duration):
    assert client.post(f'/sonify?duration_seconds={duration}').status_code == 422


def test_oversize():
    with pytest.raises(ImageError) as exc:
        decode_image(b'0' * (MAX_BYTES + 1))
    assert exc.value.status == 413
    r = client.post('/sonify', content=b'0' * (MAX_BYTES + 1024 * 1024 + 1))
    assert r.status_code == 413


def test_animation_and_large_dimensions():
    buf = io.BytesIO()
    Image.new('RGB', (2, 2), 'red').save(buf, format='PNG', save_all=True, append_images=[Image.new('RGB', (2, 2), 'blue')])
    with pytest.raises(ImageError) as exc:
        decode_image(buf.getvalue())
    assert exc.value.status == 415
    buf = io.BytesIO()
    Image.new('L', (8193, 1)).save(buf, format='PNG')
    with pytest.raises(ImageError) as exc:
        decode_image(buf.getvalue())
    assert exc.value.status == 413


def test_busy_and_cors():
    with render_lock:
        r = client.post('/sonify')
        assert r.status_code == 503
        assert r.headers['retry-after'] == '2'
    r = client.options('/sonify', headers={'Origin': 'http://localhost:3000', 'Access-Control-Request-Method': 'POST'})
    assert r.headers['access-control-allow-origin'] == 'http://localhost:3000'


def test_openapi_and_health():
    assert client.get('/health').json()['engine'] == 'STRAUSS'
    assert '/sonify' in client.get('/openapi.json').json()['paths']


def test_exif_orientation():
    image = Image.new('RGB', (8, 4), 'red')
    exif = Image.Exif()
    exif[274] = 6
    buf = io.BytesIO()
    image.save(buf, format='JPEG', exif=exif)
    decoded, _ = decode_image(buf.getvalue())
    assert decoded.shape == (8, 4, 3)


def test_16bit_rejected():
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8), 3000, dtype=np.uint16)).save(buf, format='PNG')
    with pytest.raises(ImageError) as exc:
        decode_image(buf.getvalue())
    assert exc.value.status == 415


def test_missing_fallback_fails_clearly(monkeypatch, tmp_path):
    import backend.app as module
    monkeypatch.setattr(module, 'ASSETS', tmp_path)
    r = client.post('/sonify')
    assert r.status_code == 503


def test_normalized_playback_is_audible_and_bounded():
    for payload in ({}, {'files': {'file': ('dark.png', picture(3), 'image/png')}}):
        r = client.post('/sonify?duration_seconds=2', **payload)
        samples = pcm(r).astype(float) / 32767
        assert .79 <= np.max(np.abs(samples)) <= .81
        assert r.headers['x-audio-normalization'] == 'peak-0.8'
        assert int(r.headers['x-event-count']) > 0


def test_transparent_image_gets_explanation():
    r = client.post('/sonify', files={'file': ('transparent.png', picture(0, mode='RGBA'))})
    assert r.status_code == 422
    assert 'transparent' in r.json()['detail']


def test_download_contract_is_binary_not_json():
    spec = client.get('/openapi.json').json()
    content = spec['paths']['/sonify']['post']['responses']['200']['content']
    assert set(content) == {'audio/wav'}
    assert content['audio/wav']['schema']['format'] == 'binary'
    r = client.post('/sonify?duration_seconds=2')
    assert r.headers['content-disposition'].startswith('attachment;')
    assert int(r.headers['content-length']) == len(r.content) == 88244
    assert r.content[:4] == b'RIFF' and r.content[8:12] == b'WAVE'
    assert len(pcm(r)) == 44100


@pytest.mark.parametrize('rgb,voice', [([1, 0, 0], 'red'), ([0, 1, 0], 'green'), ([0, 0, 1], 'blue')])
def test_primary_hues_select_expected_instrument(rgb, voice):
    image = np.full((12, 48, 3), rgb, dtype=np.float64)
    events, metadata = grid_events(image, 12)
    assert {e['voice'] for e in events} == {voice}
    assert metadata['mappingVersion'] == 'color-voices-v1'


def test_saturation_controls_color_neutral_blend():
    for color, expected in [([1, 0, 0], 1.), ([1, .5, .5], .5), ([1, 1, 1], 0.)]:
        events, _ = grid_events(np.full((12, 48, 3), color, dtype=float), 12)
        first = [e for e in events if e['row'] == 0 and e['column'] == 0]
        colored = sum(e['strength'] for e in first if e['voice'] != 'neutral')
        assert colored == pytest.approx(expected, abs=1e-6)
        assert sum(e['strength'] for e in first) == pytest.approx(1, abs=1e-6)


def test_intermediate_hue_blends_neighbors():
    events, _ = grid_events(np.ones((12, 48, 3)) * [1, 1, 0], 12)
    assert {e['voice'] for e in events} == {'red', 'green'}
    assert all(e['strength'] == pytest.approx(.5, abs=1e-6) for e in events)


def test_equal_brightness_colors_produce_distinct_actual_audio():
    from backend.audio import render_audio
    from scipy.io import wavfile
    outputs, centers = [], []
    for color in ([.05 / .2126, 0, 0], [0, .05 / .7152, 0], [0, 0, .05 / .0722]):
        # Same location, same brightness: only hue and its instrument change.
        image = np.zeros((12, 48, 3))
        image[6, :] = color
        events, _ = grid_events(image, 2)
        payload = render_audio(events, 2)
        rate, values = wavfile.read(io.BytesIO(payload))
        spectrum = abs(np.fft.rfft(values.astype(float))) ** 2
        centers.append(float(np.sum(np.fft.rfftfreq(len(values), 1 / rate) * spectrum) / spectrum.sum()))
        outputs.append(payload)
    assert len({len(p) for p in outputs}) == 1  # Equal WAV size does not imply equal sound.
    assert len(set(outputs)) == 3
    assert centers[0] < centers[1] < centers[2]


def test_vertical_position_changes_rendered_pitch():
    from backend.audio import render_audio
    from scipy.io import wavfile
    peaks = []
    for row in (11, 0):
        image = np.zeros((12, 48, 3))
        image[row, :] = [0, 1, 0]
        events, _ = grid_events(image, 2)
        rate, values = wavfile.read(io.BytesIO(render_audio(events, 2)))
        peaks.append(np.fft.rfftfreq(len(values), 1 / rate)[np.argmax(abs(np.fft.rfft(values)))])
    assert peaks[1] > peaks[0] * 4


def test_sustained_voice_has_no_gaps_between_columns():
    # A uniform colored strip should sound throughout, including column edges.
    from backend.audio import render_audio
    image = np.zeros((12, 48, 3))
    image[6, :] = [0, 1, 0]
    events, _ = grid_events(image, 2)
    rate, samples = __import__('scipy').io.wavfile.read(io.BytesIO(render_audio(events, 2)))
    amplitude = samples.astype(float) / 32768
    window = rate // 100  # 10 ms, shorter than a 2 s / 48 column
    rms = [np.sqrt(np.mean(amplitude[i:i + window] ** 2))
           for i in range(rate // 20, len(amplitude) - rate // 20, window)]
    assert min(rms) > .03
    assert max(rms) / min(rms) < 3


def test_color_transition_remains_audible_through_scan():
    from backend.audio import render_audio
    image = np.zeros((12, 48, 3))
    image[6, :24] = [1, 0, 0]
    image[6, 24:] = [0, 0, 1]
    events, _ = grid_events(image, 2)
    rate, samples = __import__('scipy').io.wavfile.read(io.BytesIO(render_audio(events, 2)))
    for center in (.5, 1., 1.5):
        span = samples[int((center - .02) * rate):int((center + .02) * rate)]
        assert np.sqrt(np.mean(span.astype(float) ** 2)) > 100


def test_clockwise_ray_samples_cardinal_image_regions():
    image = np.zeros((120, 120, 3))
    image[:55, 55:65] = [1, 0, 0]    # north: red
    image[55:65, 65:] = [0, 1, 0]    # east: green
    image[65:, 55:65] = [0, 0, 1]    # south: blue
    image[55:65, :55] = [1, 1, 0]   # west: yellow
    grid = radial_grid(image)
    assert np.allclose(grid[11, 0], [1, 0, 0], atol=.01)
    assert np.allclose(grid[11, 12], [0, 1, 0], atol=.01)
    assert np.allclose(grid[11, 24], [0, 0, 1], atol=.01)
    assert np.allclose(grid[11, 36], [1, 1, 0], atol=.01)
    _, mapping = image_events(image, 12)
    assert mapping['startAngleDegrees'] == 0
    assert mapping['angleDegreesPerSample'] == 7.5
    assert mapping['radialBands'] == 12
    assert mapping['angularDirection'] == 'clockwise'


def test_radial_scan_follows_full_rectangular_edge():
    image = np.ones((60, 120, 3))
    grid = radial_grid(image)
    assert np.allclose(grid, 1)
    assert np.isfinite(grid).all()
    response = client.get('/demo')
    assert response.status_code == 200
    assert 'id="scan-line"' in response.text
    assert 'id="scan-circle"' in response.text
    assert 'value="circle_outward"' in response.text
    assert 'value="circle_inward"' in response.text
    assert 'audio.currentTime' not in response.text  # actual variable is player
    assert 'player.currentTime/player.duration' in response.text



def test_circle_scan_moves_between_center_and_outer_ring():
    image = np.zeros((120, 120, 3))
    center = (119 / 2, 119 / 2)
    yy, xx = np.ogrid[:120, :120]
    radius = np.hypot(xx - center[0], yy - center[1])
    image[radius < 20] = [1, 0, 0]
    image[radius > 40] = [0, 0, 1]
    outward = circle_grid(image)
    inward = circle_grid(image, inward=True)
    assert np.allclose(outward[:, 0], [1, 0, 0], atol=.01)
    assert np.allclose(outward[:, -1], [0, 0, 1], atol=.01)
    assert np.allclose(inward[:, 0], outward[:, -1])
    assert np.allclose(inward[:, -1], outward[:, 0])
    for mode, direction in [('circle_outward', 'center-to-edge'), ('circle_inward', 'edge-to-center')]:
        _, mapping = image_events(image, 2, mode)
        assert mapping['scanMode'] == mode
        assert mapping['radialDirection'] == direction
        assert mapping['angularBands'] == 12


def test_api_scan_modes_produce_distinct_audio_and_metadata():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:48, :48] = [255, 0, 0]
    image[:48, 48:] = [0, 255, 0]
    image[48:, :48] = [0, 0, 255]
    image[48:, 48:] = [255, 255, 0]
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    data = buffer.getvalue()
    outputs = []
    for mode in ('clockwise', 'linear', 'circle_outward', 'circle_inward'):
        files = {'file': ('map.png', data, 'image/png')}
        metadata = client.post(f'/analyze?duration_seconds=2&scan_mode={mode}', files=files).json()['mapping']
        assert metadata['scanMode'] == mode
        response = client.post(f'/sonify?duration_seconds=2&scan_mode={mode}', files=files)
        assert response.headers['x-scan-mode'] == mode
        assert len(pcm(response)) == 44100
        outputs.append(response.content)
    assert len(set(outputs)) == len(outputs)
    assert len({len(output) for output in outputs}) == 1


def test_invalid_scan_mode_is_rejected():
    for path in ('/analyze', '/sonify'):
        response = client.post(f'{path}?scan_mode=diagonal')
        assert response.status_code == 422


def test_sound_template_catalog_and_validation():
    catalog = client.get('/templates').json()
    assert catalog['defaultTemplate'] == 'original'
    assert [item['id'] for item in catalog['templates']] == ['original', 'orchestral', 'medieval', 'modern', 'heavenly']
    orchestral = next(item for item in catalog['templates'] if item['id'] == 'orchestral')
    assert 'choir' in orchestral['instruments']['red']
    assert 'saxophone' in orchestral['instruments']['green']
    assert 'violin' in orchestral['instruments']['blue']
    heavenly = next(item for item in catalog['templates'] if item['id'] == 'heavenly')
    assert 'choir' in heavenly['instruments']['red']
    assert 'string' in heavenly['instruments']['green']
    assert 'chime' in heavenly['instruments']['blue']
    for path in ('/analyze', '/sonify'):
        assert client.post(f'{path}?sound_template=unknown').status_code == 422


def test_all_templates_render_distinct_continuous_audio():
    # Same colorful map and scan: timbre is the only requested change.
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:48, :48] = [255, 0, 0]
    image[:48, 48:] = [0, 255, 0]
    image[48:, :48] = [0, 0, 255]
    image[48:, 48:] = [255, 255, 0]
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    sounds = []
    for template in ('original', 'orchestral', 'medieval', 'modern', 'heavenly'):
        response = client.post(f'/sonify?duration_seconds=2&scan_mode=linear&sound_template={template}',
                               files={'file': ('map.png', buffer.getvalue(), 'image/png')})
        assert response.headers['x-sound-template'] == template
        pcm_values = pcm(response).astype(float)
        assert len(pcm_values) == 44100
        assert np.std(pcm_values) > 100
        sounds.append(response.content)
        analysis = client.post(f'/analyze?duration_seconds=2&sound_template={template}',
                               files={'file': ('map.png', buffer.getvalue(), 'image/png')}).json()
        assert analysis['soundTemplate'] == template
    assert len(set(sounds)) == 5
    assert len({len(sound) for sound in sounds}) == 1


def test_original_remains_default_template():
    from backend.audio import render_audio
    grid = np.ones((12, 48, 3)) * [0, 1, 0]
    events, _ = grid_events(grid, 2)
    assert render_audio(events, 2) == render_audio(events, 2, template='original')


def test_demo_exposes_template_selector_and_sends_choice():
    html = client.get('/demo').text
    assert 'id="sound-template"' in html
    for template in ('original', 'orchestral', 'medieval', 'modern', 'heavenly'):
        assert f'value="{template}"' in html
    assert 'sound_template:soundTemplate.value' in html
