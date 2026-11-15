"""FastAPI service for user-provided raster maps and a bundled NASA fallback."""
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .imaging import ImageError, MAX_BYTES, COLUMNS, decode_image, image_events
from .audio import SAMPLE_RATE, render_audio, template_catalog, SOUND_TEMPLATES

ASSETS = Path(__file__).parent / 'assets'
SOURCE = json.loads((ASSETS / 'nasa-ocean.json').read_text())
render_lock = threading.Lock()
logger = logging.getLogger(__name__)


@contextmanager
def render_request():
    """Release the render slot and return a safe error if processing fails."""
    if not render_lock.acquire(blocking=False):
        raise HTTPException(503, 'Sonification worker is busy. Retry shortly.', headers={'Retry-After': '2'})
    try:
        yield
    except ImageError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Sonification request failed')
        raise HTTPException(500, 'The server could not finish processing. Try again.') from exc
    finally:
        render_lock.release()


class BodyLimit:
    """Bound total multipart bytes before the multipart parser buffers files."""
    def __init__(self, app, limit=MAX_BYTES + 1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > self.limit:
                return await JSONResponse({'detail': 'Request exceeds the 11 MiB multipart limit.'}, status_code=413)(scope, receive, send)
            if not message.get('more_body', False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


app = FastAPI(title='Earth Jukebox Image Sonification', version='1.0.0',
              description='Upload a still raster map or omit the file for NASA imagery. Audio is rendered server-side by STRAUSS, not streamed live.')
app.add_middleware(BodyLimit)
app.add_middleware(CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000,http://192.168.1.82:3000').split(',') if x.strip()],
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type'],
    expose_headers=['X-Source-Id', 'X-Source-Url', 'X-Image-SHA256', 'X-Audio-Duration', 'X-Sample-Rate', 'X-Audio-Normalization', 'X-Event-Count', 'X-Mapping-Version', 'X-Scan-Mode', 'X-Sound-Template', 'Content-Disposition'])

Duration = Annotated[float, Query(ge=2, le=30, allow_inf_nan=False, description='Playback scan duration, not observation time.')]
ScanMode = Literal['clockwise', 'linear', 'circle_outward', 'circle_inward']
SoundTemplate = Literal['original', 'orchestral', 'medieval', 'modern', 'heavenly']
Voices = Annotated[str, Query(pattern=r'^(?:(?:red|green|blue|neutral)(?:,(?:red|green|blue|neutral))*)?$', description='Comma-separated enabled voices; empty means silence.')]


@app.get('/health')
def health():
    return {'status': 'ok', 'engine': 'STRAUSS', 'version': '1.0.3'}


@app.get('/demo', include_in_schema=False)
def demo():
    return FileResponse(Path(__file__).parent / 'demo.html', media_type='text/html')


@app.get('/templates')
def templates():
    return {'defaultTemplate': 'original', 'templates': template_catalog()}


@app.get('/sources')
def sources():
    return {'defaultSourceId': SOURCE['id'], 'sources': [SOURCE]}


@app.get('/legend/{voice}', response_class=Response,
         responses={200: {'content': {'audio/wav': {}}, 'description': 'Two-second isolated instrument preview'}})
def legend(voice: Literal['red', 'green', 'blue', 'neutral'], sound_template: SoundTemplate = 'original'):
    with render_request():
        # A fixed middle band isolates timbre while retaining template octaves.
        events = [{'voice': voice, 'row': 5, 'column': column, 'volume': 1 / 12}
                  for column in range(COLUMNS)]
        audio = render_audio(events, 2, True, sound_template)
        return Response(audio, media_type='audio/wav', headers={'Cache-Control': 'no-store'})


@app.get('/sources/nasa-ocean/image')
def source_image():
    image = ASSETS / 'nasa-ocean.jpg'
    if not image.is_file():
        raise HTTPException(503, 'Bundled NASA image is unavailable.')
    return FileResponse(image, media_type='image/jpeg')


async def load_input(file):
    if file is None:
        try:
            data = (ASSETS / 'nasa-ocean.jpg').read_bytes()
        except OSError as exc:
            raise HTTPException(503, 'Bundled NASA image is unavailable.') from exc
        if hashlib.sha256(data).hexdigest() != SOURCE['sha256']:
            raise HTTPException(503, 'Bundled NASA image failed its integrity check.')
        source = SOURCE
    else:
        try:
            data = await file.read(MAX_BYTES + 1)
        except OSError as exc:
            logger.exception('Could not read uploaded image')
            raise HTTPException(500, 'The server could not read the upload. Select the image again and retry.') from exc
        finally:
            await file.close()
        source = {'id': 'user-upload', 'title': 'User-provided image', 'sourceUrl': None,
                  'credits': 'User-provided; no NASA attribution inferred.'}
    return data, source


def process(data, source, duration, audio, normalize=True, scan_mode='clockwise', template='original', voices='red,green,blue,neutral'):
    # Serialize expensive renders per worker and reject excess work instead of queuing.
    with render_request():
        rgb, fmt = decode_image(data)
        events, mapping = image_events(rgb, duration, scan_mode)
        if audio and not events:
            raise HTTPException(422, 'No audible image features: the image is black, transparent, or too dark after resizing. Try a brighter image or crop to the map content.')
        enabled = set(voices.split(',')) - {''}
        timeline = [{voice: 0. for voice in ('red', 'green', 'blue', 'neutral')} for _ in range(COLUMNS)]
        for event in events:
            timeline[event['column']][event['voice']] += event['volume']
        mapping['voiceTimeline'] = timeline + [timeline[-1].copy()]
        mapping['enabledVoices'] = sorted(enabled)
        events = [event for event in events if event['voice'] in enabled]
        mapping['eventCount'] = len(events)
        mapping['voiceEventCounts'] = {voice: sum(event['voice'] == voice for event in events) for voice in timeline[0]}
        metadata = {'source': source, 'imageSha256': hashlib.sha256(data).hexdigest(),
                    'image': {'width': rgb.shape[1], 'height': rgb.shape[0], 'format': fmt},
                    'mapping': mapping, 'sampleRate': SAMPLE_RATE,
                    'soundTemplate': template,
                    'instruments': {voice: preset['instrument'] for voice, preset in SOUND_TEMPLATES[template]['voices'].items()}}
        return (render_audio(events, duration, normalize, template), metadata) if audio else metadata


@app.post('/analyze')
async def analyze(duration_seconds: Duration = 12, scan_mode: ScanMode = 'clockwise', sound_template: SoundTemplate = 'original', file: Annotated[UploadFile | None, File()] = None, voices: Voices = 'red,green,blue,neutral'):
    data, source = await load_input(file)
    return await run_in_threadpool(process, data, source, duration_seconds, False, True, scan_mode, sound_template, voices)


@app.post('/sonify', response_class=Response, responses={200: {
    'content': {'audio/wav': {'schema': {'type': 'string', 'format': 'binary'}}},
    'description': 'Downloadable mono 16-bit PCM WAV'}})
async def sonify(duration_seconds: Duration = 12, scan_mode: ScanMode = 'clockwise', sound_template: SoundTemplate = 'original', normalize: Annotated[bool, Query(description='Normalize playback peak to 80 percent; disable to compare absolute brightness between images.')] = True, file: Annotated[UploadFile | None, File()] = None, voices: Voices = 'red,green,blue,neutral'):
    data, source = await load_input(file)
    audio, metadata = await run_in_threadpool(process, data, source, duration_seconds, True, normalize, scan_mode, sound_template, voices)
    return Response(audio, media_type='audio/wav', headers={
        'Content-Disposition': 'attachment; filename="sonification.wav"',
        'Cache-Control': 'no-store',
        'X-Source-Id': source['id'], 'X-Source-Url': source['sourceUrl'] or '',
        'X-Image-SHA256': metadata['imageSha256'], 'X-Audio-Duration': str(duration_seconds),
        'X-Sample-Rate': str(SAMPLE_RATE),
        'X-Audio-Normalization': 'peak-0.8' if normalize else 'fixed-gain',
        'X-Mapping-Version': metadata['mapping']['mappingVersion'],
        'X-Scan-Mode': metadata['mapping']['scanMode'],
        'X-Sound-Template': sound_template,
        'X-Event-Count': str(metadata['mapping']['eventCount'])})
