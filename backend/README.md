# FastAPI + STRAUSS image sonification

This is the requested server-side image workflow. It is separate from the existing offline video feature pipeline and browser Web Audio player. No frontend upload UI has been added. Existing `/data/ocean.json` remains unchanged.

## Start

From the repository root, with Python 3.11–3.14 (tested on Python 3.14, macOS ARM64):

```sh
python3 -m venv .venv-api
.venv-api/bin/python -m pip install -r requirements-api.txt
.venv-api/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/demo to upload an image, generate sound, and watch the selected scan marker. The API explorer is at http://127.0.0.1:8001/docs. Next.js connects to this service through `NEXT_PUBLIC_SONIFY_API_URL` (default `http://localhost:8001`), while map selection remains on port 8000. STRAUSS dependencies include scientific and notebook packages, so the first install is substantially larger than FastAPI alone. No sound device, external model, API key, or FFmpeg executable is needed to render these WAVs.

## Frontend contract

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | None | Service status and engine |
| `GET /sources` | None | Default NASA source, credits, original URL, image URL and hash |
| `GET /templates` | None | Sound templates and red/green/blue instrument labels |
| `GET /sources/nasa-ocean/image` | None | Bundled NASA JPEG for display |
| `POST /analyze?duration_seconds=12` | Optional multipart field `file` | Image dimensions, provenance and explanation of the sound mapping |
| `POST /sonify?duration_seconds=12` | Optional multipart field `file` | `audio/wav`, mono by default or spatial stereo signed 16-bit PCM, 22,050 Hz |

`spatial_audio=true` positions STRAUSS sources in 3D coordinates and renders a stereo WAV for headphone playback; the default is mono. `eight_d_audio=true` also produces stereo and adds stronger scan-aligned panning with a short cross-ear echo; it can be selected without `spatial_audio=true`. Stereo provides directional movement, not head-tracked binaural localization. `X-Spatial-Audio` reports the output mode and `X-8D-Audio` reports the effect.

`sound_template` selects `original` (default), `orchestral`, `medieval`, `modern`, or `heavenly` for both `/analyze` and `/sonify`. `scan_mode` is a query parameter for both `/analyze` and `/sonify`: `clockwise` (default), `linear`, `circle_outward`, or `circle_inward`. `duration_seconds` defaults to 12 and allows 2–30 seconds. `normalize=true` (default) makes the WAV audible by scaling its peak to 80% of full scale. Use `normalize=false` to preserve absolute loudness comparisons between images. Supply `file` as multipart/form-data. Omit it entirely for the NASA fallback; an empty or invalid supplied file is an error, never a silent fallback. The server uses actual decoded content, not the filename or declared MIME type. Frontend file-picker hint: `.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff`.

The frontend should POST a `FormData` containing the file (allow the browser to set the multipart boundary), read the successful response as a Blob, create an object URL, and play it in an audio element after a user action. Revoke old object URLs. For the default, POST with no body and show `/sources/nasa-ocean/image`. Relative image URLs in source metadata are relative to the API host. No image files or generated audio are persisted by the API; multipart parsing may temporarily spool uploads to disk, then closes/removes them.

For a synchronized marker, use `progress = audio.currentTime / audio.duration`. With `scan_mode=clockwise`, use `angle = 2π × progress`, starting at 12 o’clock, and cast a ray from the image center to the rectangular edge. With `scan_mode=linear`, show a full-height vertical line at `progress × image width`. For `circle_outward`, expand a centered ring from radius zero to half the shorter image side; `circle_inward` reverses that path. `/demo` implements all four modes. The audio timeline scans a still image; it is not Earth observation time. The response is fully rendered before playback, not a live synthesis stream. The Next.js frontend calls these endpoints for uploaded maps and visualization sound.

WAV response headers include `X-Sound-Template` and `X-Source-Id`, `X-Source-Url`, `X-Image-SHA256`, `X-Audio-Duration`, and `X-Sample-Rate`. Use `/sources` or `/analyze` for full credits and mapping explanation. User uploads are labeled as user-provided and never automatically attributed to NASA.

Try it:

```sh
curl -f -X POST 'http://127.0.0.1:8000/sonify?duration_seconds=12' -o /tmp/nasa-sonification.wav
curl -f -X POST 'http://127.0.0.1:8000/sonify?duration_seconds=12' -F 'file=@/absolute/path/map.png' -o /tmp/my-map.wav
curl -f -X POST 'http://127.0.0.1:8000/analyze' -F 'file=@/absolute/path/map.png'
```

Errors use JSON `detail`: 413 for limits, 415 for unsupported/undecodable formats, 422 for empty image or invalid options, 503 with `Retry-After` when a worker is busy. Do not interpret error responses as WAVs.

## Accepted images and transformations

Still PNG, JPEG, WebP, BMP and TIFF supported by both Pillow header validation and OpenCV decoding. Raster images only: PDF, SVG, video, GIF, animations, multi-page TIFF, and non-8-bit data are rejected. A map is a raster picture of a map; this service does not parse geospatial files or infer geographic coordinates. Limits: 10 MiB file, 11 MiB entire multipart request, 16 megapixels, 8192 pixels per side. Dimensions are checked before OpenCV allocation. EXIF orientation is applied; grayscale is expanded to RGB; transparency is composited onto black.

In clockwise mode, OpenCV samples the image along 48 radial rays (7.5° apart), from 12 o’clock clockwise. Each ray ends at the rectangular image edge. It has 12 radial bands from center to edge, with three bilinear sub-samples per band. In linear mode, OpenCV area-resizes the entire image into 48 columns and 12 rows; columns advance left to right. In the circle modes, 48 concentric rings progress outward or inward; each ring samples 12 angular sectors with three bilinear samples per sector. Each cell's gamma-encoded RGB brightness is `0.2126R + 0.7152G + 0.0722B`, with channels divided by 255. Cells at or below 1/255 are silent; when all cells are silent, `/sonify` returns a 422 explanation.

The `color-voices-v1` image mapping is the same in every sound template:

| Image property | Sound |
| --- | --- |
| Scan position | Clockwise angles, left-to-right columns, or expanding/contracting rings; 48 values are smoothly interpolated over the duration |
| Hue | Red = rounded bass; green = soft midrange; blue = bell-like; intermediate hues blend neighboring voices |
| Brightness | Volume of each cell |
| Saturation | Strength of colored layers; neutral layer strength is `1 - saturation` |
| Position across line | Clockwise: higher pitch near center; linear: higher pitch near top; circles: pitch varies by angular sector |

OpenCV HSV supplies hue in degrees and saturation in [0,1]. Hue anchors are red 0°, green 120°, blue 240°; interpolation is circular, so purple blends blue/red and yellow blends red/green. Hue weights sum to 1; colored strengths equal saturation times those weights. Gray uses a neutral sine, not the red instrument. A voice's volume is brightness × strength / 12. No per-image contrast stretching is applied.

In the Original template, the base row scale, bottom to top, is C3, D3, E3, G3, A3, C4, D4, E4, G4, A4, C5, D5. Red plays one octave lower; green and neutral use that scale; blue plays one octave higher. STRAUSS synthesizes each layer with sine partials (frequency ratio, amplitude): red `(1,.85), (2,.15)`; green `(1,.8), (2,.15), (3,.05)`; blue `(1,.55), (2.76,.30), (5.4,.15)`; neutral `(1,1)`. All colors use sustained STRAUSS Object voices. Each radial-band oscillator runs through the full clip while its volume interpolates between angles. Red remains a rounded bass, green a soft midrange sound, and blue has bell-like partials without short per-column strikes. The envelope has a short opening attack; exact parameters are in `backend/audio.py`. Every voice's partial amplitudes sum to one, and all color/neutral strengths sum to one.

A fixed 0.7 gain follows the STRAUSS mix. By default the final WAV peak is normalized to 0.8 for audibility; relative loudness and color mixtures within a map are preserved, but absolute loudness across maps is not. Set `normalize=false` for fixed gain and cross-image brightness comparisons. Rendering uses STRAUSS Objects, Synthesizer, Score and Sonification. All mappings are artistic representations of pixels, not inferred physical measurements.

Every pixel contributes, including labels, legends and land. No unverified generic ocean mask is applied to arbitrary uploads. Output conveys rendered color, brightness and location, not temperature, velocity, altitude, or other physical units. The scan samples only the 48 rotating rays and normalizes their lengths, so it does not preserve geographic distances.

## NASA fallback provenance

The bundled `assets/nasa-ocean.jpg` is the still image from [NASA SVS 3912: Global Sea Surface Currents and Temperature](https://svs.gsfc.nasa.gov/3912/). Credit: NASA/Goddard Space Flight Center Scientific Visualization Studio. Exact download URL and SHA-256 are in `assets/nasa-ocean.json`. The API checks the hash before using the fallback. It does not fetch a changing resource at request time, so fallback works without network access.

The supplied official resources page could not be retrieved. Its resource link was located through the [HackAthlone local-event reproduction](https://hackathlone.com/challenges/the-earth-information-jukebox) and checked directly against NASA SVS. The current official challenge requirements still need checking before submission. See [SVS media usage guidance](https://svs.gsfc.nasa.gov/help/).

## Tests and deployment

```sh
.venv-api/bin/python -m pytest backend/tests -q
```

Tests exercise actual STRAUSS rendering, NASA fallback, user upload attribution, silence, brightness response, spatial mappings, accepted formats, invalid input, animation/dimension/body limits, duration validation, busy behavior and CORS.

Deploy as a separate Python service (for example Render); the Next.js app can remain on Vercel. Install `requirements-api.txt`, then start `python -m uvicorn backend.app:app --host 0.0.0.0 --port $PORT`. Set `CORS_ORIGINS` to comma-separated frontend origins; defaults are localhost:3000 and 127.0.0.1:3000. This service permits one decode/render operation at a time per worker and returns 503 when busy. For a public deployment set request timeouts and rate/concurrency limits at the hosting proxy. There is no job queue or persistent database. This change adds a Python runtime requirement for image sonification, unlike the original static-video workflow.

Library references: [STRAUSS documentation](https://strauss.readthedocs.io/en/stable/index.html), [FastAPI optional file uploads](https://fastapi.tiangolo.com/tutorial/request-files/).

## Verified example

`examples/nasa-ocean.wav` is a 12-second response rendered by this API from the bundled NASA map. `examples/validation.json` records its sample count-derived duration, encoding, peak, RMS, and source hash. All 46 backend tests passed, along with the six original offline-pipeline tests. A Starlette warning about its future test-client HTTP library is emitted by the test dependency; it does not affect API rendering. STRAUSS is pinned to 1.0.3 because 1.0.0 failed with modern IPython.

If playback seems empty, check the HTTP status before saving the response: a 422 JSON error is not a WAV. Use `/analyze` to inspect `mapping.eventCount`; zero indicates no audible features. A 12-second successful response is 529,244 bytes. Swagger may only offer a download link: save it and open in an audio player. Restart the API or use `--reload` after changing the backend.

The `/sonify` OpenAPI success response explicitly uses `audio/wav` with a binary schema (not JSON). Responses use `Content-Disposition: attachment` for Swagger downloads. A real HTTP download was verified at 529,244 bytes and 12 seconds. Refresh `/docs` after restarting to discard the old schema.

Color mapping can be confirmed in `/analyze` at `mapping.mappingVersion` and `mapping.voiceEventCounts`, or in the WAV response header `X-Mapping-Version: color-voices-v1`. Equal-duration WAV files have equal sizes even when their sounds differ. Color examples in `examples/` use identical geometry and brightness to demonstrate the instrument differences. Tests also verify 10 ms windows have no gaps between angular positions and that a red-to-blue transition stays audible.

Regenerate the NASA example and the three 3-second color demonstrations with:

```sh
.venv-api/bin/python -m backend.generate_examples
```

The color demonstrations place the same horizontal strip in otherwise black images with equal calculated brightness; their differing timbres come from the hue mapping alone.

For clockwise mode, the visual marker starts at the top center, moves clockwise to right, bottom, left, and back to top. In linear mode, the full-height marker moves from the left edge to the right edge. In the circle modes, the marker expands from the center or contracts from the outer diameter. The clockwise endpoint is clipped to the true image rectangle (which may not be square). Sonification samples 48 scan positions and STRAUSS interpolates their values, while the marker moves smoothly with the audio clock. In clockwise mode, the 12 bands represent center-to-edge distance. In linear mode, they represent top-to-bottom image rows. In the circle modes, they represent angular sectors. None of the modes assigns scientific geographic coordinates.

The WAV response includes `X-Scan-Mode`, and `/analyze` returns `mapping.scanMode`. Change the demo selector before generating a new sound; the previous WAV is cleared when the mode changes.

## Sound templates

`cosmic_remnant` takes inspiration from NASA's [Cassiopeia A sonification](https://chandra.si.edu/photo/2024/sonify9/). It adds a high synth, piano-like blue voice, string-like green voice, brass-like red voice, and small metallic accents for bright neutral features. An outward circle scan begins with a triangle-like center cue. These are synthesized interpretations of Earth-map RGB data, not Chandra, Webb, or Hubble measurements.

The image scan, hue/saturation/brightness rules, and sample rate stay the same; `sound_template` changes the STRAUSS synthesis palette and, for Heavenly, adds a quiet echo. The default `original` profile preserves the previous bass/midrange/bell synthesis. `orchestral` assigns red to a choir-like tone, green to a saxophone-like tone, and blue to a violin-like tone. `medieval` uses low horn/lute/reed-like sounds. `modern` uses synth bass/electric pad/glassy lead. `heavenly` uses a warm angelic choir, floating string pad, and crystal chimes, with a quiet three-tap echo that softens the mix. All are **synthesized approximations**, not recordings or licensed instrument samples. Each remains a continuous sustained voice. Exact sine-partial ratios, octaves and envelopes are in `backend/audio.py`; discover labels with `GET /templates`. No template changes the scientific meaning of the image.

The supplied [YouTube reference](https://www.youtube.com/watch?v=ULNMBnooyeQ&t=43s) could not be retrieved by the available web tool, so its audio was not used or copied. The named instrument assignments above follow the user's explicit instructions.

The checked-in `backend/examples/nasa-*.wav` files demonstrate the five earlier templates. Regenerate them, including `cosmic_remnant`, with `.venv-api/bin/python -m backend.generate_examples`. Equal-duration WAVs have equal sizes even when the sound differs.

## Interactive audio legend

The demo also offers map-voice checkboxes and a compact playback explanation.
Both `/analyze` and `/sonify` accept `voices=red,blue` (default: all four voices).
Use `voices=` for silence. Missing selected colors also produce silence while
the scan continues. Hue blending still applies, so yellow can contribute to red
and green. Changing selections requires regeneration. Preview buttons are independent.
`mapping.voiceTimeline` contains 49 brightness-weighted voice levels (48 scan
positions plus the held endpoint); interpolate at playback fraction × 48.
`mapping.enabledVoices` identifies enabled voices. These levels precede audio
normalization and instrument envelopes/echoes, and are not acoustic loudness.

In `/demo`, use **Hear red**, **Hear green**, **Hear blue**, or **Hear gray / neutral**
to preview an isolated voice in the selected template. Each preview lasts two
seconds. The preview template selector defaults to following the map template;
you can also audition any template independently without changing map settings.
Each preview uses the same STRAUSS renderer as map audio, at a fixed middle scan
band. Peak-normalized previews teach timbre, not absolute image brightness.
Preview playback pauses map playback and vice versa. Changing templates clears
the previous preview. The map audio and scan marker remain separate.

Frontends can request `GET /legend/red?sound_template=orchestral` (or green,
blue, neutral and any supported template). The response is an in-memory mono
16-bit WAV; no preview files are saved on the server.

All four scan paths accept `region_shape=circle|square|full` on `/sonify` and `/analyze`. Outward and inward scans follow rectangular contours for square and full-map images so the entire image is represented.
