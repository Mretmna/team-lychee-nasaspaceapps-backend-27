"""Bounded raster decoding and deterministic image-to-event mapping."""
import io
import warnings
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
FORMATS = {'PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF'}
ROWS, COLUMNS = 12, 48
NOTES = ['C3', 'D3', 'E3', 'G3', 'A3', 'C4', 'D4', 'E4', 'G4', 'A4', 'C5', 'D5']

class ImageError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def decode_image(data: bytes):
    if not data:
        raise ImageError('The uploaded image is empty.')
    if len(data) > MAX_BYTES:
        raise ImageError('Image exceeds the 10 MiB limit.', 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as header:
                fmt = header.format
                width, height = header.size
                if fmt not in FORMATS:
                    raise ImageError('Use a still PNG, JPEG, WebP, BMP, or TIFF image.', 415)
                if width * height > MAX_PIXELS or max(width, height) > 8192:
                    raise ImageError('Image exceeds 16 megapixels or 8192 pixels per side.', 413)
                if getattr(header, 'n_frames', 1) != 1:
                    raise ImageError('Animated and multipage images are not supported.', 415)
                orientation = header.getexif().get(274, 1)
            with Image.open(io.BytesIO(data)) as verification:
                verification.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageError('Image dimensions exceed the decoding limit.', 413) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, ImageError):
            raise
        raise ImageError('File is not a valid supported raster image.', 415) from exc
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ImageError('OpenCV could not decode the image.', 415) from exc
    if image is None:
        raise ImageError('OpenCV could not decode the image.', 415)
    if image.dtype != np.uint8:
        raise ImageError('Only 8-bit images are supported; export as 8-bit PNG or JPEG.', 415)
    if image.shape[:2] != (height, width):
        raise ImageError('Image header and decoded dimensions disagree.')
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255
    elif image.shape[2] == 4:
        rgb = image[:, :, [2, 1, 0]].astype(np.float64) / 255
        rgb *= image[:, :, 3:4] / 255  # transparent pixels contribute silence
    elif image.shape[2] == 3:
        rgb = image[:, :, ::-1].astype(np.float64) / 255
    else:
        raise ImageError('Unsupported image channels.', 415)
    # Match displayed EXIF orientation without trusting a filename or MIME type.
    if orientation == 2: rgb = np.fliplr(rgb)
    elif orientation == 3: rgb = np.rot90(rgb, 2)
    elif orientation == 4: rgb = np.flipud(rgb)
    elif orientation == 5: rgb = np.transpose(rgb, (1, 0, 2))
    elif orientation == 6: rgb = np.rot90(rgb, 3)
    elif orientation == 7: rgb = np.flipud(np.fliplr(np.transpose(rgb, (1, 0, 2))))
    elif orientation == 8: rgb = np.rot90(rgb)
    return np.ascontiguousarray(rgb), fmt


def radial_grid(rgb):
    """Sample 48 clockwise rays, from the image center to its rectangle edge."""
    height, width = rgb.shape[:2]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    result = np.empty((ROWS, COLUMNS, 3), dtype=np.float64)
    # Three samples along each radial band reduce pixel-grid aliasing while
    # remaining exactly on the ray shown by the demo's scan marker.
    for column in range(COLUMNS):
        angle = 2 * np.pi * column / COLUMNS
        dx, dy = np.sin(angle), -np.cos(angle)
        distances = []
        if dx > 1e-12: distances.append((width - 1 - cx) / dx)
        if dx < -1e-12: distances.append(cx / -dx)
        if dy > 1e-12: distances.append((height - 1 - cy) / dy)
        if dy < -1e-12: distances.append(cy / -dy)
        length = min(distances)
        for row in range(ROWS):
            fractions = (row + np.array([1 / 6, 1 / 2, 5 / 6])) / ROWS
            xs = (cx + dx * length * fractions).astype(np.float32)[None, :]
            ys = (cy + dy * length * fractions).astype(np.float32)[None, :]
            sampled = cv2.remap(rgb, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            result[row, column] = sampled.mean(axis=1)[0]
    return result



def circle_grid(rgb, inward=False):
    """Sample concentric rings inside the image, one ring per scan position."""
    height, width = rgb.shape[:2]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    outer_radius = min(cx, cy)
    result = np.empty((ROWS, COLUMNS, 3), dtype=np.float64)
    for column in range(COLUMNS):
        step = COLUMNS - column - .5 if inward else column + .5
        radius = outer_radius * step / COLUMNS
        for row in range(ROWS):
            angles = 2 * np.pi * (row + np.array([1 / 6, 1 / 2, 5 / 6])) / ROWS
            xs = (cx + radius * np.sin(angles)).astype(np.float32)[None, :]
            ys = (cy - radius * np.cos(angles)).astype(np.float32)[None, :]
            sampled = cv2.remap(rgb, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            result[row, column] = sampled.mean(axis=1)[0]
    return result


def rectangular_ring_grid(rgb, inward=False):
    """Sample contours that expand to the edge of the whole rectangle."""
    height, width = rgb.shape[:2]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    result = np.empty((ROWS, COLUMNS, 3), dtype=np.float64)
    for column in range(COLUMNS):
        step = COLUMNS - column - .5 if inward else column + .5
        fraction = step / COLUMNS
        for row in range(ROWS):
            angles = 2 * np.pi * (row + np.array([1 / 6, 1 / 2, 5 / 6])) / ROWS
            dx, dy = np.sin(angles), -np.cos(angles)
            lengths = np.full(3, np.inf)
            for direction, center, limit in ((dx, cx, width - 1), (dy, cy, height - 1)):
                positive = direction > 1e-12
                negative = direction < -1e-12
                lengths[positive] = np.minimum(lengths[positive], (limit - center) / direction[positive])
                lengths[negative] = np.minimum(lengths[negative], center / -direction[negative])
            xs = (cx + dx * lengths * fraction).astype(np.float32)[None, :]
            ys = (cy + dy * lengths * fraction).astype(np.float32)[None, :]
            sampled = cv2.remap(rgb, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            result[row, column] = sampled.mean(axis=1)[0]
    return result


def image_events(rgb, duration, scan_mode='clockwise', region_shape='circle'):
    if scan_mode == 'linear':
        events, metadata = grid_events(cv2.resize(rgb, (COLUMNS, ROWS), interpolation=cv2.INTER_AREA), duration)
        metadata.update(scanMode='linear', regionShape=region_shape, scan='continuous left-to-right volume interpolation',
                        scanDirection='left-to-right', scanPositions=COLUMNS,
                        verticalBands=ROWS)
        return events, metadata
    if scan_mode in ('circle_outward', 'circle_inward'):
        inward = scan_mode == 'circle_inward'
        events, metadata = grid_events(circle_grid(rgb, inward) if region_shape == 'circle' else rectangular_ring_grid(rgb, inward), duration)
        rectangular = region_shape != 'circle'
        metadata.update(scanMode=scan_mode, regionShape=region_shape,
                        scan=('contracting rectangle from edge to center' if inward else 'expanding rectangle from center to edge') if rectangular else ('contracting circle from outer diameter to center' if inward else 'expanding circle from center to outer diameter'),
                        scanPositions=COLUMNS, angularBands=ROWS,
                        pivotNormalized=[.5, .5], outerRadiusNormalized='rectangle boundary' if rectangular else 'half the shorter image side',
                        radialDirection='edge-to-center' if inward else 'center-to-edge',
                        pitch='Pitch varies by angular sector around the ring; the scan radius determines time.')
        return events, metadata
    if scan_mode != 'clockwise':
        raise ValueError('scan_mode must be clockwise, linear, circle_outward, or circle_inward')
    events, metadata = grid_events(radial_grid(rgb), duration)
    metadata.update(scanMode='clockwise', regionShape=region_shape, scan='clockwise ray from center to image edge',
                    startAngleDegrees=0, angularDirection='clockwise',
                    angleDegreesPerSample=360 / COLUMNS,
                    pivotNormalized=[.5, .5], radialBands=ROWS,
                    radialDirection='center-to-edge')
    return events, metadata


def grid_events(grid, duration):
    """Map a 12x48 scan grid to STRAUSS voices (also used by tests)."""
    grid = cv2.resize(grid, (COLUMNS, ROWS), interpolation=cv2.INTER_AREA)
    brightness = grid @ np.array([0.2126, 0.7152, 0.0722])
    hsv = cv2.cvtColor(grid.astype(np.float32), cv2.COLOR_RGB2HSV)
    events = []
    voice_counts = {name: 0 for name in ('neutral', 'red', 'green', 'blue')}
    for x in range(COLUMNS):
        for y in range(ROWS):
            level = float(brightness[y, x])
            if level <= 1 / 255:
                continue
            hue, saturation = float(hsv[y, x, 0]), float(hsv[y, x, 1])
            # Adjacent hue anchors blend smoothly, including red across 360/0.
            sector = hue / 120
            lower = int(sector) % 3
            fraction = sector - int(sector)
            weights = np.zeros(3)
            weights[lower] = 1 - fraction
            weights[(lower + 1) % 3] = fraction
            strengths = {'neutral': 1 - saturation,
                         **dict(zip(('red', 'green', 'blue'), saturation * weights))}
            for voice, strength in strengths.items():
                if strength <= 1e-6:
                    continue
                events.append({'time': x / COLUMNS, 'pitch': (ROWS - 1 - y + .5) / ROWS,
                               'volume': level * float(strength) / ROWS,
                               'voice': voice, 'strength': float(strength),
                               'brightness': level, 'hueDegrees': hue, 'saturation': saturation,
                               'column': x, 'row': y})
                voice_counts[voice] += 1
    return events, {
        'scan': 'continuous angular volume interpolation', 'rows': ROWS, 'columns': COLUMNS,
        'durationSeconds': duration, 'secondsPerScanPosition': duration / COLUMNS,
        'mappingVersion': 'color-voices-v1',
        'pitch': 'Higher pitch near the center, lower pitch toward the edge; red uses one octave below the base scale, blue one octave above.',
        'hue': 'HSV hue blends red (0 degrees), green (120 degrees), and blue (240 degrees) instruments around the hue circle.',
        'instruments': {'red': 'rounded bass', 'green': 'soft midrange', 'blue': 'bell-like', 'neutral': 'soft neutral sine for grayscale'},
        'saturation': 'Colored layer strength equals HSV saturation; the neutral layer gets 1 minus saturation.',
        'voiceEventCounts': voice_counts,
        'notesLowToHigh': NOTES,
        'volume': 'Brightness = 0.2126 R + 0.7152 G + 0.0722 B; each voice volume = brightness times layer strength divided by 12.',
        'normalization': 'Fixed feature bounds; no per-image contrast stretching. Audio playback is peak-normalized by default; use normalize=false for fixed gain.',
        'alpha': 'Composited onto black; fully transparent pixels are silent.',
        'scope': 'Entire image including any labels, legends, land, and borders; no automatic map segmentation.',
        'meaning': 'Sonification of rendered pixels, not measured Earth-science quantities. Scan time is not observation time. STRAUSS Objects sustain each row tone while its volume interpolates between columns.',
        'eventCount': len(events),
    }
