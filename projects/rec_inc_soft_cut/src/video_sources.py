"""Resolve local imported camera videos without renaming experimental cases."""
from pathlib import Path
import json
import re

VIDEO_ROOT = Path(__file__).resolve().parents[1] / 'data/videos'


def validate_case_names(names):
    if not names or len(set(names)) != len(names):
        raise ValueError('Provide unique local video stems')
    if any(re.fullmatch(r'[0-9]{4}[A-Za-z0-9][A-Za-z0-9_-]*', name) is None for name in names):
        raise ValueError('Use dated local video stems without paths or extensions')
    return list(names)


def video_directory(name):
    validate_case_names([name])
    matches = [p.parent.resolve() for p in VIDEO_ROOT.glob(f'*/{name}.mp4') if p.is_file()]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous local video stem {name}: {matches}')
    if matches:
        matches[0].relative_to(VIDEO_ROOT.resolve())
        return matches[0]
    # Retain the old default for preflight diagnostics and old datasets.
    if name.startswith('0930'):
        return VIDEO_ROOT / '0930'
    raise FileNotFoundError(f'No imported MP4 for {name} under {VIDEO_ROOT}')


def input_profile(name):
    directory = video_directory(name)
    path = directory/f'{name}_profile.json'
    if path.is_file():
        profile = json.loads(path.read_text(encoding='utf-8'))
        if type(profile.get('grid_enabled')) is not bool or type(profile.get('overlay_enabled')) is not bool:
            raise ValueError('Input profile needs explicit grid and overlay flags')
        if not 0 < profile.get('acquisition_fps', 0) < 100000:
            raise ValueError('Invalid acquisition clock in input profile')
        record = directory/f'{name}.json'
        if record.is_file():
            imported = json.loads(record.read_text(encoding='utf-8'))
            if profile.get('source') is not None and profile['source'] != imported['source']:
                raise ValueError('Input profile belongs to a different source video')
        return profile
    record = directory/f'{name}.json'
    if record.is_file() and json.loads(record.read_text(encoding='utf-8')).get('grid_0915_verified') is False:
        raise ValueError('Unverified camera overlay: supply an inspected input profile')
    return dict(grid_enabled=True, overlay_enabled=True, acquisition_fps=200.,
                acquisition_clock_status='historical_assumption_200_fps')
