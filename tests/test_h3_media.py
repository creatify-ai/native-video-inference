"""RGB protocol integrity, bounded transport, and encoder cancellation."""
from pathlib import Path
import socket
import struct
import sys
import threading
import time

import pytest

from models.minimax_h3_official.media import MediaOptions, MP4Stream, stereo_sample_count


@pytest.fixture
def encoder(tmp_path):
    path = tmp_path / 'encoder'
    path.write_text('#!' + sys.executable + '\nimport shutil,sys\n'
                    'with open(sys.argv[-1], "wb") as out: shutil.copyfileobj(sys.stdin.buffer, out)\n')
    path.chmod(0o755)
    return str(path)


def connect(stream):
    client = socket.socket(socket.AF_UNIX)
    client.connect(str(stream.socket_path))
    return client


def header(width=2, height=1, frames=2):
    return b'H3RGB001' + struct.pack('<IIII', width, height, frames, 0)


@pytest.mark.parametrize('capture', [False, True])
def test_rgb_chunks_preserve_every_byte_and_publish_only_on_completion(tmp_path, encoder, capture):
    output = tmp_path / 'output.mp4'
    raw = tmp_path / 'frames.raw' if capture else None
    with MP4Stream(output, tmp_path / 'unused.wav', 2, 1, 2,
                   ffmpeg=encoder, capture_path=raw) as stream:
        with connect(stream) as client:
            client.sendall(header())
            for first, data in enumerate((b'abcdef', b'ghijkl')):
                client.sendall(struct.pack('<IIQ', first, 1, len(data)) + data)
            assert not output.exists()
            client.sendall(struct.pack('<IIQ', 2, 0, 0))
        report = stream.finish()
        assert report['frames'] == 2 and report['rgb_bytes'] == 12
        assert report['raw_capture'] == capture
    assert output.read_bytes() == b'abcdefghijkl'
    assert not stream.socket_path.exists()
    assert stream.process.poll() == 0 and not stream.thread.is_alive()
    if capture:
        assert raw.read_bytes() == output.read_bytes()
    else:
        assert not (tmp_path / 'frames.raw').exists()


@pytest.mark.parametrize('kind', ['geometry', 'order', 'truncated', 'early_end'])
def test_rgb_protocol_rejects_incomplete_or_reordered_output(tmp_path, encoder, kind):
    output = tmp_path / 'output.mp4'
    with MP4Stream(output, tmp_path / 'unused.wav', 2, 1, 2, ffmpeg=encoder) as stream:
        with connect(stream) as client:
            client.sendall(header(width=4) if kind == 'geometry' else header())
            if kind == 'order':
                client.sendall(struct.pack('<IIQ', 1, 1, 6))
            elif kind == 'truncated':
                client.sendall(struct.pack('<IIQ', 0, 1, 6) + b'x')
            elif kind == 'early_end':
                client.sendall(struct.pack('<IIQ', 2, 0, 0))
        with pytest.raises(RuntimeError, match='RGB streaming failed'):
            stream.finish()
    assert not output.exists() and not stream.partial.exists()
    assert stream.process.poll() is not None and not stream.thread.is_alive()


def test_cancel_before_producer_connects_reaps_encoder_and_receiver(tmp_path, encoder):
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='cancelled'):
        with MP4Stream(tmp_path / 'output.mp4', tmp_path / 'unused.wav', 2, 1, 2, ffmpeg=encoder) as stream:
            raise RuntimeError('cancelled')
    assert time.monotonic() - start < 3
    assert stream.process.poll() is not None and not stream.thread.is_alive()
    assert not any(t.name == 'h3-mp4-stream' for t in threading.enumerate())


def test_encoder_failure_never_publishes_a_partial_file(tmp_path):
    encoder = tmp_path / 'failure'
    encoder.write_text('#!' + sys.executable + '\nimport sys\nsys.stderr.write("injected error")\nsys.exit(7)\n')
    encoder.chmod(0o755)
    with MP4Stream(tmp_path / 'output.mp4', tmp_path / 'unused.wav', 2, 1, 1, ffmpeg=str(encoder)) as stream:
        with connect(stream) as client:
            client.sendall(header(frames=1) + struct.pack('<IIQ', 0, 1, 6) + b'abcdef' + struct.pack('<IIQ', 1, 0, 0))
        with pytest.raises(RuntimeError):
            stream.finish()
    assert not stream.output.exists() and not stream.partial.exists()
    assert stream.process.poll() == 7


def test_raw_and_hash_options_require_explicit_opt_in():
    assert MediaOptions.parse() == MediaOptions('mp4', 'ffmpeg', False, False)
    assert MediaOptions.parse({'format': 'raw', 'hash_outputs': True}).hash_outputs
    for invalid in ({'format': 'unknown'}, {'capture_raw': 1}, {'hash_outputs': 'yes'}, {'extra': True}):
        with pytest.raises(ValueError):
            MediaOptions.parse(invalid)


@pytest.mark.parametrize('shape', [[2, 165600], [2, 1, 165600]])
def test_native_planar_stereo_metadata_with_or_without_singleton_axis(shape):
    assert stereo_sample_count({'output_shape': shape, 'sampling_rate': 32000, 'nonfinite': 0}) == 165600


@pytest.mark.parametrize('shape', [[], [2, 0], [1, 2, 10], [2, 2, 10], [10, 2]])
def test_invalid_audio_layout_cannot_be_encoded_as_stereo(shape):
    with pytest.raises(ValueError):
        stereo_sample_count({'output_shape': shape, 'sampling_rate': 32000, 'nonfinite': 0})
