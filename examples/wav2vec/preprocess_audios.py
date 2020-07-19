import argparse
import collections
import glob
import multiprocessing
import os
import traceback

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from webrtcvad import Vad


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', metavar='DIR', help='root directory containing mp3 files to index')
    parser.add_argument('dst_path', metavar='DIR', help='output directory')
    parser.add_argument('--src-ext', default='mp3', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--dst-ext', default='flac', type=str, metavar='EXT', help='extension to convert to')
    return parser


PreprocessResult = collections.namedtuple('PreprocessResult', [
    'src_duration',
    'dst_durations'
])


""" source: https://github.com/wiseman/py-webrtcvad/blob/master/example.py """
def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n


""" source: https://github.com/wiseman/py-webrtcvad/blob/master/example.py """
def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, aggressiveness, audio):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    vad = Vad(aggressiveness)
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frame_generator(30, audio, sample_rate):
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f for f in voiced_frames])


def split_audio(src_path, dst_path):
    assert os.path.isfile(src_path)
    file_ext = os.path.splitext(src_path)[-1]
    audio_segment = AudioSegment.from_file(src_path, format=file_ext.replace('.', ''))
    if audio_segment.frame_rate != 16000:
        audio_segment = audio_segment.set_frame_rate(16000)
    if audio_segment.sample_width != 2:
        audio_segment = audio_segment.set_sample_width(2)
    if audio_segment.channels != 1:
        audio_segment = audio_segment.set_channels(1)
    chunk_durations = []
    for i, raw_chunk in enumerate(vad_collector(sample_rate=audio_segment.frame_rate, frame_duration_ms=30,
                                                padding_duration_ms=300, aggressiveness=3, audio=audio_segment.raw_data)):
        if len(raw_chunk) & 1 != 0:
            raw_chunk = raw_chunk[:-1]
        chunk = AudioSegment(raw_chunk, frame_rate=audio_segment.frame_rate, sample_width=2, channels=1)
        chunk.export(f'{os.path.splitext(dst_path)[0]}.{i}.flac', format='flac')
        chunk_durations.append(float(chunk.duration_seconds))
    return PreprocessResult(src_duration=float(audio_segment.duration_seconds), dst_durations=chunk_durations)


def convert_audio(src_path, dst_dir):
    src_dirname = os.path.basename(os.path.dirname(src_path))
    src_basename_no_ext = os.path.basename(os.path.splitext(src_path)[0])
    # keep sub-folder structure to prevent filename conflicts:
    dst_path = os.path.join(dst_dir, src_dirname, f'{src_basename_no_ext}.flac')
    if not os.path.exists(dst_path):
        try:
            return split_audio(src_path, dst_path)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to convert {src_path} to {dst_path}: {e}")
    return None


def main(args):
    dir_path = os.path.realpath(args.src_path)
    dst_dir = os.path.realpath(args.dst_path)
    search_path = os.path.join(dir_path, '**/*.' + args.src_ext)
    src_paths = glob.glob(search_path, recursive=True)

    os.makedirs(dst_dir, exist_ok=True)

    progress_bar = tqdm(total=len(src_paths))

    src_durations, dst_durations = [], []

    def on_result(preprocess_result: PreprocessResult):
        if preprocess_result is not None:
            src_durations.append(preprocess_result.src_duration)
            dst_durations.extend(preprocess_result.dst_durations)
        progress_bar.update()

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = []
        for src_path in src_paths:
            src_path = os.path.realpath(src_path)
            results.append(pool.apply_async(convert_audio, args=(src_path, dst_dir), callback=on_result))
        for result in results:
            result.wait()

    print("Raw audio stats:")
    print("Mean length: ", np.mean(src_durations), "(Std: ", np.std(src_durations), ")")
    print("Median length: ", np.median(src_durations))
    print("Min length: ", np.min(src_durations))
    print("Max length: ", np.max(src_durations))

    print("Processed audio stats:")
    print("Mean length: ", np.mean(dst_durations), "Std:", np.std(dst_durations))
    print("Median length: ", np.median(dst_durations))
    print("Min length: ", np.min(dst_durations))
    print("Max length: ", np.max(dst_durations))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
