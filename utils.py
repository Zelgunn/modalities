import tensorflow as tf
import numpy as np
from moviepy.editor import VideoClip, AudioClip
from moviepy.audio.AudioClip import AudioArrayClip


def int64_list_feature(int64_list) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def float_list_feature(float_list) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=float_list))


def bytes_list_feature(bytes_list) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))


def write_video_with_audio(path: str, video: np.ndarray, audio: np.ndarray, video_fps, audio_fps, norm_audio=True):
    if norm_audio:
        audio = (audio - audio.min()) / (audio.max() - audio.min())
        audio = audio * 2.0 - 1.0

    def make_video_frame(t: float):
        index = int(t * video_fps)
        frame = video[index]
        return frame

    def make_audio_frame(t: float):
        indexes = (np.asarray(t) * audio_fps).astype(np.int32)
        frame = audio[indexes]
        return frame

    video_duration = len(video) / video_fps
    audio_duration = len(audio) / audio_fps
    duration = min(video_duration, audio_duration)

    print("Video duration : {} - Audio duration : {}".format(round(video_duration, 2), round(audio_duration, 2)))

    video_clip = VideoClip(make_frame=make_video_frame, duration=duration)
    # audio_clip = AudioArrayClip(array=audio, fps=audio_fps)
    audio_clip = AudioClip(make_frame=make_audio_frame, duration=duration, fps=audio_fps)

    clip: VideoClip = video_clip.set_audio(audio_clip)
    clip.write_videofile(filename=path, fps=video_fps, audio_fps=audio_fps, codec="mpeg4")
