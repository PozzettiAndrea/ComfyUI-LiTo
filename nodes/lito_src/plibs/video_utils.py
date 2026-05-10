#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implements util functions for getting information from a video.

import typing as T

import av
import ffmpeg


def get_video_info_with_ffmpeg(filename: str, estimate_timestamps: bool = True):
    """
    Use ffmpeg to get video meta data.

    Args:
        filename:
            filename of the video

    Returns:
        frame_rate:
            fps.  Exact frame time may not be accurate for videos with variable frame rates.
        duration:
            in second
        width_px:
            number of pixels horizontally
        height_px:
            number of pixels vertically
        codec:
        total_frames:
            total number of frames. Not accurate for variable frame rate.
    """
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

    # Calculate total frames if not directly available
    total_frames = video_info.get("nb_frames")
    if total_frames is None:
        duration = float(video_info.get("duration", 0))
        frame_rate = float(eval(video_info["r_frame_rate"]))
        total_frames = int(duration * frame_rate)
    else:
        total_frames = int(total_frames)

    if estimate_timestamps:
        fps = float(eval(video_info["r_frame_rate"]))
        inv_fps = 1.0 / fps
        timestamps = [i * inv_fps for i in range(total_frames)]
    else:
        timestamps = None

    return {
        "frame_rate": float(eval(video_info["r_frame_rate"])),
        "duration": float(video_info.get("duration", 0)),
        "width_px": int(video_info["width"]),
        "height_px": int(video_info["height"]),
        "codec": video_info["codec_name"],
        "total_frames": total_frames,  # don't rely on this
        "frame_timestamps": timestamps,
    }


def get_video_info_with_pyav(
    filename: str,
    collect_timestamps: bool = True,
) -> T.Dict[str, T.Union[int, float, bool, T.List[float]]]:
    """Get detailed information about a video file including frame timestamps if requested.

    Args:
        video_path (str): Path to the video file
        collect_timestamps (bool, optional): Whether to collect frame timestamps.
            Set to False for large videos to avoid memory issues. Defaults to True.

    Returns:
        Dict[str, Union[int, float, bool, List[float]]]: Dictionary containing video information:
            - frame_rate (float): Average frame rate in frames per second
            - duration (float): Total duration in seconds
            - width_px (int): Frame width in pixels
            - height_px (int): Frame height in pixels
            - codec (str): Name of the video codec
            - total_frames (int): Total number of frames
            - time_base (float): Fundamental time unit (in seconds) for timestamps.
                               Expressed as fraction (e.g., 1/90000 means each tick is 1/90000 seconds)
            - is_variable_frame_rate (bool): Whether video has variable frame rate. Accurate if collect_timestamps is True
            - frame_timestamps (List[float]): List of frame timestamps in seconds.
                Only included if collect_timestamps=True

    Raises:
        av.error.ValueError: If the video file cannot be opened or is invalid
        FileNotFoundError: If the video file does not exist
    """

    with av.open(filename) as container:
        stream = container.streams.video[0]

        duration = float(container.duration) / av.time_base if container.duration else 0

        frame_times = []
        is_variable_rate = False

        if collect_timestamps:
            for packet in container.demux(video=0):
                if packet.pts is not None:
                    timestamp = float(packet.pts * stream.time_base)
                    frame_times.append(timestamp)
            total_frames = len(frame_times)

            # Check if frame rate is variable by analyzing intervals
            if len(frame_times) > 2:
                intervals = [frame_times[i + 1] - frame_times[i] for i in range(len(frame_times) - 1)]
                # Compare max and min intervals with some tolerance
                min_interval = min(intervals)
                max_interval = max(intervals)
                # Consider it variable if max interval differs from min by more than 1%
                is_variable_rate = (max_interval - min_interval) / min_interval > 0.01
        else:
            total_frames = stream.frames
            if total_frames == 0:
                total_frames = sum(1 for packet in container.demux(video=0) if packet.pts is not None)

        info = {
            "frame_rate": float(stream.average_rate),
            "duration": duration,
            "width_px": stream.width,
            "height_px": stream.height,
            "codec": stream.codec_context.name,
            "total_frames": total_frames,
            "time_base": float(stream.time_base),
            "is_variable_frame_rate": is_variable_rate,
        }

        if collect_timestamps:
            info["frame_timestamps"] = frame_times

        return info
