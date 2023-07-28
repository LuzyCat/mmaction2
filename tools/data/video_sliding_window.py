import os
import cv2
import argparse
from moviepy.editor import VideoFileClip

def parse_args():
    parser = argparse.ArgumentParser(description="Video Sliding Window Sampler")
    parser.add_argument("input_directory", type=str, help="입력 비디오 파일이 있는 디렉토리 경로")
    parser.add_argument("output_directory", type=str, help="샘플 비디오 파일을 저장할 디렉토리 경로")
    parser.add_argument("--window_size", type=int, default=10, help="슬라이딩 윈도우 크기 (초 단위)")
    parser.add_argument("--stride", type=int, default=5, help="슬라이딩 윈도우 간격 (초 단위)")
    args = parser.parse_args()

    return args

def sliding_window(video, window_size, stride):
    frame_rate = video.fps  # 영상의 프레임 레이트
    window_length = int(window_size * frame_rate)
    stride_length = int(stride * frame_rate)

    num_frames = int(video.duration * frame_rate)
    windowed_samples = []

    for start_frame in range(0, num_frames - window_length + 1, stride_length):
        end_frame = start_frame + window_length
        windowed_sample = video.subclip(start_frame / frame_rate, end_frame / frame_rate)
        windowed_samples.append(windowed_sample)

    return windowed_samples

def process_video_file(input_file, output_dir, window_size=10, stride=5):
    # 입력 비디오 파일 로드
    video = VideoFileClip(input_file)

    # 비디오 길이가 10초를 넘을 경우에만 슬라이딩 윈도우 적용
    if video.duration > window_size:
        # 슬라이딩 윈도우 적용
        windowed_samples = sliding_window(video, window_size, stride)

        # 새로운 파일로 저장
        filename_without_extension = os.path.splitext(os.path.basename(input_file))[0]
        for i, sample in enumerate(windowed_samples):
            output_file = os.path.join(output_dir, f"{filename_without_extension}_sample_{i}.mp4")
            sample.write_videofile(output_file, codec="libx264", audio_codec="aac", threads = 8)

    video.close()

def main(args):
    # 디렉토리에 있는 모든 파일 검색
    for file_name in os.listdir(args.input_directory):
        if file_name.lower().endswith(".mp4"):
            file_path = os.path.join(args.input_directory, file_name)
            process_video_file(file_path, args.output_directory, args.window_size, args.stride)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    