import os
from moviepy.editor import VideoFileClip
import argparse
from datetime import datetime

def convert_mp4_to_gif(mp4_path, gif_path=None, fps=15, resize_factor=0.5):
    """
    将MP4视频转换为GIF动画
    
    参数:
        mp4_path (str): MP4文件路径
        gif_path (str, optional): 输出GIF文件路径，如果为None则自动生成
        fps (int): GIF的帧率
        resize_factor (float): 调整大小的比例因子，用于减小GIF文件大小
    
    返回:
        str: 生成的GIF文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"找不到MP4文件: {mp4_path}")
    
    # 如果没有指定输出路径，则自动生成
    if gif_path is None:
        output_dir = os.path.dirname(mp4_path)
        filename = os.path.splitext(os.path.basename(mp4_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(output_dir, f"{filename}_{timestamp}.gif")
    
    # 加载视频
    print(f"正在加载视频: {mp4_path}")
    video_clip = VideoFileClip(mp4_path)
    
    # 调整大小以减小GIF文件大小
    if resize_factor != 1.0:
        video_clip = video_clip.resize(resize_factor)
    
    # 转换为GIF
    print(f"正在转换为GIF，帧率: {fps}...")
    video_clip.write_gif(gif_path, fps=fps, program='ffmpeg')
    
    print(f"转换完成! GIF已保存到: {gif_path}")
    return gif_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将MP4视频转换为GIF动画")
    parser.add_argument("mp4_path", help="输入MP4文件路径")
    parser.add_argument("--gif_path", help="输出GIF文件路径 (可选)")
    parser.add_argument("--fps", type=int, default=15, help="GIF帧率 (默认: 15)")
    parser.add_argument("--resize", type=float, default=0.5, help="调整大小的比例因子 (默认: 0.5)")
    
    args = parser.parse_args()
    
    convert_mp4_to_gif(args.mp4_path, args.gif_path, args.fps, args.resize)