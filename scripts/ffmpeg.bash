#!/bin/bash

# 源目录和目标目录
SOURCE_DIR="./PATH_TO_LOCAL_RESOURCE"
TARGET_DIR="./PATH_TO_LOCAL_RESOURCE"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 查找所有mp4文件并进行压缩
find "$SOURCE_DIR" -type f -name "*.mp4" | while read -r video; do
    # 获取相对路径
    rel_path="${video#$SOURCE_DIR/}"
    # 创建目标文件夹
    target_folder="$TARGET_DIR/$(dirname "$rel_path")"
    mkdir -p "$target_folder"
    
    # 设置输出文件路径
    output_file="$target_folder/$(basename "$video")"
    
    echo "压缩: $video -> $output_file"
    
    # 使用ffmpeg压缩视频
    ffmpeg -i "$video" -vf "scale=640:360" -b:v 500k "$output_file"
done

echo "压缩完成！所有压缩后的视频已保存到 $TARGET_DIR"