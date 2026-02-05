#!/bin/bash

# 批量转换PDF为TXT的脚本（递归处理所有子目录）
# 使用方法: ./convert_pdfs.sh [目录路径]
# 如果不指定目录，则使用脚本所在目录

# 获取要处理的目录（如果提供了参数则使用参数，否则使用脚本所在目录）
if [ -n "$1" ]; then
    TARGET_DIR="$1"
else
    TARGET_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录不存在: $TARGET_DIR"
    exit 1
fi

echo "开始递归转换目录: $TARGET_DIR"
echo "=========================================="
echo ""

# 统计变量
total_count=0
success_count=0
fail_count=0

# 使用find递归查找所有PDF文件
while IFS= read -r -d '' pdf_file; do
    # 获取文件所在目录
    file_dir=$(dirname "$pdf_file")
    
    # 获取文件名（不含路径和扩展名）
    base_name=$(basename "$pdf_file" .pdf)
    
    # 生成输出文件名（保存在PDF文件同一目录）
    txt_file="$file_dir/${base_name}.txt"
    
    # 显示相对路径（相对于目标目录）
    rel_path="${pdf_file#$TARGET_DIR/}"
    
    echo "正在转换: $rel_path -> ${rel_path%.pdf}.txt"
    
    # 执行转换命令
    pdftotext "$pdf_file" - | awk 'NF > 0 {printf "%s ", $0} NF == 0 {print "\n"}' > "$txt_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功转换: ${rel_path%.pdf}.txt"
        ((success_count++))
    else
        echo "✗ 转换失败: $rel_path"
        ((fail_count++))
    fi
    ((total_count++))
    echo ""
done < <(find "$TARGET_DIR" -type f -name "*.pdf" -print0)

echo "=========================================="
echo "批量转换完成！"
echo "总计: $total_count 个文件"
echo "成功: $success_count 个"
echo "失败: $fail_count 个"