#!/bin/bash
# process_all_pdfs.sh

# 设置源文件夹和目标文件夹
SOURCE_DIR="breast_cancer_planner_agent/breast_cancer_data/paper/ruijin_knowledge"
TARGET_DIR="breast_cancer_planner_agent/breast_cancer_data/paper/ruijin_knowledge_md"

# 确保目标文件夹存在
mkdir -p "$TARGET_DIR"

# 查找所有PDF文件并处理
find "$SOURCE_DIR" -type f -name "*.pdf" -print0 | while IFS= read -r -d '' pdf_file; do
    # 构建目标文件路径
    rel_path="${pdf_file#$SOURCE_DIR/}"  # 去掉源文件夹前缀
    txt_file="$TARGET_DIR/${rel_path%.pdf}.txt"  # 修改扩展名
    
    # 创建目标文件夹（如果需要）
    mkdir -p "$(dirname "$txt_file")"
    
    # 处理PDF：先用pdftotext提取，再用awk修复断行
    echo "处理: $pdf_file → $txt_file"
    
    # 方法1：使用pdftotext + awk
    pdftotext -layout -nopgbrk "$pdf_file" - | \
        awk 'NF > 0 {printf "%s ", $0} NF == 0 {print "\n"}' > "$txt_file"
    
    # 可选：统计行数
    lines=$(wc -l < "$txt_file" 2>/dev/null || echo "0")
    words=$(wc -w < "$txt_file" 2>/dev/null || echo "0")
    echo "  完成: $lines 行, $words 词"
done

echo "批量处理完成！"