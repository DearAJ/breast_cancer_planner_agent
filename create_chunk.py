"""
文档分块和摘要生成脚本
- 解析Markdown文档并进行分块
- 为表格生成摘要
- 合并小块
- 保存处理后的数据块（供向量数据库构建使用）
"""

import re
import pickle
import json
from typing import List, Dict
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class MarkdownParser:
    """Markdown文档解析器：将文档分块"""
    
    def __init__(self):
        self.title_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.+\|$', re.MULTILINE)
    
    def parse_document(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        blocks = []
        current_section = None
        current_subsection = None
        current_content = []
        title_hierarchy = []
        in_table = False
        table_lines = []
        line_num = 0
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # 检查是否是标题
            title_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if title_match:
                # 如果正在处理表格，先保存表格
                if in_table and table_lines:
                    table_text = '\n'.join(table_lines)
                    blocks.append(self._create_table_block(
                        table_text, title_hierarchy.copy(),
                        line_num - len(table_lines), file_path
                    ))
                    table_lines = []
                    in_table = False
                
                # 保存之前的内容块
                if current_content:
                    if current_section:
                        blocks.append(self._create_text_block(
                            current_content, title_hierarchy.copy(), 
                            line_num - len(current_content), file_path
                        ))
                
                # 处理新标题
                level = len(title_match.group(1))
                title_text = title_match.group(2).strip()
                
                # 更新标题层级
                title_hierarchy = title_hierarchy[:level-1] + [title_text]
                
                # 根据层级设置section
                if level == 1:
                    current_section = title_text
                    current_subsection = None
                elif level == 2:
                    current_subsection = title_text
                else:
                    # 三级及以下标题作为子节
                    pass
                
                current_content = []
                continue
            
            # 检查是否是表格行
            if line.strip().startswith('|'):
                in_table = True
                table_lines.append(line)
                continue
            elif in_table:
                # 表格结束（遇到非表格行）
                if table_lines:
                    table_text = '\n'.join(table_lines)
                    blocks.append(self._create_table_block(
                        table_text, title_hierarchy.copy(), 
                        line_num - len(table_lines), file_path
                    ))
                    table_lines = []
                in_table = False
            
            # 普通内容行
            if not in_table:
                if line.strip():
                    current_content.append(line)
                elif current_content:
                    # 空行，可能是段落分隔，保存当前段落
                    blocks.append(self._create_text_block(
                        current_content, title_hierarchy.copy(),
                        line_num - len(current_content), file_path
                    ))
                    current_content = []
        
        # 处理最后的内容
        if in_table and table_lines:
            table_text = '\n'.join(table_lines)
            blocks.append(self._create_table_block(
                table_text, title_hierarchy.copy(),
                line_num - len(table_lines), file_path
            ))
        elif current_content:
            blocks.append(self._create_text_block(
                current_content, title_hierarchy.copy(),
                line_num - len(current_content), file_path
            ))
        
        return blocks
    
    def _create_text_block(self, content_lines: List[str], 
                          title_hierarchy: List[str], 
                          start_line: int, file_path: str) -> Dict:
        """创建文本块"""
        content = '\n'.join(content_lines).strip()
        if not content:
            return None
        
        return {
            'type': 'text',
            'content': content,
            'title_hierarchy': title_hierarchy.copy(),
            'start_line': start_line,
            'end_line': start_line + len(content_lines) - 1,
            'source_file': file_path,
            'metadata': {
                'block_type': 'text',
                'title_path': title_hierarchy.copy(),
                'position': f"{file_path}:{start_line}-{start_line + len(content_lines) - 1}"
            }
        }
    
    def _create_table_block(self, table_text: str,
                           title_hierarchy: List[str],
                           start_line: int, file_path: str) -> Dict:
        table_lines = table_text.split('\n')
        end_line = start_line + len(table_lines) - 1
        return {
            'type': 'table',
            'content': table_text,
            'title_hierarchy': title_hierarchy.copy(),
            'start_line': start_line,
            'end_line': end_line,
            'source_file': file_path,
            'metadata': {
                'block_type': 'table',
                'title_path': title_hierarchy.copy(),
                'position': f"{file_path}:{start_line}-{end_line}"
            }
        }


class TableSummarizer:
    def __init__(self, llm_api_url: str = "http://0.0.0.0:30003/v1",
                 model_name: str = "Qwen2.5-7B-Instruct"):
        self.llm = ChatOpenAI(
            base_url=llm_api_url,
            model=model_name,
            temperature=0.0,
            timeout=300,
            api_key=""
        )
        self.system_prompt = """你是一位医学文献分析专家。你的任务是将Markdown格式的表格转换为一段简洁、准确的文字摘要。
摘要应该：
1. 保留表格的关键信息（表头、主要数据、分类等）
2. 使用自然流畅的中文描述
3. 突出医学相关的关键信息
4. 保持简洁，通常不超过200字"""
    
    def summarize_table(self, table_text: str) -> str:
        """为表格生成文字摘要"""
        prompt = f"""请将以下Markdown表格转换为一段简洁的文字摘要：

{table_text}

请直接输出摘要，不要添加其他说明。"""
        
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            summary = response.content.strip()
            
            # 清理可能包含的推理过程标签
            import re
            # 移除各种推理标签及其内容
            patterns = [
                r'<think>.*?</think>',
                r'<think>.*?</think>',
                r'<reasoning>.*?</reasoning>',
                r'<think>.*?</think>',
            ]
            for pattern in patterns:
                summary = re.sub(pattern, '', summary, flags=re.DOTALL | re.IGNORECASE)
            # 移除单独的标签（没有闭合标签的情况，从标签开始到结尾）
            summary = re.sub(r'<think>.*', '', summary, flags=re.DOTALL | re.IGNORECASE)
            summary = re.sub(r'<think>.*', '', summary, flags=re.DOTALL | re.IGNORECASE)
            summary = re.sub(r'<think>.*', '', summary, flags=re.DOTALL | re.IGNORECASE)
            summary = summary.strip()
            
            return summary
        except Exception as e:
            print(f"生成表格摘要时出错: {e}")
            # 如果失败，返回简化的表格描述
            lines = table_text.split('\n')
            if len(lines) > 0:
                header = lines[0] if lines else ""
                return f"表格内容：{header[:100]}..."
            return "表格内容"


class ChunkMerger:
    """块合并器：将太小的块与前一个合并"""
    
    def __init__(self, min_chunk_size: int = 100):
        self.min_chunk_size = min_chunk_size
    
    def merge_small_chunks(self, blocks: List[Dict]) -> List[Dict]:
        if not blocks:
            return []
        
        merged_blocks = []
        current_block = None
        
        for block in blocks:
            if block is None:
                continue
            
            content = block.get('content', '')
            content_size = len(content)
            
            # 表格块不合并
            if block.get('type') == 'table':
                if current_block:
                    merged_blocks.append(current_block)
                    current_block = None
                merged_blocks.append(block)
                continue
            
            # 如果当前块太小，尝试与前一个合并
            if current_block and content_size < self.min_chunk_size:
                # 检查是否可以合并（同一节内）
                if (current_block.get('title_hierarchy') == block.get('title_hierarchy') or
                    self._is_same_section(current_block, block)):
                    # 合并内容
                    current_block['content'] += '\n\n' + content
                    current_block['end_line'] = block.get('end_line', current_block.get('end_line'))
                    current_block['metadata']['position'] = (
                        f"{current_block['source_file']}:"
                        f"{current_block['start_line']}-{current_block['end_line']}"
                    )
                    continue
            
            # 保存当前块，开始新块
            if current_block:
                merged_blocks.append(current_block)
            current_block = block.copy()
        
        # 保存最后一个块
        if current_block:
            merged_blocks.append(current_block)
        
        return merged_blocks
    
    def _is_same_section(self, block1: Dict, block2: Dict) -> bool:
        h1 = block1.get('title_hierarchy', [])
        h2 = block2.get('title_hierarchy', [])
        
        # 至少前两级标题相同
        if len(h1) >= 2 and len(h2) >= 2:
            return h1[:2] == h2[:2]
        elif len(h1) >= 1 and len(h2) >= 1:
            return h1[0] == h2[0]
        return False


class DocumentProcessor:
    def __init__(self,
                 llm_api_url: str = "http://0.0.0.0:30003/v1",
                 llm_model: str = "Qwen2.5-7B-Instruct",
                 min_chunk_size: int = 100):
        """
        初始化文档处理器
        
        Args:
            llm_api_url: LLM API地址（用于生成表格摘要）
            llm_model: LLM模型名称
            min_chunk_size: 最小块大小（小于此大小的块会被合并）
        """
        self.table_summarizer = TableSummarizer(llm_api_url, llm_model)
        self.parser = MarkdownParser()
        self.merger = ChunkMerger(min_chunk_size=min_chunk_size)
    
    def process_directory(self, data_dir: str, output_file: str = "processed_blocks.pkl"):
        """
        处理目录中的所有Markdown文件：分块 + 生成摘要
        
        Args:
            data_dir: 包含Markdown文件的目录
            output_file: 输出文件路径（保存处理后的块）
        
        Returns:
            处理后的块列表
        """
        data_path = Path(data_dir)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"开始处理目录: {data_dir}")
        
        # 获取所有Markdown文件
        md_files = list(data_path.glob("*.md"))
        if not md_files:
            print(f"警告：在 {data_dir} 中未找到Markdown文件")
            return []
        
        all_blocks = []
        
        # 第一步：解析所有文件（分块）
        print(f"\n=== 第一步：文档分块 ===")
        for md_file in md_files:
            print(f"\n处理文件: {md_file.name}")
            blocks = self.parser.parse_document(str(md_file))
            print(f"  解析得到 {len(blocks)} 个原始块")
            all_blocks.extend(blocks)
        
        print(f"\n总共解析得到 {len(all_blocks)} 个原始块")
        
        # 第二步：为表格生成摘要
        print(f"\n=== 第二步：生成表格摘要 ===")
        table_count = 0
        for block in all_blocks:
            if block.get('type') == 'table':
                table_count += 1
                print(f"  处理表格 {table_count} (行 {block['start_line']}-{block['end_line']})...")
                summary = self.table_summarizer.summarize_table(block['content'])
                block['summary'] = summary
                block['search_content'] = summary  # 检索时使用摘要
            else:
                block['search_content'] = block['content']
        
        print(f"  共处理了 {table_count} 个表格")
        
        # 第三步：合并小块
        print(f"\n=== 第三步：合并小块 ===")
        merged_blocks = self.merger.merge_small_chunks(all_blocks)
        print(f"合并后得到 {len(merged_blocks)} 个块")
        
        # 保存处理后的块
        print(f"\n=== 保存结果 ===")
        # 保存pkl文件（供程序使用）
        with open(output_path, 'wb') as f:
            pickle.dump(merged_blocks, f)
        print(f"  PKL文件已保存到: {output_path}")
        
        # 同时保存jsonl文件（方便查看）
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for block in merged_blocks:
                json.dump(block, f, ensure_ascii=False)
                f.write('\n')
        print(f"  JSONL文件已保存到: {jsonl_path}")
        
        print(f"\n处理完成！总共处理了 {len(merged_blocks)} 个块")
        
        return merged_blocks


if __name__ == "__main__":
    # 文档分块和摘要生成
    processor = DocumentProcessor(
        llm_api_url="http://0.0.0.0:30003/v1",
        llm_model="Qwen3-8B"
    )
    
    processor.process_directory(
        data_dir="data/markdown",
        output_file="data/chunks/chunks.pkl"
    )
    
    print("\n文档分块和摘要生成完成！")
