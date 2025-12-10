"""
文档分块和摘要生成脚本
- 解析Markdown文档并进行分块
- 为表格生成摘要
- 合并小块
- 保存处理后的数据块，供向量数据库构建使用
"""

import re
import pickle
import json
from typing import List, Dict
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class MarkdownParser:
    def __init__(self):
        self.title_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.+\|$', re.MULTILINE)
    
    def parse_document(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        blocks = []
        current_section = None
        current_content = []
        title_hierarchy = []
        in_table = False
        table_lines = []
        line_num = 0
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
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


class ChapterBookmarkGenerator:
    """章节书签生成器：为每个章节生成摘要索引，用于分层检索"""
    
    def __init__(self, llm_api_url: str = "http://0.0.0.0:30003/v1",
                 model_name: str = "Qwen2.5-7B-Instruct"):
        self.llm = ChatOpenAI(
            base_url=llm_api_url,
            model=model_name,
            temperature=0.0,
            timeout=300,
            api_key=""
        )
        self.system_prompt = """你是一位医学文献分析专家。你的任务是为文档章节生成简洁准确的摘要。
摘要应该：
1. 概括该章节的主要内容和主题
2. 突出关键信息点
3. 保持简洁，通常100-200字"""
    
    def generate_chapter_summary(self, chapter_title: str, chapter_content: str) -> str:
        """为章节生成摘要"""
        prompt = f"""请为以下章节生成简洁摘要：

章节标题：{chapter_title}

章节内容：
{chapter_content[:2000]}  # 限制长度避免过长

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
            patterns = [
                r'<think>.*?</think>',
                r'<reasoning>.*?</reasoning>',
            ]
            for pattern in patterns:
                summary = re.sub(pattern, '', summary, flags=re.DOTALL | re.IGNORECASE)
            summary = re.sub(r'<think>.*', '', summary, flags=re.DOTALL | re.IGNORECASE)
            summary = summary.strip()
            
            return summary
        except Exception as e:
            print(f"生成章节摘要时出错: {e}")
            return f"章节：{chapter_title}"
    
    def generate_bookmarks(self, blocks: List[Dict]) -> List[Dict]:
        """生成章节书签列表
        
        区分章书签和节书签：
        - 章书签（一级标题）：包含该章下所有块，无论是否有节
        - 节书签（二级标题）：只包含该节下的块
        """

        # 第一步：按章（一级标题）组织块，同时记录每章下的节（二级标题）
        chapter_map = {}  # key: (file, chapter_title), value: list of (block_index, block)        
        section_map = {}  # key: (file, chapter_title, section_title), value: list of (block_index, block)
        
        for idx, block in enumerate(blocks):
            title_hierarchy = block.get('title_hierarchy', [])
            source_file = block.get('source_file', '')
            
            if len(title_hierarchy) >= 1:
                chapter_title = title_hierarchy[0]
                chapter_key = (source_file, chapter_title)
                
                # 记录到章
                if chapter_key not in chapter_map:
                    chapter_map[chapter_key] = []
                chapter_map[chapter_key].append((idx, block))
                
                # 如果有二级标题，记录到节
                if len(title_hierarchy) >= 2:
                    section_title = title_hierarchy[1]
                    section_key = (source_file, chapter_title, section_title)
                    if section_key not in section_map:
                        section_map[section_key] = []
                    section_map[section_key].append((idx, block))
            else:
                # 没有标题的块，归入"未分类"章
                chapter_key = (source_file, '未分类')
                if chapter_key not in chapter_map:
                    chapter_map[chapter_key] = []
                chapter_map[chapter_key].append((idx, block))
        
        # 第二步：生成书签
        bookmarks = []
        
        # 为每章生成章书签
        for (source_file, chapter_title), block_list in chapter_map.items():
            # 检查该章是否有子节
            has_sections = any(
                s[0] == source_file and s[1] == chapter_title 
                for s in section_map.keys()
            )
            
            # 收集章内容（用于生成摘要）
            chapter_contents = []
            chapter_block_indices = []
            for block_idx, block in block_list:
                content = block.get('search_content', block.get('content', ''))
                if content.strip():
                    chapter_contents.append(content[:1024])  # 限制每个块的长度
                    chapter_block_indices.append(block_idx)
            
            chapter_content = '\n\n'.join(chapter_contents)
            
            # 如果有节，先为每节生成节摘要，然后用节的摘要生成章的摘要
            if has_sections:
                # 先为每个节生成节书签和摘要
                section_summaries = []  # 收集所有节的摘要，用于生成章摘要
                for (file, chap, section_title), section_block_list in section_map.items():
                    if file == source_file and chap == chapter_title:
                        # 收集节内容（用于生成摘要）
                        section_contents = []
                        section_block_indices = []
                        for block_idx, block in section_block_list:
                            content = block.get('search_content', block.get('content', ''))
                            if content.strip():
                                section_contents.append(content[:1024])
                                section_block_indices.append(block_idx)
                        
                        section_content = '\n\n'.join(section_contents)
                        
                        # 生成节摘要
                        if section_content:
                            section_summary = self.generate_chapter_summary(
                                f"{chapter_title} > {section_title}", 
                                section_content
                            )
                        else:
                            section_summary = f"节：{section_title}"
                        
                        section_summaries.append(f"{section_title}: {section_summary}")
                        
                        # 创建节书签
                        bookmarks.append({
                            'type': 'chapter_bookmark',
                            'chapter_level': 'section',  # 标记为节
                            'chapter_path': [chapter_title, section_title],
                            'chapter_title': f"{chapter_title} > {section_title}",
                            'chapter_summary': section_summary,
                            'search_content': section_summary,
                            'block_indices': section_block_indices,  # 只包含该节下的块
                            'source_file': source_file,
                            'block_count': len(section_block_indices),
                            'parent_chapter': chapter_title,  # 记录父章
                            'metadata': {
                                'block_type': 'chapter_bookmark',
                                'chapter_level': 'section',
                                'chapter_path': [chapter_title, section_title],
                                'parent_chapter': chapter_title,
                                'is_bookmark': True
                            }
                        })
                
                # 基于节的摘要生成章的摘要
                if section_summaries:
                    sections_summary_text = '\n\n'.join(section_summaries)
                    # 使用专门的 prompt 基于节摘要生成章摘要
                    prompt = f"""请基于以下各节的摘要，为该章生成一个整体摘要：

章节标题：{chapter_title}

各节摘要：
{sections_summary_text[:2000]}

请直接输出章的摘要，不要添加其他说明。"""
                    try:
                        messages = [
                            SystemMessage(content=self.system_prompt),
                            HumanMessage(content=prompt)
                        ]
                        response = self.llm.invoke(messages)
                        summary = response.content.strip()
                        
                        # 清理可能包含的推理过程标签
                        import re
                        patterns = [
                            r'<think>.*?</think>',
                            r'<reasoning>.*?</reasoning>',
                        ]
                        for pattern in patterns:
                            summary = re.sub(pattern, '', summary, flags=re.DOTALL | re.IGNORECASE)
                        summary = re.sub(r'<think>.*', '', summary, flags=re.DOTALL | re.IGNORECASE)
                        summary = summary.strip()
                    except Exception as e:
                        print(f"基于节摘要生成章摘要时出错: {e}")
                        # 如果失败，使用章内容生成摘要
                        summary = self.generate_chapter_summary(chapter_title, chapter_content) if chapter_content else f"章节：{chapter_title}"
                else:
                    summary = self.generate_chapter_summary(chapter_title, chapter_content) if chapter_content else f"章节：{chapter_title}"
            else:
                # 无节：直接基于章内容生成章摘要
                if chapter_content:
                    summary = self.generate_chapter_summary(chapter_title, chapter_content)
                else:
                    summary = f"章节：{chapter_title}"
            
            # 创建章书签
            bookmarks.append({
                'type': 'chapter_bookmark',
                'chapter_level': 'chapter',  # 标记为章
                'chapter_path': [chapter_title],
                'chapter_title': chapter_title,
                'chapter_summary': summary,
                'search_content': summary,  # 检索时使用摘要
                'block_indices': chapter_block_indices,  # 包含该章下所有块
                'source_file': source_file,
                'block_count': len(chapter_block_indices),
                'has_sections': has_sections,  # 标记是否有子节
                'metadata': {
                    'block_type': 'chapter_bookmark',
                    'chapter_level': 'chapter',
                    'chapter_path': [chapter_title],
                    'has_sections': has_sections,
                    'is_bookmark': True
                }
            })
        
        return bookmarks


class DocumentProcessor:
    def __init__(self, llm_api_url, llm_model, min_chunk_size, generate_bookmarks):
        self.table_summarizer = TableSummarizer(llm_api_url, llm_model)
        self.parser = MarkdownParser()
        self.merger = ChunkMerger(min_chunk_size=min_chunk_size)
        self.bookmark_generator = ChapterBookmarkGenerator(llm_api_url, llm_model) if generate_bookmarks else None
        self.generate_bookmarks = generate_bookmarks
    
    def process_directory(self, data_dir: str, output_file: str):
        data_path = Path(data_dir)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_files = list(data_path.glob("*.md"))
        if not md_files:
            print(f"警告：在 {data_dir} 中未找到Markdown文件")
            return []

        all_blocks = []
        
        # 第一步：分块
        for md_file in md_files:
            blocks = self.parser.parse_document(str(md_file))
            all_blocks.extend(blocks)
        print(f"\n总共解析得到 {len(all_blocks)} 个块")
        
        # 第二步：为表格生成摘要
        table_count = 0
        for block in all_blocks:
            if block.get('type') == 'table':
                table_count += 1
                summary = self.table_summarizer.summarize_table(block['content'])
                block['summary'] = summary
                block['search_content'] = summary  # 检索时使用摘要
            else:
                block['search_content'] = block['content']
        print(f"  共处理了 {table_count} 个表格")
        
        # 第三步：合并小块
        merged_blocks = self.merger.merge_small_chunks(all_blocks)
        print(f"合并后得到 {len(merged_blocks)} 个块")
        
        # 第四步：生成章节书签（用于分层检索）
        bookmarks = []
        if self.generate_bookmarks and self.bookmark_generator:
            print(f"\n=== 第四步：生成章节书签 ===")
            bookmarks = self.bookmark_generator.generate_bookmarks(merged_blocks)
            print(f"  生成了 {len(bookmarks)} 个章节书签")
            for bookmark in bookmarks:
                print(f"    - {bookmark['chapter_title']} ({bookmark['block_count']} 个块)")

        
        # 保存处理后的块和书签
        print(f"\n=== 保存结果 ===")
        
        output_data = {
            'blocks': merged_blocks,
            'bookmarks': bookmarks  # 章节书签列表
        }
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"  PKL文件已保存到: {output_path}")
        
        
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            # 先保存书签
            for bookmark in bookmarks:
                json.dump(bookmark, f, ensure_ascii=False)
                f.write('\n')
            # 再保存内容块
            for block in merged_blocks:
                json.dump(block, f, ensure_ascii=False)
                f.write('\n')
        print(f"  JSONL文件已保存到: {jsonl_path}")
        
        print(f"\n处理完成！")
        print(f"  - 内容块: {len(merged_blocks)} 个")
        print(f"  - 章节书签: {len(bookmarks)} 个")
        
        return output_data


if __name__ == "__main__":
    processor = DocumentProcessor(
        llm_api_url="http://0.0.0.0:30003/v1",
        llm_model="Qwen3-8B",
        min_chunk_size=250,
        generate_bookmarks=True
    )
    
    processor.process_directory(
        data_dir="data/markdown",
        output_file="data/chunks/chunks.pkl"
    )
