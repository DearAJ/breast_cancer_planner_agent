"""
FAISS向量数据库构建脚本（树结构+分开存储）
- 从已处理的块文件加载数据
- 分开存储章节索引和内容块索引
- 构建树结构（章->节->块）
- 支持DFS式分层检索
"""

import pickle
import json
from typing import List, Dict, Optional, Set
from pathlib import Path
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings


class VectorDatabaseBuilder:
    def __init__(self, embedding_api_url, embedding_model):
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=embedding_api_url,
            openai_api_key="",
            timeout=300
        )
    
    def build_from_processed_blocks(self, processed_blocks_file: str, output_dir: str):
        """
        从已处理的块文件构建 FAISS 向量数据库（树结构+分开存储）
        """
        processed_path = Path(processed_blocks_file)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)
        
        merged_blocks = data.get('blocks', [])
        bookmarks = data.get('bookmarks', [])
        
        # 第一步：构建树结构
        print("=== 第一步：构建树结构 ===")
        tree = self._build_tree(bookmarks)
        print(f"树结构构建完成：{len(tree)} 个章节节点")
        
        # 第二步：分开生成embeddings
        print("\n=== 第二步：生成embeddings ===")
        
        # 处理章节（章+节）
        chapter_embeddings = []
        chapter_texts = []
        chapter_metadatas = []
        chapter_id_to_index = {}  # 章节ID到索引的映射
        
        for i, bookmark in enumerate(bookmarks):
            search_text = bookmark.get('search_content', bookmark.get('chapter_summary', ''))
            if not search_text.strip():
                continue
            
            try:
                embedding = self.embeddings.embed_query(search_text)
                chapter_embeddings.append(embedding)
                chapter_texts.append(json.dumps({
                    'type': 'bookmark',
                    'chapter_title': bookmark.get('chapter_title', ''),
                    'chapter_summary': bookmark.get('chapter_summary', ''),
                    'block_indices': bookmark.get('block_indices', []),
                    'chapter_path': bookmark.get('chapter_path', []),
                    'chapter_level': bookmark.get('chapter_level', 'chapter'),
                    'parent_chapter': bookmark.get('parent_chapter', None)
                }, ensure_ascii=False))
                
                metadata = bookmark.get('metadata', {}).copy()
                metadata['is_bookmark'] = True
                metadata['bookmark_index'] = i  # 记录原始书签索引
                chapter_metadatas.append(metadata)
                
                # 记录章节ID到索引的映射（使用章节路径作为ID）
                chapter_id = tuple(bookmark.get('chapter_path', []))
                chapter_id_to_index[chapter_id] = len(chapter_embeddings) - 1
                
            except Exception as e:
                print(f"  生成章节embedding时出错 (书签 {i}): {e}")
                continue
        
        print(f"  章节embeddings: {len(chapter_embeddings)} 个")
        
        # 处理内容块（叶子节点）
        block_embeddings = []
        block_texts = []
        block_metadatas = []
        
        for i, block in enumerate(merged_blocks):
            if (i + 1) % 10 == 0:
                print(f"  处理进度: {i + 1}/{len(merged_blocks)}")
            
            search_text = block.get('search_content', block.get('content', ''))
            if not search_text.strip():
                continue
            
            try:
                embedding = self.embeddings.embed_query(search_text)
                block_embeddings.append(embedding)
                block_texts.append(block.get('content', ''))
                metadata = block.get('metadata', {}).copy()
                metadata['is_bookmark'] = False
                metadata['block_index'] = i  # 记录原始块索引
                block_metadatas.append(metadata)
            except Exception as e:
                print(f"  生成块embedding时出错 (块 {i}): {e}")
                continue
        
        print(f"  内容块embeddings: {len(block_embeddings)} 个")
        
        # 第三步：构建分开的FAISS索引
        print("\n=== 第三步：构建FAISS索引 ===")
        
        if not chapter_embeddings or not block_embeddings:
            print("错误：章节或内容块embeddings为空")
            return
        
        dimension = len(chapter_embeddings[0])
        
        # 章节索引
        chapter_index = faiss.IndexFlatL2(dimension)
        chapter_embeddings_array = np.array(chapter_embeddings).astype('float32')
        chapter_index.add(chapter_embeddings_array)
        print(f"  章节索引: {chapter_index.ntotal} 个向量")
        
        # 内容块索引
        block_index = faiss.IndexFlatL2(dimension)
        block_embeddings_array = np.array(block_embeddings).astype('float32')
        block_index.add(block_embeddings_array)
        print(f"  内容块索引: {block_index.ntotal} 个向量")
        
        # 第四步：保存索引和元数据
        print("\n=== 第四步：保存文件 ===")
        
        # 保存章节索引
        chapter_index_path = output_path / "chapter_index.faiss"
        faiss.write_index(chapter_index, str(chapter_index_path))
        
        chapter_texts_path = output_path / "chapter_texts.pkl"
        with open(chapter_texts_path, 'wb') as f:
            pickle.dump(chapter_texts, f)
        
        chapter_metadatas_path = output_path / "chapter_metadatas.pkl"
        with open(chapter_metadatas_path, 'wb') as f:
            pickle.dump(chapter_metadatas, f)
        
        # 保存内容块索引
        block_index_path = output_path / "block_index.faiss"
        faiss.write_index(block_index, str(block_index_path))
        
        block_texts_path = output_path / "block_texts.pkl"
        with open(block_texts_path, 'wb') as f:
            pickle.dump(block_texts, f)
        
        block_metadatas_path = output_path / "block_metadatas.pkl"
        with open(block_metadatas_path, 'wb') as f:
            pickle.dump(block_metadatas, f)
        
        # 保存树结构和映射
        tree_path = output_path / "tree.pkl"
        with open(tree_path, 'wb') as f:
            pickle.dump(tree, f)
        
        chapter_id_to_index_path = output_path / "chapter_id_to_index.pkl"
        with open(chapter_id_to_index_path, 'wb') as f:
            pickle.dump(chapter_id_to_index, f)
        
        # 保存原始数据（用于兼容）
        bookmarks_path = output_path / "bookmarks.pkl"
        with open(bookmarks_path, 'wb') as f:
            pickle.dump(bookmarks, f)
        
        blocks_path = output_path / "blocks.pkl"
        with open(blocks_path, 'wb') as f:
            pickle.dump(merged_blocks, f)
        
        print(f"\n向量数据库已保存到: {output_path}")
        print(f"  - 章节索引: {chapter_index_path}")
        print(f"  - 内容块索引: {block_index_path}")
        print(f"  - 树结构: {tree_path}")
    
    def _build_tree(self, bookmarks: List[Dict]) -> Dict:
        """
        构建树结构：{chapter_id: {'children': [section_ids], 'block_indices': [...]}}
        chapter_id 是 tuple(chapter_path)
        """
        tree = {}
        
        # 先找出所有章（一级标题）
        chapters = {}
        sections = {}
        
        for bookmark in bookmarks:
            chapter_path = bookmark.get('chapter_path', [])
            chapter_level = bookmark.get('chapter_level', 'chapter')
            chapter_id = tuple(chapter_path)
            
            if chapter_level == 'chapter':
                chapters[chapter_id] = bookmark
            elif chapter_level == 'section':
                sections[chapter_id] = bookmark
        
        # 构建树：章 -> 节 -> 块
        for chapter_id, chapter_bookmark in chapters.items():
            if chapter_id not in tree:
                tree[chapter_id] = {
                    'children': [],  # 子节ID列表
                    'block_indices': chapter_bookmark.get('block_indices', []),
                    'bookmark': chapter_bookmark
                }
            
            # 找出该章下的所有节
            parent_chapter = chapter_bookmark.get('chapter_title', '')
            for section_id, section_bookmark in sections.items():
                if section_bookmark.get('parent_chapter') == parent_chapter:
                    tree[chapter_id]['children'].append(section_id)
                    # 为节也创建节点
                    if section_id not in tree:
                        tree[section_id] = {
                            'children': [],  # 节没有子节点
                            'block_indices': section_bookmark.get('block_indices', []),
                            'bookmark': section_bookmark
                        }
        
        return tree


class VectorStore:
    """向量存储检索器（树结构+分开存储，支持DFS式检索）"""
    
    def __init__(self, db_path, embedding_model, openai_api_base, openai_api_key):
        self.db_path = Path(db_path)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            timeout=300
        )
        
        # 加载章节索引
        chapter_index_path = self.db_path / "chapter_index.faiss"
        if chapter_index_path.exists():
            self.chapter_index = faiss.read_index(str(chapter_index_path))
            with open(self.db_path / "chapter_texts.pkl", 'rb') as f:
                self.chapter_texts = pickle.load(f)
            with open(self.db_path / "chapter_metadatas.pkl", 'rb') as f:
                self.chapter_metadatas = pickle.load(f)
        else:
            # 兼容旧格式：尝试加载混合索引
            print("警告：未找到章节索引，尝试加载旧格式...")
            self.chapter_index = faiss.read_index(str(self.db_path / "faiss.index"))
            with open(self.db_path / "texts.pkl", 'rb') as f:
                all_texts = pickle.load(f)
            with open(self.db_path / "metadata.pkl", 'rb') as f:
                all_metadatas = pickle.load(f)
            
            # 分离章节和内容块
            self.chapter_texts = []
            self.chapter_metadatas = []
            chapter_indices = []
            for i, metadata in enumerate(all_metadatas):
                if metadata.get('is_bookmark', False):
                    self.chapter_texts.append(all_texts[i])
                    self.chapter_metadatas.append(metadata)
                    chapter_indices.append(i)
            
            # 重建章节索引（只包含章节向量）
            if chapter_indices:
                dimension = self.chapter_index.d
                temp_index = faiss.IndexFlatL2(dimension)
                # 需要重新加载embeddings，这里简化处理
                print("警告：旧格式不支持完整功能，建议重新构建索引")
        
        # 加载内容块索引
        block_index_path = self.db_path / "block_index.faiss"
        if block_index_path.exists():
            self.block_index = faiss.read_index(str(block_index_path))
            with open(self.db_path / "block_texts.pkl", 'rb') as f:
                self.block_texts = pickle.load(f)
            with open(self.db_path / "block_metadatas.pkl", 'rb') as f:
                self.block_metadatas = pickle.load(f)
        else:
            # 兼容旧格式
            print("警告：未找到内容块索引，使用旧格式...")
            self.block_index = self.chapter_index  # 临时使用
            self.block_texts = []
            self.block_metadatas = []
        
        # 加载树结构
        tree_path = self.db_path / "tree.pkl"
        if tree_path.exists():
            with open(tree_path, 'rb') as f:
                self.tree = pickle.load(f)
        else:
            self.tree = {}
            print("警告：未找到树结构文件")
        
        # 加载章节ID到索引的映射
        chapter_id_to_index_path = self.db_path / "chapter_id_to_index.pkl"
        if chapter_id_to_index_path.exists():
            with open(chapter_id_to_index_path, 'rb') as f:
                self.chapter_id_to_index = pickle.load(f)
        else:
            self.chapter_id_to_index = {}
        
        # 加载原始数据（用于兼容）
        bookmarks_path = self.db_path / "bookmarks.pkl"
        blocks_path = self.db_path / "blocks.pkl"
        if bookmarks_path.exists():
            with open(bookmarks_path, 'rb') as f:
                self.bookmarks = pickle.load(f)
        else:
            self.bookmarks = []
        
        if blocks_path.exists():
            with open(blocks_path, 'rb') as f:
                self.blocks = pickle.load(f)
        else:
            self.blocks = []
        
        print(f"向量数据库加载完成")
        print(f"  - 章节索引: {self.chapter_index.ntotal} 个向量")
        print(f"  - 内容块索引: {self.block_index.ntotal} 个向量")
        print(f"  - 树节点: {len(self.tree)} 个")
    
    def search(self, query: str, k: int) -> List[Dict]:
        """
        基础检索：在内容块索引中检索（返回叶子节点）
        """
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.block_index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.block_texts):
                results.append({
                    'content': self.block_texts[idx],
                    'metadata': self.block_metadatas[idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def retrieve(self, query: str, k: int) -> str:
        """
        检索并格式化结果（供Agent调用）
        """
        results = self.search(query, k)
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            title_path = metadata.get('title_path', [])
            title_str = ' > '.join(title_path) if title_path else '未知章节'
            formatted_results.append(
                f"【检索结果 {i}】\n"
                f"来源: {title_str}\n"
                f"位置: {metadata.get('position', '未知')}\n"
                f"内容:\n{result['content']}\n"
            )
        
        return "\n\n".join(formatted_results)
    
    def hierarchical_search(self, query: str, top_chapters: int = 2, topk: int = 5) -> str:
        """
        DFS式分层检索：
        1. 先检索最相关的top_chapters个章节
        2. 对每个章节，收集其下的所有内容块（DFS）
        3. 在候选块中检索，返回topk个叶子节点
        """
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 第一步：检索相关章节
        distances, indices = self.chapter_index.search(query_vector, top_chapters * 2)  # 多检索一些，过滤掉节
        
        # 筛选出章（不是节）
        chapter_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chapter_metadatas):
                metadata = self.chapter_metadatas[idx]
                chapter_level = metadata.get('chapter_level', 'chapter')
                if chapter_level == 'chapter':  # 只要章，不要节
                    bookmark_data = json.loads(self.chapter_texts[idx])
                    chapter_id = tuple(bookmark_data.get('chapter_path', []))
                    chapter_results.append({
                        'chapter_id': chapter_id,
                        'index': int(idx),
                        'distance': float(distances[0][i]),
                        'bookmark_data': bookmark_data
                    })
                    if len(chapter_results) >= top_chapters:
                        break
        
        if not chapter_results:
            # 如果没有找到章，回退到普通检索
            return self.retrieve(query, k=topk)
        
        # 第二步：DFS收集每个章节下的所有内容块
        candidate_block_indices = set()
        chapter_info = []
        
        for chapter_result in chapter_results:
            chapter_id = chapter_result['chapter_id']
            bookmark_data = chapter_result['bookmark_data']
            
            # DFS收集该章节下的所有块
            block_indices = self._collect_blocks_dfs(chapter_id)
            candidate_block_indices.update(block_indices)
            
            chapter_info.append({
                'chapter_title': bookmark_data.get('chapter_title', '未知章节'),
                'chapter_summary': bookmark_data.get('chapter_summary', ''),
                'block_count': len(block_indices)
            })
        
        if not candidate_block_indices:
            return self.retrieve(query, k=topk)
        
        # 第三步：在候选块中检索topk
        candidate_indices_list = list(candidate_block_indices)
        if not self.blocks:
            # 如果没有原始块数据，直接从索引检索
            return self.retrieve(query, k=topk)
        
        # 获取候选块的embeddings（从block_index中）
        # 由于block_index的索引对应block_texts的索引，我们需要找到对应的块
        candidate_embeddings = []
        candidate_block_data = []
        
        for block_idx in candidate_indices_list:
            if block_idx < len(self.blocks):
                block = self.blocks[block_idx]
                search_text = block.get('search_content', block.get('content', ''))
                if search_text.strip():
                    try:
                        embedding = self.embeddings.embed_query(search_text)
                        candidate_embeddings.append(embedding)
                        candidate_block_data.append((block_idx, block))
                    except:
                        continue
        
        if not candidate_embeddings:
            return self.retrieve(query, k=topk)
        
        # 计算相似度
        candidate_embeddings_array = np.array(candidate_embeddings).astype('float32')
        similarities = np.dot(candidate_embeddings_array, query_vector.T).flatten()
        
        # 获取topk
        top_indices = np.argsort(similarities)[-topk:][::-1]
        
        # 格式化结果
        formatted_results = []
        formatted_results.append(f"=== 分层检索结果（DFS） ===\n")
        formatted_results.append(f"查询: {query}\n")
        formatted_results.append(f"找到 {len(chapter_results)} 个相关章节\n")
        
        for i, chapter in enumerate(chapter_info, 1):
            formatted_results.append(f"\n【相关章节 {i}】{chapter['chapter_title']}")
            formatted_results.append(f"章节摘要: {chapter['chapter_summary']}")
            formatted_results.append(f"包含内容块: {chapter['block_count']} 个")
        
        formatted_results.append(f"\n【检索到的相关内容（Top {len(top_indices)}）】\n")
        for rank, top_idx in enumerate(top_indices, 1):
            block_idx, block = candidate_block_data[top_idx]
            content = block.get('content', '')[:500]
            title_path = block.get('metadata', {}).get('title_path', [])
            title_str = ' > '.join(title_path) if title_path else '未知'
            similarity_score = float(similarities[top_idx])
            
            formatted_results.append(
                f"[{rank}] {title_str} (相似度: {similarity_score:.4f})\n"
                f"{content}...\n"
            )
        
        return "\n".join(formatted_results)
    
    def _collect_blocks_dfs(self, chapter_id: tuple) -> List[int]:
        """
        DFS收集章节下的所有内容块索引
        """
        if chapter_id not in self.tree:
            return []
        
        block_indices = []
        node = self.tree[chapter_id]
        
        # 添加当前节点的块
        block_indices.extend(node.get('block_indices', []))
        
        # 递归处理子节点（节）
        for child_id in node.get('children', []):
            child_blocks = self._collect_blocks_dfs(child_id)
            block_indices.extend(child_blocks)
        
        return block_indices


def load_vector_store(db_path: str,
                     embedding_api_url: str,
                     embedding_model: str,
                     openai_api_key: str = "") -> VectorStore:
    """
    便捷函数：加载向量数据库
    
    Args:
        db_path: 向量数据库路径
        embedding_api_url: Embedding API地址
        embedding_model: Embedding模型名称
        openai_api_key: API密钥（可选）
    
    Returns:
        VectorStore实例
    """
    return VectorStore(
        db_path=db_path,
        embedding_model=embedding_model,
        openai_api_base=embedding_api_url,
        openai_api_key=openai_api_key
    )


if __name__ == "__main__":
    builder = VectorDatabaseBuilder(
        embedding_api_url="http://0.0.0.0:30003/v1",
        embedding_model="Qwen3-Embedding-8B"
    )
    
    builder.build_from_processed_blocks(
        processed_blocks_file="data/chunks/chunks.pkl",
        output_dir="data/vector_db"
    )
