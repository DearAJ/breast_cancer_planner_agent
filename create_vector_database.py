"""
FAISS向量数据库构建脚本
- 从已处理的块文件加载数据
- 生成embeddings
- 构建FAISS索引
"""

import pickle
import json
from typing import List, Dict, Optional
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
        
        self.texts = []
        self.metadatas = []
        self.embeddings_list = []
    
    def build_from_processed_blocks(self, processed_blocks_file: str, output_dir: str):
        """
        从已处理的块文件构建FAISS向量数据库 输出到output_dir
        """
        processed_path = Path(processed_blocks_file)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not processed_path.exists():
            raise FileNotFoundError(f"处理后的块文件不存在: {processed_blocks_file}")
    
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)
        
        # 兼容新旧格式：新格式是字典，旧格式是列表
        if isinstance(data, dict):
            merged_blocks = data.get('blocks', [])
            bookmarks = data.get('bookmarks', [])
            print(f"加载了 {len(merged_blocks)} 个内容块")
            print(f"加载了 {len(bookmarks)} 个章节书签")
        else:
            # 旧格式：直接是块列表
            merged_blocks = data
            bookmarks = []
            print(f"加载了 {len(merged_blocks)} 个块（旧格式，无书签）")
        
        # 第一步：生成embeddings（先处理书签，再处理内容块）
        print(f"\n=== 第一步：生成embeddings ===")
        
        # 处理章节书签（章节级检索）
        bookmark_indices = []  # 记录书签的索引范围
        if bookmarks:
            print(f"  处理 {len(bookmarks)} 个章节书签...")
            for i, bookmark in enumerate(bookmarks):
                search_text = bookmark.get('search_content', bookmark.get('chapter_summary', ''))
                if not search_text.strip():
                    continue
                
                try:
                    embedding = self.embeddings.embed_query(search_text)
                    self.embeddings_list.append(embedding)
                    # 存储书签的完整信息
                    self.texts.append(json.dumps({
                        'type': 'bookmark',
                        'chapter_title': bookmark.get('chapter_title', ''),
                        'chapter_summary': bookmark.get('chapter_summary', ''),
                        'block_indices': bookmark.get('block_indices', []),
                        'chapter_path': bookmark.get('chapter_path', [])
                    }, ensure_ascii=False))
                    metadata = bookmark.get('metadata', {}).copy()
                    metadata['is_bookmark'] = True
                    metadata['block_type'] = 'chapter_bookmark'
                    self.metadatas.append(metadata)
                    bookmark_indices.append(len(self.embeddings_list) - 1)
                except Exception as e:
                    print(f"  生成书签embedding时出错 (书签 {i}): {e}")
                    continue
        
        # 处理内容块（块级检索）
        block_start_idx = len(self.embeddings_list)
        for i, block in enumerate(merged_blocks):
            if (block_start_idx + i) % 10 == 0:
                print(f"  处理进度: {block_start_idx + i}/{len(merged_blocks) + len(bookmarks)}")
            
            # 使用search_content生成embedding（对于表格，这是摘要）
            search_text = block.get('search_content', block.get('content', ''))
            if not search_text.strip():
                continue
            
            try:
                embedding = self.embeddings.embed_query(search_text)
                self.embeddings_list.append(embedding)
                self.texts.append(block.get('content', ''))
                metadata = block.get('metadata', {}).copy()
                metadata['is_bookmark'] = False
                metadata['block_index'] = i  # 记录原始块索引
                self.metadatas.append(metadata)
            except Exception as e:
                print(f"  生成embedding时出错 (块 {i}): {e}")
                continue
        
        print(f"\n成功生成 {len(self.embeddings_list)} 个embeddings")
        print(f"  - 章节书签: {len(bookmark_indices)} 个")
        print(f"  - 内容块: {len(self.embeddings_list) - len(bookmark_indices)} 个")
        
        # 第二步：构建FAISS索引
        print(f"\n=== 第二步：构建FAISS索引 ===")
        dimension = len(self.embeddings_list[0]) if self.embeddings_list else 0
        if dimension == 0:
            print("错误：没有有效的embeddings")
            return
        
        # 使用L2距离的IndexFlat
        index = faiss.IndexFlatL2(dimension)
        
        # 转换为numpy数组
        embeddings_array = np.array(self.embeddings_list).astype('float32')
        index.add(embeddings_array)
        
        print(f"FAISS索引构建完成，维度: {dimension}, 向量数: {index.ntotal}")
        
        # 第三步：保存索引和元数据
        print(f"\n=== 第三步：保存文件 ===")
        faiss_index_path = output_path / "faiss.index"
        metadata_path = output_path / "metadata.pkl"
        texts_path = output_path / "texts.pkl"
        bookmarks_path = output_path / "bookmarks.pkl"  # 保存书签信息
        blocks_path = output_path / "blocks.pkl"  # 保存原始内容块（用于分层检索）
        
        faiss.write_index(index, str(faiss_index_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadatas, f)
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        with open(bookmarks_path, 'wb') as f:
            pickle.dump(bookmarks, f)
        with open(blocks_path, 'wb') as f:
            pickle.dump(merged_blocks, f)
        
        print(f"\n向量数据库已保存到: {output_path}")
        print(f"  - FAISS索引: {faiss_index_path}")
        print(f"  - 元数据: {metadata_path}")
        print(f"  - 文本内容: {texts_path}")
        print(f"  - 章节书签: {bookmarks_path}")
        print(f"  - 原始内容块: {blocks_path}")


class VectorStore:
    """向量存储检索器（供LiteratureRAGAgent使用）"""
    
    def __init__(self, db_path: str = "vector_db",
                 embedding_api_url: str = "http://0.0.0.0:30002/v1",
                 embedding_model: str = "Qwen3-Embedding-8B"):
        """
        初始化向量存储检索器
        
        Args:
            db_path: 向量数据库路径
            embedding_api_url: Embedding API地址（用于查询时生成embedding）
            embedding_model: Embedding模型名称
        """
        self.db_path = Path(db_path)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=embedding_api_url,
            openai_api_key="",
            timeout=300
        )
        
        # 加载索引和元数据
        self.index = faiss.read_index(str(self.db_path / "faiss.index"))
        with open(self.db_path / "metadata.pkl", 'rb') as f:
            self.metadatas = pickle.load(f)
        with open(self.db_path / "texts.pkl", 'rb') as f:
            self.texts = pickle.load(f)
        
        # 加载书签和原始块（用于分层检索）
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
        
        print(f"向量数据库加载完成，包含 {self.index.ntotal} 个向量")
        if self.bookmarks:
            print(f"  - 章节书签: {len(self.bookmarks)} 个")
        if self.blocks:
            print(f"  - 内容块: {len(self.blocks)} 个")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        检索相似文档
        
        Args:
            query: 查询文本
            k: 返回前k个最相似的结果
        
        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 搜索
        distances, indices = self.index.search(query_vector, k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append({
                    'content': self.texts[idx],
                    'metadata': self.metadatas[idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def retrieve(self, query: str, k: int = 5) -> str:
        """
        检索并格式化结果（供LiteratureRAGAgent使用）
        
        Args:
            query: 查询文本
            k: 返回前k个最相似的结果
        
        Returns:
            格式化后的检索结果字符串
        """
        results = self.search(query, k)
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            
            # 处理书签
            if metadata.get('is_bookmark', False):
                bookmark_data = json.loads(result['content'])
                title_str = bookmark_data.get('chapter_title', '未知章节')
                formatted_results.append(
                    f"【章节书签 {i}】\n"
                    f"章节: {title_str}\n"
                    f"摘要: {bookmark_data.get('chapter_summary', '')}\n"
                )
            else:
                # 处理内容块
                title_path = metadata.get('title_path', [])
                title_str = ' > '.join(title_path) if title_path else '未知章节'
                formatted_results.append(
                    f"【检索结果 {i}】\n"
                    f"来源: {title_str}\n"
                    f"位置: {metadata.get('position', '未知')}\n"
                    f"内容:\n{result['content']}\n"
                )
        
        return "\n\n".join(formatted_results)
    
    def hierarchical_search(self, query: str, top_chapters: int = 3, chunks_per_chapter: int = 3) -> str:
        """
        分层检索：先检索相关章节，再检索章节内的具体内容
        
        Args:
            query: 查询文本
            top_chapters: 返回前top_chapters个最相关的章节
            chunks_per_chapter: 每个章节返回前chunks_per_chapter个最相关的内容块
        
        Returns:
            格式化后的分层检索结果字符串
        """
        # 第一步：检索相关章节（只检索书签）
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 搜索所有向量（包括书签和内容块）
        distances, indices = self.index.search(query_vector, min(100, self.index.ntotal))
        
        # 筛选出书签结果
        chapter_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadatas):
                metadata = self.metadatas[idx]
                if metadata.get('is_bookmark', False):
                    chapter_results.append({
                        'index': int(idx),
                        'distance': float(distances[0][i])
                    })
                    if len(chapter_results) >= top_chapters:
                        break
        
        if not chapter_results:
            # 如果没有找到书签，回退到普通检索
            return self.retrieve(query, k=5)
        
        # 第二步：对每个相关章节，检索其内的具体内容块
        formatted_results = []
        formatted_results.append(f"=== 分层检索结果 ===\n")
        formatted_results.append(f"查询: {query}\n")
        formatted_results.append(f"找到 {len(chapter_results)} 个相关章节\n")
        
        for chapter_idx, chapter_result in enumerate(chapter_results, 1):
            bookmark_data = json.loads(self.texts[chapter_result['index']])
            chapter_title = bookmark_data.get('chapter_title', '未知章节')
            block_indices = bookmark_data.get('block_indices', [])
            
            formatted_results.append(f"\n【相关章节 {chapter_idx}】{chapter_title}")
            formatted_results.append(f"章节摘要: {bookmark_data.get('chapter_summary', '')}")
            
            if block_indices and self.blocks:
                # 为该章节内的块生成embeddings并检索
                chapter_block_contents = []
                chapter_block_embeddings = []
                chapter_block_original_indices = []
                
                for block_idx in block_indices[:50]:  # 限制每个章节最多处理50个块
                    if block_idx < len(self.blocks):
                        block = self.blocks[block_idx]
                        content = block.get('search_content', block.get('content', ''))
                        if content.strip():
                            try:
                                embedding = self.embeddings.embed_query(content)
                                chapter_block_contents.append(content)
                                chapter_block_embeddings.append(embedding)
                                chapter_block_original_indices.append(block_idx)
                            except:
                                continue
                
                if chapter_block_embeddings:
                    # 计算查询与章节内块的相似度
                    chapter_embeddings_array = np.array(chapter_block_embeddings).astype('float32')
                    similarities = np.dot(chapter_embeddings_array, query_vector.T).flatten()
                    
                    # 获取最相关的几个块
                    top_indices = np.argsort(similarities)[-chunks_per_chapter:][::-1]
                    
                    formatted_results.append(f"\n  章节内相关内容（{len(top_indices)} 个块）:")
                    for rank, top_idx in enumerate(top_indices, 1):
                        block_idx = chapter_block_original_indices[top_idx]
                        block = self.blocks[block_idx]
                        content = block.get('content', '')[:500]  # 限制显示长度
                        title_path = block.get('metadata', {}).get('title_path', [])
                        title_str = ' > '.join(title_path) if title_path else '未知'
                        
                        formatted_results.append(
                            f"    [{rank}] {title_str}\n"
                            f"    {content}...\n"
                        )
            else:
                formatted_results.append("  （该章节下无内容块）")
        
        return "\n".join(formatted_results)


def load_vector_store(db_path: str,
                     embedding_api_url,
                     embedding_model) -> VectorStore:
    """
    便捷函数：加载向量数据库
    
    Args:
        db_path: 向量数据库路径
        embedding_api_url: Embedding API地址
        embedding_model: Embedding模型名称
    
    Returns:
        VectorStore实例
    """
    return VectorStore(
        db_path=db_path,
        embedding_api_url=embedding_api_url,
        embedding_model=embedding_model
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
    
    # 使用示例：
    # from create_vector_database import load_vector_store
    # from basic import create_medical_agent_graph
    # 
    # # 加载向量数据库
    # literature_vector_store = load_vector_store()
    # 
    # # 创建智能体图
    # graph = create_medical_agent_graph(
    #     literature_vector_store=literature_vector_store
    # )
