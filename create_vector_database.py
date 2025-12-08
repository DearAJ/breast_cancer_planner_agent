"""
FAISS向量数据库构建脚本
- 从已处理的块文件加载数据
- 生成embeddings
- 构建FAISS索引
"""

import pickle
from typing import List, Dict
from pathlib import Path
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings


class VectorDatabaseBuilder:
    """FAISS向量数据库构建器"""
    
    def __init__(self,
                 embedding_api_url: str = "http://0.0.0.0:30002/v1",
                 embedding_model: str = "Qwen3-Embedding-8B"):
        """
        初始化向量数据库构建器
        
        Args:
            embedding_api_url: Embedding API地址
            embedding_model: Embedding模型名称
        """
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=embedding_api_url,
            openai_api_key="",
            timeout=300
        )
        
        # 存储数据
        self.texts = []
        self.metadatas = []
        self.embeddings_list = []
    
    def build_from_processed_blocks(self, processed_blocks_file: str, output_dir: str = "vector_db"):
        """
        从已处理的块文件构建FAISS向量数据库
        
        Args:
            processed_blocks_file: 已处理的块文件路径（由generate_summaries.py生成）
            output_dir: 输出目录（保存FAISS索引和相关文件）
        """
        processed_path = Path(processed_blocks_file)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not processed_path.exists():
            raise FileNotFoundError(f"处理后的块文件不存在: {processed_blocks_file}")
        
        print(f"加载处理后的块文件: {processed_blocks_file}")
        with open(processed_path, 'rb') as f:
            merged_blocks = pickle.load(f)
        
        print(f"加载了 {len(merged_blocks)} 个块")
        
        # 第一步：生成embeddings
        print(f"\n=== 第一步：生成embeddings ===")
        for i, block in enumerate(merged_blocks):
            if i % 10 == 0:
                print(f"  处理进度: {i}/{len(merged_blocks)}")
            
            # 使用search_content生成embedding（对于表格，这是摘要）
            search_text = block.get('search_content', block.get('content', ''))
            if not search_text.strip():
                continue
            
            try:
                embedding = self.embeddings.embed_query(search_text)
                self.embeddings_list.append(embedding)
                self.texts.append(block.get('content', ''))
                self.metadatas.append(block.get('metadata', {}))
            except Exception as e:
                print(f"  生成embedding时出错 (块 {i}): {e}")
                continue
        
        print(f"\n成功生成 {len(self.embeddings_list)} 个embeddings")
        
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
        
        faiss.write_index(index, str(faiss_index_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadatas, f)
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        
        print(f"\n向量数据库已保存到: {output_path}")
        print(f"  - FAISS索引: {faiss_index_path}")
        print(f"  - 元数据: {metadata_path}")
        print(f"  - 文本内容: {texts_path}")


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
        
        print(f"向量数据库加载完成，包含 {self.index.ntotal} 个向量")
    
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
            title_path = metadata.get('title_path', [])
            title_str = ' > '.join(title_path) if title_path else '未知章节'
            
            formatted_results.append(
                f"【检索结果 {i}】\n"
                f"来源: {title_str}\n"
                f"位置: {metadata.get('position', '未知')}\n"
                f"内容:\n{result['content']}\n"
            )
        
        return "\n\n".join(formatted_results)


def load_vector_store(db_path: str = "med_multi_agent/vector_db",
                     embedding_api_url: str = "http://0.0.0.0:30002/v1",
                     embedding_model: str = "Qwen3-Embedding-8B") -> VectorStore:
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
    # 构建FAISS向量数据库
    builder = VectorDatabaseBuilder(
        embedding_api_url="http://0.0.0.0:30002/v1",
        embedding_model="Qwen3-Embedding-8B"
    )
    
    builder.build_from_processed_blocks(
        processed_blocks_file="med_multi_agent/processed_blocks.pkl",
        output_dir="med_multi_agent/vector_db"
    )
    
    print("\nFAISS向量数据库构建完成！")
    
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
