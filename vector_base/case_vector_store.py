"""
病例向量存储检索器
- 加载FAISS索引
- 支持相似度检索
- 返回格式化的病例信息
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings


class CaseVectorStore:
    """病例向量存储检索器"""
    
    def __init__(self, db_path: str, embedding_api_url: str, embedding_model: str, openai_api_key: str = ""):
        """
        Args:
            db_path: 向量数据库路径
            embedding_api_url: Embedding API地址
            embedding_model: Embedding模型名称
            openai_api_key: API密钥（可选）
        """
        self.db_path = Path(db_path)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=embedding_api_url,
            openai_api_key=openai_api_key,
            timeout=300
        )
        
        # 加载FAISS索引
        index_path = self.db_path / "case_index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f"加载病例索引: {self.index.ntotal} 个向量")
        
        # 加载文本和元数据
        with open(self.db_path / "case_texts.pkl", 'rb') as f:
            self.case_texts = pickle.load(f)
        
        with open(self.db_path / "case_metadatas.pkl", 'rb') as f:
            self.case_metadatas = pickle.load(f)
        
        # 加载原始病例数据（可选）
        cases_data_path = self.db_path / "cases_data.pkl"
        if cases_data_path.exists():
            with open(cases_data_path, 'rb') as f:
                self.cases_data = pickle.load(f)
        else:
            self.cases_data = None
        
        print(f"病例向量数据库加载完成")
    
    def retrieve(self, query: str, k: int = 5) -> str:
        """
        检索相似病例
        
        Args:
            query: 查询文本
            k: 返回top-k个结果
        
        Returns:
            格式化的检索结果字符串
        """
        # 生成查询embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # FAISS检索（返回L2距离）
        distances, indices = self.index.search(query_vector, k)
        
        # 格式化结果
        results = []
        results.append(f"=== 相似病例检索结果 ===\n")
        results.append(f"查询: {query}\n")
        results.append(f"找到 {len(indices[0])} 个相似病例\n")
        
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            if idx >= len(self.case_metadatas):
                continue
            
            metadata = self.case_metadatas[idx]
            case_id = metadata.get('case_id', '未知')
            basic_info = metadata.get('basic_info', {})
            treatment = metadata.get('treatment', {})
            
            # 计算相似度分数（L2距离转换为相似度，距离越小相似度越高）
            # 使用简单的转换：similarity = 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + float(distance))
            
            results.append(f"\n【相似病例 {rank}】")
            results.append(f"病例ID: {case_id}")
            results.append(f"相似度分数: {similarity_score:.4f} (L2距离: {distance:.4f})")
            
            # 关键信息
            if basic_info.get("年龄"):
                results.append(f"年龄: {basic_info['年龄']}岁")
            if basic_info.get("LMP"):
                results.append(f"月经状态: {basic_info['LMP']}")
            if basic_info.get("病理类型T1"):
                results.append(f"病理类型: {basic_info['病理类型T1']}")
            if basic_info.get("病理分期"):
                results.append(f"病理分期: {basic_info['病理分期']}")
            
            # 免疫组化
            ihc_parts = []
            if basic_info.get("ER(%)(T1)"):
                ihc_parts.append(f"ER: {basic_info['ER(%)(T1)']}")
            if basic_info.get("PR(%)(T1)"):
                ihc_parts.append(f"PR: {basic_info['PR(%)(T1)']}")
            if basic_info.get("CerbB-2(T1)"):
                ihc_parts.append(f"HER2: {basic_info['CerbB-2(T1)']}")
            if basic_info.get("Ki67(%)(T1)"):
                ihc_parts.append(f"Ki67: {basic_info['Ki67(%)(T1)']}")
            if ihc_parts:
                results.append(f"免疫组化: {', '.join(ihc_parts)}")
            
            # 淋巴结状态
            if basic_info.get("SLN转移数") is not None and basic_info.get("SLN活检数") is not None:
                results.append(f"前哨淋巴结: {basic_info['SLN转移数']}/{basic_info['SLN活检数']}")
            if basic_info.get("ALN转移数") is not None and basic_info.get("ALN清扫数") is not None:
                results.append(f"腋窝淋巴结: {basic_info['ALN转移数']}/{basic_info['ALN清扫数']}")
            
            # 治疗方案
            if treatment.get("辅助化疗方案"):
                results.append(f"辅助化疗方案: {treatment['辅助化疗方案']}")
            if treatment.get("辅助放疗方案"):
                results.append(f"辅助放疗方案: {treatment['辅助放疗方案']}")
            if treatment.get("辅助内分泌治疗方案"):
                results.append(f"辅助内分泌治疗方案: {treatment['辅助内分泌治疗方案']}")
            if treatment.get("辅助靶向治疗方案"):
                results.append(f"辅助靶向治疗方案: {treatment['辅助靶向治疗方案']}")
            
            # 手术信息
            if basic_info.get("手术方式"):
                results.append(f"手术方式: {basic_info['手术方式']}")
        
        return "\n".join(results)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        检索相似病例，返回结构化结果
        
        Args:
            query: 查询文本
            k: 返回top-k个结果
        
        Returns:
            包含相似病例信息的字典列表
        """
        # 生成查询embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # FAISS检索
        distances, indices = self.index.search(query_vector, k)
        
        # 构建结果列表
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(self.case_metadatas):
                continue
            
            metadata = self.case_metadatas[idx]
            similarity_score = 1.0 / (1.0 + float(distance))
            
            result = {
                'case_id': metadata.get('case_id', '未知'),
                'similarity': similarity_score,
                'distance': float(distance),
                'basic_info': metadata.get('basic_info', {}),
                'treatment': metadata.get('treatment', {}),
                'text': self.case_texts[idx] if idx < len(self.case_texts) else ''
            }
            
            # 如果有原始病例数据，添加完整信息
            if self.cases_data and result['case_id'] in self.cases_data:
                result['full_case'] = self.cases_data[result['case_id']]
            
            results.append(result)
        
        return results


def load_case_vector_store(db_path: str,
                          embedding_api_url: str,
                          embedding_model: str,
                          openai_api_key: str = "") -> CaseVectorStore:
    """
    便捷函数：加载病例向量数据库
    
    Args:
        db_path: 向量数据库路径
        embedding_api_url: Embedding API地址
        embedding_model: Embedding模型名称
        openai_api_key: API密钥（可选）
    
    Returns:
        CaseVectorStore实例
    """
    return CaseVectorStore(
        db_path=db_path,
        embedding_api_url=embedding_api_url,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key
    )

