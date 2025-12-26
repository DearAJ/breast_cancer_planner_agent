"""
病例向量数据库构建脚本
- 从train_cases.json加载病例数据
- 将病例转换为可检索的文本格式
- 生成embeddings
- 使用FAISS构建索引
- 支持相似度检索
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings


def format_case_to_text(case_id: str, case_data: Dict) -> str:
    """
    将病例数据转换为可检索的文本格式
    
    Args:
        case_id: 病例ID
        case_data: 病例数据字典
    
    Returns:
        格式化的文本字符串
    """
    basic_info = case_data.get("基本信息", {})
    treatment = case_data.get("治疗方案表单", {})
    
    # 提取关键信息
    text_parts = []
    
    # 基本信息部分
    text_parts.append(f"病例ID: {case_id}")
    
    # 患者基本信息
    if basic_info.get("年龄"):
        text_parts.append(f"年龄: {basic_info['年龄']}岁")
    if basic_info.get("性别"):
        text_parts.append(f"性别: {basic_info['性别']}")
    if basic_info.get("LMP"):
        text_parts.append(f"月经状态: {basic_info['LMP']}")
    if basic_info.get("左右侧"):
        text_parts.append(f"病变位置: {basic_info['左右侧']}")
    
    # 病理信息
    if basic_info.get("病理类型T1"):
        text_parts.append(f"病理类型: {basic_info['病理类型T1']}")
    if basic_info.get("病理类型T2"):
        text_parts.append(f"病理类型T2: {basic_info['病理类型T2']}")
    if basic_info.get("病理分期"):
        text_parts.append(f"病理分期: {basic_info['病理分期']}")
    if basic_info.get("Grade-T1"):
        text_parts.append(f"组织学分级: {basic_info['Grade-T1']}")
    if basic_info.get("肿块大小pT1"):
        text_parts.append(f"肿块大小: {basic_info['肿块大小pT1']}")
    if basic_info.get("肿块象限T1"):
        text_parts.append(f"肿块象限: {basic_info['肿块象限T1']}")
    
    # 淋巴结状态
    if basic_info.get("SLN转移数") is not None and basic_info.get("SLN活检数") is not None:
        sln_status = f"前哨淋巴结: {basic_info['SLN转移数']}/{basic_info['SLN活检数']}"
        text_parts.append(sln_status)
    if basic_info.get("ALN转移数") is not None and basic_info.get("ALN清扫数") is not None:
        aln_status = f"腋窝淋巴结: {basic_info['ALN转移数']}/{basic_info['ALN清扫数']}"
        text_parts.append(aln_status)
    if basic_info.get("总LN转移数") is not None and basic_info.get("总LN手术数") is not None:
        total_ln_status = f"总淋巴结: {basic_info['总LN转移数']}/{basic_info['总LN手术数']}"
        text_parts.append(total_ln_status)
    
    # 免疫组化指标
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
        text_parts.append(f"免疫组化: {', '.join(ihc_parts)}")
    
    # 手术信息
    if basic_info.get("手术方式"):
        text_parts.append(f"手术方式: {basic_info['手术方式']}")
    if basic_info.get("保乳切缘"):
        text_parts.append(f"保乳切缘: {basic_info['保乳切缘']}")
    
    # 其他重要信息
    if basic_info.get("21-基因RS"):
        text_parts.append(f"21-基因RS: {basic_info['21-基因RS']}")
    if basic_info.get("脉管癌栓"):
        text_parts.append(f"脉管癌栓: {basic_info['脉管癌栓']}")
    if basic_info.get("备注"):
        text_parts.append(f"备注: {basic_info['备注']}")
    
    # 治疗方案
    treatment_parts = []
    if treatment.get("辅助化疗方案"):
        treatment_parts.append(f"化疗: {treatment['辅助化疗方案']}")
    if treatment.get("辅助放疗方案"):
        treatment_parts.append(f"放疗: {treatment['辅助放疗方案']}")
    if treatment.get("辅助内分泌治疗方案"):
        treatment_parts.append(f"内分泌: {treatment['辅助内分泌治疗方案']}")
    if treatment.get("辅助靶向治疗方案"):
        treatment_parts.append(f"靶向: {treatment['辅助靶向治疗方案']}")
    if treatment_parts:
        text_parts.append(f"治疗方案: {', '.join(treatment_parts)}")
    
    return "\n".join(text_parts)


class CaseVectorDatabaseBuilder:
    """病例向量数据库构建器"""
    
    def __init__(self, embedding_api_url: str, embedding_model: str):
        """
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
    
    def build_from_cases(self, cases_file: str, output_dir: str):
        """
        从病例JSON文件构建FAISS向量数据库
        
        Args:
            cases_file: 病例JSON文件路径
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载病例数据
        print(f"正在加载病例数据: {cases_file}")
        with open(cases_file, 'r', encoding='utf-8') as f:
            cases_data = json.load(f)
        
        total_cases = len(cases_data)
        print(f"总病例数: {total_cases}")
        
        # 处理病例，生成文本和embeddings
        print("\n正在生成embeddings...")
        case_texts = []
        case_embeddings = []
        case_metadatas = []
        
        for idx, (case_id, case_data) in enumerate(cases_data.items()):
            if (idx + 1) % 100 == 0:
                print(f"  处理进度: {idx + 1}/{total_cases}")
            
            # 格式化病例为文本
            case_text = format_case_to_text(case_id, case_data)
            if not case_text.strip():
                continue
            
            try:
                # 生成embedding
                embedding = self.embeddings.embed_query(case_text)
                case_embeddings.append(embedding)
                case_texts.append(case_text)
                
                # 保存元数据
                metadata = {
                    'case_id': case_id,
                    'basic_info': case_data.get("基本信息", {}),
                    'treatment': case_data.get("治疗方案表单", {})
                }
                case_metadatas.append(metadata)
                
            except Exception as e:
                print(f"  生成病例 {case_id} 的embedding时出错: {e}")
                continue
        
        print(f"成功处理 {len(case_embeddings)} 个病例")
        
        # 构建FAISS索引
        print("\n正在构建FAISS索引...")
        if not case_embeddings:
            raise ValueError("没有可用的embeddings")
        
        dimension = len(case_embeddings[0])
        embeddings_array = np.array(case_embeddings).astype('float32')
        
        # 使用L2距离的Flat索引（适合小到中等规模数据）
        # 对于大规模数据，可以使用IndexIVFFlat或IndexHNSWFlat
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        print(f"FAISS索引构建完成: {index.ntotal} 个向量")
        
        # 保存索引和元数据
        print("\n正在保存索引和元数据...")
        faiss.write_index(index, str(output_path / "case_index.faiss"))
        
        with open(output_path / "case_texts.pkl", 'wb') as f:
            pickle.dump(case_texts, f)
        
        with open(output_path / "case_metadatas.pkl", 'wb') as f:
            pickle.dump(case_metadatas, f)
        
        # 保存原始病例数据（用于检索时返回完整信息）
        with open(output_path / "cases_data.pkl", 'wb') as f:
            pickle.dump(cases_data, f)
        
        print(f"\n✓ 病例向量数据库构建完成")
        print(f"  输出目录: {output_dir}")
        print(f"  索引文件: {output_path / 'case_index.faiss'}")
        print(f"  文本文件: {output_path / 'case_texts.pkl'}")
        print(f"  元数据文件: {output_path / 'case_metadatas.pkl'}")


if __name__ == "__main__":
    builder = CaseVectorDatabaseBuilder(
        embedding_api_url="http://0.0.0.0:30004/v1",
        embedding_model="Qwen3-Embedding-8B"
    )
    
    builder.build_from_cases(
        cases_file="/data/aj/RAG/breast_cancer_planner_agent/data/raw/train_cases.json",
        output_dir="/data/aj/RAG/breast_cancer_planner_agent/data/case_vector_db"
    )

