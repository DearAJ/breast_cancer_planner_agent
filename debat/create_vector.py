"""
生成向量库脚本
将文献目录中的文本文件转换为向量库并保存到磁盘
"""

from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np
import faiss
import pickle


def load_documents_from_directory(directory: str) -> List[Document]:
    docs = []
    dir_path = Path(directory)
    
    # 使用 rglob 递归查找所有子目录中的 txt 文件
    for txt_file in dir_path.rglob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    # 计算相对路径，保留目录结构信息
                    relative_path = txt_file.relative_to(dir_path)
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": str(txt_file), 
                            "filename": txt_file.name,
                            "relative_path": str(relative_path)
                        }
                    ))
        except Exception as e:
            print(f"读取文件 {txt_file} 时出错: {e}")
    
    return docs


def create_and_save_vector_store(
    source_dir: str,
    save_dir: str,
    embeddings: OpenAIEmbeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    documents = load_documents_from_directory(source_dir)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    doc_splits = text_splitter.split_documents(documents)
    
    doc_texts = [doc.page_content for doc in doc_splits]
    doc_embeddings = embeddings.embed_documents(doc_texts)
    
    # 创建FAISS索引
    dimension = len(doc_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embeddings_array = np.array(doc_embeddings).astype('float32')
    index.add(embeddings_array)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存索引
    faiss.write_index(index, str(save_path / "index.faiss"))
    
    # 保存文档
    with open(save_path / "documents.pkl", 'wb') as f:
        pickle.dump(doc_splits, f)
    
    print(f"向量库已保存到: {save_dir}")


if __name__ == "__main__":
    # 配置
    embedding_api_url = "http://0.0.0.0:30004/v1"
    embedding_model = "Qwen3-Embedding-8B"
    
    # 创建embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=embedding_api_url,
        openai_api_key="",
        timeout=300
    )
    
    project_root = Path(__file__).parent.parent
    source_base = project_root / "breast_cancer_data" / "model_paper" / "ruijin_knowlege"
    save_base = Path(__file__).parent / "paper_vector"
    domains = {
        "21-Gene": "21-Gene",
        "靶向治疗": "靶向治疗",
        "化疗": "化疗",
        "内分泌治疗": "内分泌治疗"
    }
    
    
    for domain_name, dir_name in domains.items():
        source_dir = str(source_base / dir_name)
        save_dir = str(save_base / dir_name)
        
        create_and_save_vector_store(
            source_dir=source_dir,
            save_dir=save_dir,
            embeddings=embeddings
        )
    
    print(f"\n{'='*60}")
    print("所有向量库生成完成！")
