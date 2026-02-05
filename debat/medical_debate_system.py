"""
医疗多智能体辩论系统
- 四个RAG智能体基于不同文献库生成判断
- 第五个智能体组织辩论，最多3轮
- 最终生成一致的回答
"""

import json
import operator
import re
from pathlib import Path
from typing import TypedDict, List, Annotated, Dict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import numpy as np
import faiss
import pickle


def clean_think_tags(text: str) -> str:
    """移除文本中的<think>...</think>标签及其内容"""
    if not text:
        return text
    # 移除<think>...</think>标签及其内容（包括多行）
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned.strip()


def create_local_llm(
    base_url: str = "http://0.0.0.0:30003/v1",
    model_name: str = "qwen3-8B",
    temperature: float = 0.0,
    timeout: int = 300
) -> ChatOpenAI:
    """创建本地LLM客户端"""
    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        api_key=""
    )


def create_embeddings(
    embedding_api_url: str = "http://0.0.0.0:30004/v1",
    embedding_model: str = "Qwen3-Embedding-8B",
    openai_api_key: str = ""
) -> OpenAIEmbeddings:
    """创建Embedding客户端"""
    return OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=embedding_api_url,
        openai_api_key=openai_api_key,
        timeout=300
    )


class SimpleVectorStore:
    """简单的向量存储，用于RAG检索"""
    
    def __init__(self, load_path: str, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        load_dir = Path(load_path)
        
        # 加载FAISS索引
        index_path = load_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"向量库索引不存在: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # 加载文档
        docs_path = load_dir / "documents.pkl"
        if not docs_path.exists():
            raise FileNotFoundError(f"向量库文档不存在: {docs_path}")
        
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"向量库已加载: {load_path} ({self.index.ntotal} 个向量)")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """检索相似文档"""
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results




# ==================== 状态定义 ====================
class DebateState(TypedDict):
    """辩论状态"""
    question: str  # 用户问题
    agent_21gene_answer: str  # 21-Gene智能体的回答
    agent_targeted_answer: str  # 靶向治疗智能体的回答
    agent_chemotherapy_answer: str  # 化疗智能体的回答
    agent_endocrine_answer: str  # 内分泌治疗智能体的回答
    debate_round: Annotated[int, operator.add]  # 辩论轮数
    debate_history: List[str]  # 辩论历史
    consensus_reached: bool  # 是否达成一致
    final_answer: str  # 最终回答


# 智能体配置映射
AGENT_CONFIG = {
    "21-Gene专家": ("agent_21gene", "agent_21gene_answer"),
    "靶向治疗专家": ("agent_targeted", "agent_targeted_answer"),
    "化疗专家": ("agent_chemotherapy", "agent_chemotherapy_answer"),
    "内分泌治疗专家": ("agent_endocrine", "agent_endocrine_answer"),
}


# ==================== RAG智能体 ====================
class RAGAgent:
    """RAG智能体基类"""
    
    def __init__(self, llm: ChatOpenAI, vector_store: SimpleVectorStore, agent_name: str, domain: str):
        self.llm = llm
        self.vector_store = vector_store
        self.agent_name = agent_name
        self.domain = domain
        self.system_prompt = f"""你是一位专注于{domain}领域的乳腺癌诊疗专家。
你的任务是基于你专业领域的文献和研究，回答关于乳腺癌诊疗的问题。
请基于检索到的文献证据，提供专业、准确、有依据的回答。
如果检索到的文献不足以回答问题，请明确说明。"""
    
    def generate_answer(self, question: str) -> str:
        """基于RAG生成回答"""
        print(f"\n【{self.agent_name}】开始生成回答...")
        
        # 检索相关文档
        if self.vector_store:
            retrieved_docs = self.vector_store.retrieve(question, k=5)
            context = "\n\n".join([doc.page_content[:1000] for doc in retrieved_docs])  # 限制长度
        else:
            context = "未找到相关文献"
            retrieved_docs = []
        
        print(f"检索到 {len(retrieved_docs)} 个相关文档")
        
        # 构建RAG提示
        rag_prompt = f"""基于以下{self.domain}领域的文献证据，回答用户的问题。

【检索到的文献内容】
{context[:3000]}  # 限制总长度

【用户问题】
{question}

请基于上述文献证据，提供专业、准确、有依据的回答。如果文献证据不足，请明确说明。
回答要简洁明了，重点突出关键信息。"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=rag_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            answer = response.content
            # 清理<think>标签
            answer = clean_think_tags(answer)
            print(f"【{self.agent_name}】回答生成完成")
            return answer
        except Exception as e:
            print(f"【{self.agent_name}】生成回答时出错: {e}")
            return f"抱歉，{self.agent_name}在生成回答时遇到错误。"


# ==================== 辩论组织智能体 ====================
class DebateModeratorAgent:
    """辩论组织智能体"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.system_prompt = """你是一位资深的医疗辩论组织专家。
你的任务是组织多个医疗专家进行辩论，帮助他们达成一致意见。

你的职责：
1. 分析各个专家的观点和依据
2. 识别观点之间的差异和共同点
3. 引导专家进行深入讨论
4. 判断是否达成一致意见
5. 如果达成一致，生成最终的综合回答

请保持客观、专业，确保最终回答基于所有专家的共识。"""
    
    def check_consensus(self, state: DebateState) -> Dict:
        """检查是否达成一致意见"""
        print("\n【辩论组织者】检查是否达成一致意见...")
        
        answers = {agent_name: state.get(answer_key, "") 
                   for agent_name, (_, answer_key) in AGENT_CONFIG.items()}
        
        debate_round = state.get("debate_round", 0)
        debate_history = state.get("debate_history", [])
        
        consensus_prompt = f"""请分析以下四位专家的观点，判断是否达成一致意见。

【用户问题】
{state['question']}

【专家观点】
21-Gene专家: {answers['21-Gene专家']}

靶向治疗专家: {answers['靶向治疗专家']}

化疗专家: {answers['化疗专家']}

内分泌治疗专家: {answers['内分泌治疗专家']}

【辩论历史】
{chr(10).join(debate_history) if debate_history else '无'}

【当前轮次】
第 {debate_round} 轮辩论

请以JSON格式返回：
{{
    "consensus_reached": true/false,
    "reason": "达成一致或未达成一致的原因",
    "key_differences": ["差异点1", "差异点2", ...],
    "common_points": ["共同点1", "共同点2", ...],
    "next_guidance": "如果需要继续辩论，请提供引导性问题或建议"
}}

注意：如果这是第3轮辩论，必须达成一致。"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=consensus_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            # 检查content是否为None或空
            if content is None:
                raise ValueError("LLM返回的content为None")
            if not content:
                raise ValueError("LLM返回的content为空字符串")
            
            # 先清理<think>标签
            content = clean_think_tags(content)
            
            # 解析JSON
            json_str = None
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                json_str = json_match.group(0) if json_match else content.strip()
            
            if not json_str or not json_str.strip():
                raise ValueError(f"提取的JSON字符串为空，清理后的内容: {repr(content[:200])}")
            
            result = json.loads(json_str)
            
            consensus_reached = result.get("consensus_reached", False)
            if debate_round >= 3:
                consensus_reached = True  # 第3轮强制达成一致
            
            print(f"共识检查结果: {'达成一致' if consensus_reached else '未达成一致'}")
            return {
                "consensus_reached": consensus_reached,
                "consensus_analysis": result
            }
        except Exception as e:
            print(f"检查共识时出错: {e}")
            # 第3轮强制达成一致
            if debate_round >= 3:
                return {
                    "consensus_reached": True,
                    "consensus_analysis": {"reason": "已达到最大辩论轮次"}
                }
            return {
                "consensus_reached": False,
                "consensus_analysis": {"reason": f"分析出错: {e}"}
            }
    
    def generate_final_answer(self, state: DebateState) -> str:
        """生成最终回答"""
        print("\n【辩论组织者】生成最终回答...")
        
        answers = {agent_name: state.get(answer_key, "") 
                   for agent_name, (_, answer_key) in AGENT_CONFIG.items()}
        
        consensus_analysis = state.get("consensus_analysis", {})
        
        final_prompt = f"""基于四位专家的辩论结果，生成最终的综合回答。

【用户问题】
{state['question']}

【各专家观点】
21-Gene专家: {answers['21-Gene专家']}

靶向治疗专家: {answers['靶向治疗专家']}

化疗专家: {answers['化疗专家']}

内分泌治疗专家: {answers['内分泌治疗专家']}

【共识分析】
{json.dumps(consensus_analysis, ensure_ascii=False, indent=2)}

【辩论历史】
{chr(10).join(state.get('debate_history', []))}

请生成一个综合、专业、基于共识的最终回答。回答应该：
1. 综合所有专家的观点
2. 突出达成共识的部分
3. 如果存在分歧，说明不同观点的依据和适用情况
4. 提供明确的诊疗建议（如果可能）
5. 说明回答的依据和局限性"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=final_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            final_answer = response.content
            # 清理<think>标签
            final_answer = clean_think_tags(final_answer)
            print("最终回答生成完成")
            return final_answer
        except Exception as e:
            print(f"生成最终回答时出错: {e}")
            return "抱歉，生成最终回答时遇到错误。"


# 全局变量存储智能体（用于节点函数访问）
_agents = None


def set_agents(agents_dict):
    """设置全局智能体"""
    global _agents
    _agents = agents_dict


# ==================== 节点函数 ====================
def _build_debate_context(state: DebateState, current_agent_name: str) -> str:
    """构建包含其他智能体观点的辩论上下文"""
    context_parts = []
    
    # 获取其他智能体的观点
    other_views = []
    for agent_name, (_, answer_key) in AGENT_CONFIG.items():
        if agent_name != current_agent_name:
            view = state.get(answer_key, "")
            if view:
                other_views.append(f"{agent_name}: {view}")
    
    if other_views:
        context_parts.extend(["【其他专家的观点】"] + other_views + [""])
    
    # 添加辩论历史
    debate_history = state.get("debate_history", [])
    if debate_history:
        context_parts.extend(["【之前的辩论历史】"] + debate_history[-2:] + [""])
    
    return "\n".join(context_parts)


def _generate_agent_answer(state: DebateState, agent_name: str) -> Dict[str, str]:
    """通用函数：生成智能体回答"""
    global _agents
    if _agents is None:
        return {AGENT_CONFIG[agent_name][1]: "智能体未初始化"}
    
    agent_key, answer_key = AGENT_CONFIG[agent_name]
    agent = _agents[agent_key]
    question = state["question"]
    
    debate_context = _build_debate_context(state, agent_name)
    if debate_context:
        question_with_context = f"{question}\n\n{debate_context}\n请基于其他专家的观点和辩论历史，重新考虑你的回答。你可以：\n1. 回应其他专家的观点\n2. 补充或修正你的观点\n3. 寻找共识点"
    else:
        question_with_context = question
    
    answer = agent.generate_answer(question_with_context)
    return {answer_key: answer}


def generate_21gene_answer(state: DebateState) -> DebateState:
    """21-Gene智能体生成回答"""
    return _generate_agent_answer(state, "21-Gene专家")


def generate_targeted_answer(state: DebateState) -> DebateState:
    """靶向治疗智能体生成回答"""
    return _generate_agent_answer(state, "靶向治疗专家")


def generate_chemotherapy_answer(state: DebateState) -> DebateState:
    """化疗智能体生成回答"""
    return _generate_agent_answer(state, "化疗专家")


def generate_endocrine_answer(state: DebateState) -> DebateState:
    """内分泌治疗智能体生成回答"""
    return _generate_agent_answer(state, "内分泌治疗专家")


def moderate_debate(state: DebateState) -> DebateState:
    """组织辩论"""
    global _agents
    if _agents is None:
        return {"consensus_reached": True, "debate_round": state.get("debate_round", 0) + 1}
    
    moderator = _agents["moderator"]
    debate_round = state.get("debate_round", 0)
    
    # 检查共识
    consensus_result = moderator.check_consensus(state)
    
    consensus_reached = consensus_result["consensus_reached"]
    consensus_analysis = consensus_result["consensus_analysis"]
    
    # 更新辩论历史，包含各智能体的观点
    debate_history = state.get("debate_history", [])
    round_summary = f"第{debate_round + 1}轮辩论: {consensus_analysis.get('reason', '')}"
    if consensus_analysis.get('key_differences'):
        round_summary += f"\n主要差异: {', '.join(consensus_analysis['key_differences'])}"
    if consensus_analysis.get('common_points'):
        round_summary += f"\n共同点: {', '.join(consensus_analysis['common_points'])}"
    
    # 添加各智能体的观点到辩论历史
    round_summary += "\n\n【各专家观点】"
    for agent_name, (_, answer_key) in AGENT_CONFIG.items():
        answer = state.get(answer_key, "")
        if answer:
            round_summary += f"\n{agent_name}: {answer[:200]}..."
    
    debate_history.append(round_summary)
    
    # 第3轮强制达成一致
    if debate_round >= 2:
        consensus_reached = True
    
    return {
        "consensus_reached": consensus_reached,
        "consensus_analysis": consensus_analysis,
        "debate_history": debate_history,
        "debate_round": debate_round + 1
    }


def generate_final_answer(state: DebateState) -> DebateState:
    """生成最终回答"""
    global _agents
    if _agents is None:
        return {"final_answer": "智能体未初始化"}
    moderator = _agents["moderator"]
    final_answer = moderator.generate_final_answer(state)
    return {"final_answer": final_answer}


def should_continue_debate(state: DebateState) -> str:
    """判断是否继续辩论"""
    consensus_reached = state.get("consensus_reached", False)
    debate_round = state.get("debate_round", 0)
    
    if consensus_reached or debate_round >= 3:
        return "end"
    else:
        # 继续辩论，需要重新生成四个智能体的回答
        return "continue_debate"


# ==================== 构建工作流 ====================
def create_debate_graph(
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    vector_base_dir: str = None
):
    """创建辩论图"""
    
    # 设置向量库路径
    if vector_base_dir is None:
        vector_base_dir = str(Path(__file__).parent / "paper_vector")
    
    print("正在加载向量库...")
    
    # 加载向量库
    vector_stores = {
        "21-Gene": SimpleVectorStore(str(Path(vector_base_dir) / "21-Gene"), embeddings),
        "靶向治疗": SimpleVectorStore(str(Path(vector_base_dir) / "靶向治疗"), embeddings),
        "化疗": SimpleVectorStore(str(Path(vector_base_dir) / "化疗"), embeddings),
        "内分泌治疗": SimpleVectorStore(str(Path(vector_base_dir) / "内分泌治疗"), embeddings),
    }
    
    # 创建RAG智能体
    agent_configs = [
        ("21-Gene专家", "21基因检测", "21-Gene"),
        ("靶向治疗专家", "靶向治疗", "靶向治疗"),
        ("化疗专家", "化疗", "化疗"),
        ("内分泌治疗专家", "内分泌治疗", "内分泌治疗"),
    ]
    agents = {}
    for agent_name, domain, vector_key in agent_configs:
        agent_key = AGENT_CONFIG[agent_name][0]
        agents[agent_key] = RAGAgent(llm, vector_stores[vector_key], agent_name, domain)
    
    agent_21gene = agents["agent_21gene"]
    agent_targeted = agents["agent_targeted"]
    agent_chemotherapy = agents["agent_chemotherapy"]
    agent_endocrine = agents["agent_endocrine"]
    
    # 创建辩论组织者
    moderator = DebateModeratorAgent(llm)
    
    # 创建状态图
    workflow = StateGraph(DebateState)
    
    # 添加节点
    workflow.add_node("generate_21gene", generate_21gene_answer)
    workflow.add_node("generate_targeted", generate_targeted_answer)
    workflow.add_node("generate_chemotherapy", generate_chemotherapy_answer)
    workflow.add_node("generate_endocrine", generate_endocrine_answer)
    workflow.add_node("moderate", moderate_debate)
    workflow.add_node("final_answer", generate_final_answer)
    
    # 设置入口点：四个RAG智能体并行执行
    workflow.set_entry_point("generate_21gene")
    
    # 四个智能体并行执行
    workflow.add_edge("generate_21gene", "generate_targeted")
    workflow.add_edge("generate_targeted", "generate_chemotherapy")
    workflow.add_edge("generate_chemotherapy", "generate_endocrine")
    workflow.add_edge("generate_endocrine", "moderate")
    
    # 辩论组织者判断是否继续
    workflow.add_conditional_edges(
        "moderate",
        should_continue_debate,
        {
            "end": "final_answer",
            "continue_debate": "generate_21gene"  # 继续辩论，重新生成四个智能体的回答
        }
    )
    
    workflow.add_edge("final_answer", END)
    
    # 设置全局智能体
    set_agents({
        "agent_21gene": agent_21gene,
        "agent_targeted": agent_targeted,
        "agent_chemotherapy": agent_chemotherapy,
        "agent_endocrine": agent_endocrine,
        "moderator": moderator
    })
    
    # 编译图
    graph = workflow.compile()
    
    return graph


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建LLM和Embeddings
    llm = create_local_llm(
        base_url="http://0.0.0.0:30003/v1",
        model_name="qwen3-8B"
    )
    
    embeddings = create_embeddings(
        embedding_api_url="http://0.0.0.0:30004/v1",
        embedding_model="Qwen3-Embedding-8B"
    )
    
    # 创建辩论图
    print("正在创建辩论系统...")
    
    graph = create_debate_graph(
        llm=llm,
        embeddings=embeddings
    )
    
    # 测试问题
    question = "对于HR+/HER2-、淋巴结阴性的早期乳腺癌患者，21基因检测RS评分为18分，年龄45岁，应该如何制定辅助治疗方案？"
    
    print("\n" + "="*60)
    print("开始医疗多智能体辩论...")
    print("="*60)
    print(f"\n【用户问题】\n{question}\n")
    
    # 初始化状态
    initial_state = {
        "question": question,
        "agent_21gene_answer": "",
        "agent_targeted_answer": "",
        "agent_chemotherapy_answer": "",
        "agent_endocrine_answer": "",
        "debate_round": 0,
        "debate_history": [],
        "consensus_reached": False,
        "final_answer": ""
    }
    
    # 运行图
    result = graph.invoke(initial_state)
    
    # 输出结果
    print("\n" + "="*60)
    print("【各专家观点】")
    print("="*60)
    print(f"\n21-Gene专家:\n{result.get('agent_21gene_answer', '')}\n")
    print(f"\n靶向治疗专家:\n{result.get('agent_targeted_answer', '')}\n")
    print(f"\n化疗专家:\n{result.get('agent_chemotherapy_answer', '')}\n")
    print(f"\n内分泌治疗专家:\n{result.get('agent_endocrine_answer', '')}\n")
    
    print("\n" + "="*60)
    print("【辩论历史】")
    print("="*60)
    for i, history in enumerate(result.get('debate_history', []), 1):
        print(f"\n{history}")
    
    print("\n" + "="*60)
    print("【最终回答】")
    print("="*60)
    print(f"\n{result.get('final_answer', '')}\n")
