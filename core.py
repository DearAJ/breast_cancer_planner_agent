from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator


def create_local_llm(
    base_url: str = "http://0.0.0.0:30003/v1",
    model_name: str = "Qwen2.5-7B-Instruct",
    temperature: float = 0.0,
    timeout: int = 300
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        api_key=""  # 本地服务通常不需要API key
    )

# ==================== 状态定义 ====================
class AgentState(TypedDict):
    patient_info: str  # 患者信息
    clinical_features: dict  # 提取的临床特征
    diagnosis_list: list  # 鉴别诊断列表
    information_gaps: list  # 信息缺口
    query_strategy: dict  # 查询策略
    case_rag_results: str  # 病例RAG检索结果
    literature_rag_results: str  # 文献RAG检索结果
    final_recommendation: str  # 最终治疗方案推荐
    treatment_plan: dict  # 结构化治疗方案
    messages: Annotated[list, add_messages]  # 消息历史


# ==================== 主规划智能体 ====================
class PlannerAgent:
    """主规划智能体：负责诊断推理与问题分解"""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """你是一位经验丰富的乳腺癌诊断规划专家。
你的任务是：
1. 解析乳腺癌患者信息，提取关键临床特征（病理类型、分期、免疫组化指标、淋巴结状态等）
2. 基于病理和免疫组化结果，确定诊断和分期
3. 识别信息缺口（如缺失的检查项目、不完整的免疫组化数据等）
4. 制定查询策略，分配给子智能体检索相似病例和文献指南

请基于乳腺癌诊疗规范进行系统性的诊断推理和规划。"""
    
    def plan(self, state: AgentState) -> AgentState:
        """主规划逻辑"""
        print("\n" + "="*60)
        print("【步骤1】主规划智能体 (PlannerAgent) 开始执行...")
        print("="*60)
        
        patient_info = state["patient_info"]
        print(f"\n正在分析患者信息...")
        
        # 构建规划提示
        planning_prompt = f"""
乳腺癌患者信息：
{patient_info}

请执行以下步骤：

1. 解析患者信息，提取关键临床特征：
   - 病理类型（DCIS、浸润性癌等）
   - 病理分期（TNM分期）
   - 免疫组化指标（ER、PR、HER2、Ki67）
   - 淋巴结状态（SLN、ALN转移情况）
   - 肿块大小、位置、分级
   - 其他重要信息（年龄、绝经状态、既往史等）

2. 基于提取的特征，确定诊断和分期（如：DCIS TisN0M0、浸润性癌等）

3. 识别信息缺口（如：缺失的HER2-FISH结果、21-基因检测、其他必要检查等）

4. 制定查询策略：
   - 根据病理类型和分期，检索相似病例（重点关注相同病理类型、分期、免疫组化特征的病例）
   - 检索相关诊疗指南和文献（如：NCCN指南、CSCO指南、相关临床研究）

请以JSON格式输出：
{{
    "clinical_features": {{
        "病理类型": "...",
        "病理分期": "...",
        "免疫组化": {{"ER": "...", "PR": "...", "HER2": "...", "Ki67": "..."}},
        "淋巴结状态": "...",
        "肿块信息": "...",
        "其他特征": "..."
    }},
    "diagnosis_list": ["主要诊断", "相关诊断", ...],
    "information_gaps": ["缺失信息1", "缺失信息2", ...],
    "query_strategy": {{
        "case_rag_query": "病例检索查询（如：DCIS TisN0M0 ER+PR+HER2- 相似病例）",
        "literature_rag_query": "文献检索查询（如：DCIS 诊疗指南 辅助治疗）",
        "execution_mode": "parallel"
    }}
}}
"""
        
        if self.llm is None:
            # 模拟响应（用于测试或未配置LLM时）
            response_content = """ """
            class MockResponse:
                content = response_content
            response = MockResponse()
        else:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=planning_prompt)
            ]
            print("正在调用LLM进行诊断规划...")
            response = self.llm.invoke(messages)
            print("✓ LLM响应已接收")
        
        # 解析响应并更新状态
        import json
        print("\n正在解析规划结果...")
        try:
            # 尝试从响应中提取JSON
            content = response.content
            # 简单的JSON提取（实际应用中可能需要更robust的解析）
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            
            plan_data = json.loads(json_str)
            
            state["clinical_features"] = plan_data.get("clinical_features", {})
            state["diagnosis_list"] = plan_data.get("diagnosis_list", [])
            state["information_gaps"] = plan_data.get("information_gaps", [])
            state["query_strategy"] = plan_data.get("query_strategy", {})
            
            # 输出规划结果
            print("\n【规划结果】")
            print(f"  临床特征: {state['clinical_features']}")
            print(f"  诊断列表: {state['diagnosis_list']}")
            print(f"  信息缺口: {state['information_gaps']}")
            print(f"  查询策略: {state['query_strategy']}")
            print("\n✓ 主规划智能体执行完成")
            
        except Exception as e:
            print(f"✗ 解析规划结果时出错: {e}")
            # 设置默认值
            state["clinical_features"] = {}
            state["diagnosis_list"] = []
            state["information_gaps"] = []
            state["query_strategy"] = {
                "case_rag_query": patient_info,
                "literature_rag_query": patient_info,
                "execution_mode": "parallel"
            }
        
        return state


# ==================== 病例RAG智能体 ====================
class CaseRAGAgent:
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store  # 向量数据库（实际应用中需要实现）
        self.system_prompt = """你是一位乳腺癌病例检索专家。
你的任务是根据查询检索相似的历史乳腺癌病例，重点关注：
- 相同病理类型和分期的病例
- 相似免疫组化特征的病例
- 这些病例的治疗方案和预后情况"""
    
    def retrieve(self, state: AgentState) -> AgentState:
        """检索历史病例"""
        print("\n" + "="*60)
        print("【步骤2】病例RAG智能体 (CaseRAGAgent) 开始执行...")
        print("="*60)
        
        query_strategy = state.get("query_strategy", {})
        query = query_strategy.get("case_rag_query", state["patient_info"])
        print(f"\n检索查询: {query}")
        
        # 模拟RAG检索（实际应用中需要连接向量数据库）
        if self.vector_store:
            # 实际检索逻辑
            print("正在从向量数据库检索相似病例...")
            results = self._retrieve_from_vector_store(query)
        else:
            # 模拟检索结果
            print("使用模拟检索结果（未连接向量数据库）...")
            diagnosis = state.get('diagnosis_list', ['未明确'])[0] if state.get('diagnosis_list') else '未明确'
            clinical_features = state.get('clinical_features', {})
            path_type = clinical_features.get('病理类型', 'DCIS')
            ihc = clinical_features.get('免疫组化', {})
            
            results = f"""
基于查询"{query}"检索到的相似乳腺癌病例：

病例1：
- 患者：40岁女性，绝经前
- 病理类型：{path_type}
- 分期：{clinical_features.get('病理分期', 'TisN0M0')}
- 免疫组化：ER {ihc.get('ER', '90%+')}, PR {ihc.get('PR', '90%+')}, HER2 {ihc.get('HER2', '1+')}, Ki67 {ihc.get('Ki67', '15%')}
- 治疗方案：根据指南，DCIS患者行单纯乳房切除术后，ER+患者建议内分泌治疗
- 随访：术后1年无复发，生活质量良好

病例2：
- 患者：45岁女性，绝经前
- 病理类型：{path_type}
- 分期：{clinical_features.get('病理分期', 'TisN0M0')}
- 免疫组化：ER {ihc.get('ER', '95%+')}, PR {ihc.get('PR', '>95%+')}, HER2 {ihc.get('HER2', '1+')}, Ki67 {ihc.get('Ki67', '15%')}
- 治疗方案：单纯乳房切除术+前哨淋巴结活检，术后给予他莫昔芬内分泌治疗5年
- 随访：术后2年无复发，定期复查正常
"""
        
        state["case_rag_results"] = results
        print("\n【病例检索结果】")
        print(results)
        print("\n✓ 病例RAG智能体执行完成")
        return state
    
    def _retrieve_from_vector_store(self, query: str) -> str:
        """从向量数据库检索（需要实际实现）"""
        # TODO: 实现实际的向量检索逻辑
        return "检索结果"


# ==================== 文献RAG智能体 ====================
class LiteratureRAGAgent:
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store  # 向量数据库（实际应用中需要实现）
        self.system_prompt = """你是一位乳腺癌医学文献检索专家。
你的任务是根据查询检索相关的乳腺癌诊疗指南、临床研究和循证医学证据，包括：
- NCCN指南、CSCO指南等权威指南
- 辅助治疗相关研究（化疗、放疗、内分泌治疗、靶向治疗）
- 预后评估和风险分层研究"""
    
    def retrieve(self, state: AgentState) -> AgentState:
        """检索医学文献"""
        print("\n" + "="*60)
        print("【步骤3】文献RAG智能体 (LiteratureRAGAgent) 开始执行...")
        print("="*60)
        
        query_strategy = state.get("query_strategy", {})
        query = query_strategy.get("literature_rag_query", state["patient_info"])
        print(f"\n检索查询: {query}")
        
        # 模拟RAG检索（实际应用中需要连接向量数据库）
        if self.vector_store:
            # 实际检索逻辑（使用分层检索）
            print("正在从向量数据库检索医学文献...")
            results = self._retrieve_from_vector_store(query, use_hierarchical=True)
        else:
            # 模拟检索结果
            print("使用模拟检索结果（未连接向量数据库）...")
            diagnosis = state.get('diagnosis_list', ['DCIS'])[0] if state.get('diagnosis_list') else 'DCIS'
            clinical_features = state.get('clinical_features', {})
            path_type = clinical_features.get('病理类型', 'DCIS')
            ihc = clinical_features.get('免疫组化', {})
            er_status = ihc.get('ER', '阳性')
            pr_status = ihc.get('PR', '阳性')
            her2_status = ihc.get('HER2', '1+')
            
            results = f"""
基于查询"{query}"检索到的乳腺癌医学文献：

指南推荐（NCCN/CSCO指南）：
- {path_type}诊疗指南（2024版）
- 对于{path_type} TisN0M0患者：
  * 手术：单纯乳房切除术或保乳手术+前哨淋巴结活检
  * 辅助治疗：根据ER/PR状态决定是否内分泌治疗
  * ER+患者：推荐他莫昔芬或芳香化酶抑制剂（绝经后）5-10年
  * 通常不需要辅助化疗和放疗（除非高危因素）

临床研究：
- 研究1：DCIS患者ER+者，内分泌治疗可降低50%同侧和对侧乳腺癌风险
- 研究2：Meta分析显示，ER+DCIS患者术后内分泌治疗显著改善无病生存率
- 研究3：对于低-中级别DCIS，单纯手术切除预后良好，10年生存率>95%

循证证据等级：
- 手术切除：A级证据（强烈推荐）
- ER+患者内分泌治疗：A级证据（强烈推荐）
- 辅助化疗：不推荐（DCIS通常不需要）
- 辅助放疗：根据保乳情况决定（全切通常不需要）
"""
        
        state["literature_rag_results"] = results
        print("\n【文献检索结果】")
        print(results)
        print("\n✓ 文献RAG智能体执行完成")
        return state
    
    def _retrieve_from_vector_store(self, query: str, use_hierarchical: bool = True) -> str:
        """从向量数据库检索
        
        Args:
            query: 查询文本
            use_hierarchical: 是否使用分层检索（先检索章节，再检索章节内内容）
        """
        if hasattr(self.vector_store, 'hierarchical_search') and use_hierarchical:
            # 使用分层检索
            print("使用分层检索模式：先检索章节，再检索章节内内容...")
            return self.vector_store.hierarchical_search(query, top_chapters=3, chunks_per_chapter=3)
        elif hasattr(self.vector_store, 'retrieve'):
            # 使用普通检索
            return self.vector_store.retrieve(query, k=5)
        else:
            # 兼容其他类型的vector_store
            return "检索结果"


# ==================== 总结智能体 ====================
class SummaryAgent:
    """总结智能体：汇总信息并输出治疗方案推荐"""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """你是一位资深的乳腺癌诊疗专家。
你的任务是综合所有信息，包括：
1. 患者的原始信息（病理、分期、免疫组化等）
2. 提取的临床特征
3. 诊断和分期
4. 检索到的相似历史病例
5. 检索到的医学指南和文献

基于这些信息，输出个性化的乳腺癌治疗方案推荐。推荐应该：
- 基于NCCN/CSCO等权威指南
- 考虑患者的具体情况（年龄、绝经状态、病理特征、免疫组化等）
- 明确是否需要辅助化疗、放疗、内分泌治疗、靶向治疗
- 提供具体的治疗方案、用药剂量、疗程
- 说明预期效果、随访计划和注意事项

重要：你必须以JSON格式输出结构化的治疗方案，格式如下：
{
    "辅助化疗方案": "具体方案（如：EC->T、TAC等）或'不需要'",
    "辅助放疗方案": "需要" 或 "不需要",
    "辅助内分泌治疗方案": "具体方案（如：(SERM->AI)*5y、AI*5y等）或'不需要'",
    "辅助靶向治疗方案": "具体方案（如：CDK4/6抑制剂、曲妥珠单抗等）或'不需要'"
}"""
    
    def summarize(self, state: AgentState) -> AgentState:
        """汇总信息并生成治疗方案"""
        print("\n" + "="*60)
        print("【步骤4】总结智能体 (SummaryAgent) 开始执行...")
        print("="*60)
        print("\n正在汇总所有信息并生成治疗方案推荐...")
        
        summary_prompt = f"""
请基于以下信息，生成个性化的乳腺癌治疗方案推荐：

【患者原始信息】
{state['patient_info']}

【提取的临床特征】
{state.get('clinical_features', {})}

【诊断和分期】
{state.get('diagnosis_list', [])}

【信息缺口】
{state.get('information_gaps', [])}

【历史病例检索结果】
{state.get('case_rag_results', '')}

【医学文献检索结果】
{state.get('literature_rag_results', '')}

请按照以下结构输出治疗方案推荐：

首先，请以JSON格式输出结构化的治疗方案（这是最重要的部分）：
{{
    "辅助化疗方案": "具体方案（如：EC->T、TAC、AC->T等）或'不需要'",
    "辅助放疗方案": "需要" 或 "不需要",
    "辅助内分泌治疗方案": "具体方案（如：(SERM->AI)*5y、AI*5y、SERM*5y等）或'不需要'",
    "辅助靶向治疗方案": "具体方案（如：CDK4/6抑制剂、曲妥珠单抗、帕妥珠单抗等）或'不需要'"
}}

然后，请提供详细的文字说明：

1. 【诊断总结】
   - 主要诊断和分期
   - 关键病理特征总结

2. 【辅助治疗方案详细说明】
   - 辅助化疗方案：详细说明（如需要，说明具体药物、剂量、周期）
   - 辅助放疗方案：详细说明（如需要，说明适应症、剂量、范围）
   - 辅助内分泌治疗方案：详细说明（如需要，说明药物选择、剂量、疗程）
   - 辅助靶向治疗方案：详细说明（如需要，说明药物、剂量、疗程）

3. 【治疗方案依据】
   - 基于哪些指南和证据
   - 为什么选择这些方案

4. 【预期效果和预后】
   - 预期治疗效果
   - 预后评估

5. 【随访计划】
   - 随访时间安排
   - 随访检查项目

6. 【注意事项】
   - 治疗相关注意事项
   - 不良反应监测

7. 【需要补充的信息】
   - 如果存在信息缺口，说明需要补充的检查或信息

注意：JSON格式的结构化治疗方案必须放在输出的最前面，用```json```代码块包裹。
"""
        
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=summary_prompt)
        ]
        print("正在调用LLM生成最终治疗方案...")
        response = self.llm.invoke(messages)
        state["final_recommendation"] = response.content
        print("✓ LLM响应已接收")
        
        # 解析结构化治疗方案
        import json
        print("\n正在解析结构化治疗方案...")
        try:
            content = response.content
            # 尝试从响应中提取JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content and "{" in content:
                # 尝试找到第一个{到最后一个}之间的内容
                start_idx = content.find("{")
                end_idx = content.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx+1]
                else:
                    json_str = content.split("```")[1].split("```")[0].strip()
            else:
                # 尝试直接查找JSON对象
                start_idx = content.find("{")
                end_idx = content.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx+1]
                else:
                    json_str = content
            
            treatment_plan = json.loads(json_str)
            state["treatment_plan"] = treatment_plan
            print("✓ 结构化治疗方案解析成功")
            print(f"  辅助化疗方案: {treatment_plan.get('辅助化疗方案', '未指定')}")
            print(f"  辅助放疗方案: {treatment_plan.get('辅助放疗方案', '未指定')}")
            print(f"  辅助内分泌治疗方案: {treatment_plan.get('辅助内分泌治疗方案', '未指定')}")
            print(f"  辅助靶向治疗方案: {treatment_plan.get('辅助靶向治疗方案', '未指定')}")
        except Exception as e:
            print(f"✗ 解析结构化治疗方案时出错: {e}")
            # 设置默认值
            state["treatment_plan"] = {
                "辅助化疗方案": "未确定",
                "辅助放疗方案": "未确定",
                "辅助内分泌治疗方案": "未确定",
                "辅助靶向治疗方案": "未确定"
            }
        
        print("\n✓ 总结智能体执行完成")
        
        return state




# ==================== LangGraph 工作流构建 ====================
def create_medical_agent_graph(
    llm=None, 
    case_vector_store=None, 
    literature_vector_store=None,
    local_model_url: str = "http://0.0.0.0:30003/v1",
    local_model_name: str = "Qwen2.5-7B-Instruct"
):
    """创建医疗诊断多智能体图
    
    Args:
        llm: 自定义的LLM实例，如果为None则自动创建
        case_vector_store: 病例向量数据库
        literature_vector_store: 文献向量数据库
        use_local_model: 是否使用本地模型（默认True）
        local_model_url: 本地模型API地址
        local_model_name: 本地模型名称
    
    工作流：
    1. planner（主规划智能体）→ 解析患者信息，制定查询策略
    2. 根据策略执行RAG检索：
       - 并行模式：case_rag 和 literature_rag 顺序执行（LangGraph会自动优化）
       - 顺序模式：case_rag → literature_rag
    3. summarizer（总结智能体）→ 汇总所有信息，输出治疗方案
    """
    
    if llm is None:
        try:
            llm = create_local_llm(
                base_url=local_model_url,
                model_name=local_model_name
            )
            print(f"已连接到本地模型: {local_model_url}/{local_model_name}")
        except Exception as e:
                print(f"连接本地模型失败: {e}")
                llm = None
    
    planner = PlannerAgent(llm)
    case_rag = CaseRAGAgent(llm, case_vector_store)
    literature_rag = LiteratureRAGAgent(llm, literature_vector_store)
    summarizer = SummaryAgent(llm)
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("planner", planner.plan)
    workflow.add_node("case_rag", case_rag.retrieve)
    workflow.add_node("literature_rag", literature_rag.retrieve)
    workflow.add_node("summarizer", summarizer.summarize)
    
    # 设置入口点
    workflow.set_entry_point("planner")
    
    # 规划后进入病例检索
    workflow.add_edge("planner", "case_rag")
    
    # 病例检索后进入文献检索
    # 注意：虽然这里是顺序执行，但如果两个检索操作是独立的且使用异步，可以在实际应用中修改为真正的并行执行
    workflow.add_edge("case_rag", "literature_rag")
    
    # 文献检索后进入总结
    workflow.add_edge("literature_rag", "summarizer")
    
    # 结束
    workflow.add_edge("summarizer", END)
    
    return workflow.compile()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建图（使用本地模型）
    graph = create_medical_agent_graph(
        local_model_url="http://0.0.0.0:30003/v1",
        local_model_name="Qwen2.5-7B-Instruct"
    )
    
    # 示例患者信息
    patient_info = """
    基本信息：
    - 年龄：41
    - 性别：女
    - 住院号：z1173791
    - 主诊医生：医生
    - 身高：1.63m
    - 体重：52.0kg
    - 体表面积：1.55
    - KPS：100
    - LMP：绝经前, 2023-09-09
    - 原床位：2202
    - 左右侧：左侧
    - 伴发疾病及服药情况：局麻背部皮下肿物切除术
    - 乳腺癌/卵巢癌既往或家族史：无
    - 异常检验及检查：胸片检查两肺纹理增多，建议至呼吸科随访。肌酐 54μmol/L  总胆固醇 5.48↑mmol/L  前白蛋白 174↓mg/L
    - c-肿块大小：3.5cm
    - 病理分期：T:is, N:0, M:0
    - 活检日期：2023-09-19
    - 病理类型T1-CNB：DCIS
    - 手术日期：2023-09-22
    - 手术方式：左乳癌单纯乳房切除术+前哨淋巴结活检术+低位组淋巴结清扫术
    - 肿块象限T1：外下
    - 病理类型T1：原位实性乳头状癌, 伴局灶粉刺样坏死及微钙化
    - 肿块大小pT1：瘤体：范围无法准确评估
    - Grade-T1：I-II
    - 肿块象限T2：外下
    - 病理类型T2：原位实性乳头状癌
    - 肿块大小pT2：0.35cm
    - Grade-T2：I-II
    - SLN活检数：4
    - SLN转移数：0
    - ALN清扫数：0
    - ALN转移数：0
    - 总LN手术数：4
    - 总LN转移数：0
    - ER(%)(T1-CNB)：约 90%+，染色中等-强
    - PR(%)(T1-CNB)：约 90%+，染色中
    - CerbB-2(T1-CNB)：1+
    - Ki67(%)(T1-CNB)：15
    - ER(%)(T1)：约 95%+，染色中等-强
    - PR(%)(T1)：>95%+，染色强
    - CerbB-2(T1)：1+
    - Ki67(%)(T1)：15
    - ER(%)(T2)：约 95%+，染色中等-强
    - PR(%)(T2)：>95%+，染色强
    - CerbB-2(T2)：1+
    - Ki67(%)(T2)：15
    - 备注：低位组：腋窝少量脂肪组织，经仔细查找，未查见明显肿大淋巴结。12 点钟方向局灶黏液囊肿性病变伴非典型导管增生（ADH）。
    """
    
    print(patient_info)
    # 初始化状态
    initial_state = {
        "patient_info": patient_info,
        "clinical_features": {},
        "diagnosis_list": [],
        "information_gaps": [],
        "query_strategy": {},
        "case_rag_results": "",
        "literature_rag_results": "",
        "final_recommendation": "",
        "treatment_plan": {},
        "messages": []
    }
    
    # 运行图
    print("\n" + "="*60)
    print("开始执行医疗诊断流程...")
    print("="*60)
    
    result = graph.invoke(initial_state)
    
    
    print("\n" + "="*60)
    print("【详细治疗方案说明】")
    print("="*60)
    print(result["final_recommendation"])
    
    print("\n" + "="*60)
    print("流程执行完成！")
