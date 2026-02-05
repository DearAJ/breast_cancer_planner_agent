"""
运行医疗多智能体辩论系统的示例脚本
"""

from medical_debate_system import (
    create_local_llm,
    create_embeddings,
    create_debate_graph
)
from pathlib import Path


def main():
    """主函数"""
    # 创建LLM和Embeddings
    print("正在初始化LLM和Embeddings...")
    llm = create_local_llm(
        base_url="http://0.0.0.0:30003/v1",
        model_name="qwen3-8B"
    )
    
    embeddings = create_embeddings(
        embedding_api_url="http://0.0.0.0:30004/v1",
        embedding_model="Qwen3-Embedding-8B"
    )
    
    # 创建辩论图
    print("\n正在创建辩论系统...")
    
    graph = create_debate_graph(
        llm=llm,
        embeddings=embeddings
    )
    
    # 用户问题
    print("\n" + "="*60)
    print("医疗多智能体辩论系统")
    print("="*60)
    
    question = input("\n请输入您的问题（或按Enter使用默认问题）: ").strip()
    if not question:
        question = "对于HR+/HER2-、淋巴结阴性的早期乳腺癌患者，21基因检测RS评分为18分，年龄45岁，应该如何制定辅助治疗方案？"
    
    print(f"\n【用户问题】\n{question}\n")
    print("开始辩论...\n")
    
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
    try:
        result = graph.invoke(initial_state)
        
        # 输出结果
        print("\n" + "="*60)
        print("【各专家观点】")
        print("="*60)
        print(f"\n【21-Gene专家】\n{result.get('agent_21gene_answer', '')}\n")
        print(f"\n【靶向治疗专家】\n{result.get('agent_targeted_answer', '')}\n")
        print(f"\n【化疗专家】\n{result.get('agent_chemotherapy_answer', '')}\n")
        print(f"\n【内分泌治疗专家】\n{result.get('agent_endocrine_answer', '')}\n")
        
        print("\n" + "="*60)
        print("【辩论历史】")
        print("="*60)
        for i, history in enumerate(result.get('debate_history', []), 1):
            print(f"\n{history}")
        
        print("\n" + "="*60)
        print("【最终回答】")
        print("="*60)
        print(f"\n{result.get('final_answer', '')}\n")
        
        print("\n" + "="*60)
        print("辩论完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
