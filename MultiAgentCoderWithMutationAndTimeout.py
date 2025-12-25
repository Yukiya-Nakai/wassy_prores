import os
import json
import subprocess
import operator
from typing import List, TypedDict, Dict, Any, Optional

# LangChain / LangGraph 関連ライブラリ
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 0. 設定と準備 ---

# APIキーの設定
if "OPENAI_API_KEY" not in os.environ:
    # os.environ["OPENAI_API_KEY"] = "sk-..." 
    print("⚠️ Warning: OPENAI_API_KEY is not set. Please set it via os.environ or manual input.")

# モデル設定 (GPT-4o推奨)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# --- 1. State (共有メモリ) の定義 ---

class AgentState(TypedDict):
    requirements: str            # ユーザーの要求
    
    # Role A (Planning)
    po_output: str
    architect_output: str
    critic_output: str
    design_plan: Dict
    replan_count: int            # ★追加: Role A内での再設計回数カウンター
    
    # Dev & QA
    test_code: str
    impl_code: str
    test_result: str
    feedback: str
    
    # Control
    iteration: int               # 全体のループ回数
    mutation_logs: List[str]
    current_phase: str
    next_action: str


# --- 2. 出力スキーマ (Pydantic Models) ---

class FinalSpec(BaseModel):
    function_name: str = Field(description="関数名")
    inputs: List[Dict[str, str]] = Field(description="引数リスト")
    output_type: str = Field(description="返り値の型")
    description: str = Field(description="概要")
    requirements: List[str] = Field(description="機能要件")
    edge_cases: List[str] = Field(description="エッジケースリスト")
    
    needs_replan: bool = Field(description="再設計が必要ならTrue")
    replan_reason: str = Field(description="再設計が必要な理由")

class TestOutput(BaseModel):
    thought_process: str
    test_code: str

class CodeOutput(BaseModel):
    thought_process: str
    impl_code: str

class ReflectionOutput(BaseModel):
    analysis: str
    action: str = Field(description="retry_code, retry_test, replan, mutation_check, finish")
    feedback: str

class MutantOutput(BaseModel):
    mutant_code: str
    mutation_description: str


# --- 3. ノード関数 ---

# === Role A: Planning Squad ===

def node_planner_po(state: AgentState) -> AgentState:
    """[Role A1] Product Owner"""
    print("\n🔹 [Role A1] Product Owner Analyzing...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "ユーザー要望から開発機能の目的と主要ストーリーを定義してください。技術詳細は不要です。"),
        ("human", state["requirements"])
    ])
    res = (prompt | llm).invoke({})
    return {
        "po_output": res.content,
        "iteration": 0,
        "replan_count": 0, # カウンター初期化
        "mutation_logs": [],
        "current_phase": "dev",
        "feedback": ""
    }

def node_planner_architect(state: AgentState) -> AgentState:
    """[Role A2] Architect"""
    feedback = state.get("feedback", "")
    replan_cnt = state.get("replan_count", 0)
    print(f"\n🔹 [Role A2] Architect Designing... (Replan: {replan_cnt}, FB: {feedback})")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "POの定義に基づきPython実装方針を設計してください。フィードバックがあれば修正してください。"),
        ("human", f"Request: {state['requirements']}\nPO: {state['po_output']}\nFeedback: {feedback}")
    ])
    res = (prompt | llm).invoke({})
    return {"architect_output": res.content}

def node_planner_critic(state: AgentState) -> AgentState:
    """[Role A3] Critic"""
    print("\n🔹 [Role A3] Critic Reviewing...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Architect案のエッジケース・論理欠陥・セキュリティリスクを厳しく指摘してください。"),
        ("human", state["architect_output"])
    ])
    res = (prompt | llm).invoke({})
    return {"critic_output": res.content}

def node_planner_reviser(state: AgentState) -> AgentState:
    """[Role A4] Reviser (with Timeout Logic)"""
    print("\n🔹 [Role A4] Reviser Compiling...")
    replan_cnt = state.get("replan_count", 0)
    MAX_REPLANS = 3  # ★設定: 最大再設計回数
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""これまでの議論を統合し、最終仕様書(JSON)を作成してください。
現在の再設計回数: {replan_cnt} / {MAX_REPLANS}

重要:
1. Criticの指摘が致命的で修正不可能な場合のみ `needs_replan=True` としてください。
2. ただし、些細な問題であれば `needs_replan=False` として仕様書を完成させてください。
"""),
        ("human", f"PO: {state['po_output']}\nArch: {state['architect_output']}\nCritic: {state['critic_output']}")
    ])
    
    chain = prompt | llm.with_structured_output(FinalSpec)
    result = chain.invoke({})
    
    # ★ タイムアウト判定ロジック
    if result.needs_replan:
        if replan_cnt >= MAX_REPLANS:
            print(f"   ⚠️ Replan Limit Reached ({MAX_REPLANS}). Forcing proceed despite objections.")
            # 強制的に進めるため、フラグを無視して仕様書として扱う
            # (needs_replan=Trueのままだと後続が困るので、このままdesign_planに入れて進める)
            return {
                "design_plan": result.model_dump(),
                "feedback": f"Warning: Spec finalized after {MAX_REPLANS} replans. Critic issues may remain.",
                "next_action": "proceed"
            }
        else:
            print(f"   ⚠️ Reviser Requesting Replan ({replan_cnt+1}/{MAX_REPLANS}): {result.replan_reason}")
            return {
                "design_plan": {},
                "feedback": result.replan_reason,
                "next_action": "replan_internal",
                "replan_count": replan_cnt + 1
            }
            
    print(f"   -> Spec Finalized: {result.function_name}")
    return {
        "design_plan": result.model_dump(),
        "next_action": "proceed",
        "feedback": ""
    }


# === Role B, C, D, E ===

def node_tester(state: AgentState) -> AgentState:
    """[Role B] Tester"""
    print("\n🔹 [Role B] Tester Running...")
    spec = state["design_plan"]
    fb = state.get("feedback", "")
    phase = state.get("current_phase", "dev")
    
    if phase == "mutation":
        instr = f"Mutation Check Failed: {fb}. Add tests to kill this mutant."
    else:
        instr = f"Create initial tests. FB: {fb}. Cover edge cases: {spec.get('edge_cases')}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "pytestコードを作成してください。`import pytest`必須。"),
        ("human", f"Spec: {json.dumps(spec)}\nInstr: {instr}")
    ])
    res = (prompt | llm.with_structured_output(TestOutput)).invoke({})
    return {"test_code": res.test_code, "feedback": ""}

def node_coder(state: AgentState) -> AgentState:
    """[Role C] Coder"""
    print("\n🔹 [Role C] Coder Running...")
    spec = state["design_plan"]
    test = state["test_code"]
    fb = state.get("feedback", "")
    impl = state.get("impl_code", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "テストを通す実装コードを作成してください。"),
        ("human", f"Spec: {json.dumps(spec)}\nTest: {test}\nImpl: {impl}\nFB: {fb}")
    ])
    res = (prompt | llm.with_structured_output(CodeOutput)).invoke({})
    return {"impl_code": res.impl_code}

def node_executor(state: AgentState) -> AgentState:
    """[Role D] Executor"""
    print("\n🔹 [Role D] Executor Running...")
    with open("implementation.py", "w", encoding="utf-8") as f: f.write(state["impl_code"])
    with open("test_suite.py", "w", encoding="utf-8") as f: f.write(state["test_code"])
    
    try:
        res = subprocess.run(["pytest", "test_suite.py", "-v", "--tb=short"], capture_output=True, text=True, timeout=10)
        return {"test_result": res.stdout + res.stderr}
    except Exception as e:
        return {"test_result": str(e)}

def node_reflector(state: AgentState) -> AgentState:
    """[Role D] Reflector"""
    print("\n🔹 [Role D] Reflector Running...")
    res = state["test_result"]
    phase = state["current_phase"]
    itr = state["iteration"]
    
    # 全体ループ制限
    if itr > 20: return {"next_action": "finish", "feedback": "Global Loop Limit"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze logs. Action: retry_code, retry_test, replan, mutation_check, finish."),
        ("human", f"Phase: {phase}\nLog: {res}")
    ])
    decision = (prompt | llm.with_structured_output(ReflectionOutput)).invoke({})
    print(f"   -> Decision: {decision.action}")
    
    new_state = {"feedback": decision.feedback, "next_action": decision.action, "iteration": itr + 1}
    if decision.action == "mutation_check": new_state["current_phase"] = "mutation"
    
    # ★ ここでも replan 時にカウントをリセットするか検討可能だが、
    # 今回はAに戻る際は「大きな手戻り」として replan_count はそのまま(あるいはリセット)でもよい。
    # 簡易化のためリセットしないでおく。
    return new_state

def node_mutation_tester(state: AgentState) -> AgentState:
    """[Role E] Mutation Tester"""
    print("\n🔹 [Role E] Mutation Tester Running...")
    original_impl = state["impl_code"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Create a mutant code with 1 subtle bug."),("human", original_impl)
    ])
    mutant = (prompt | llm.with_structured_output(MutantOutput)).invoke({})
    
    with open("implementation.py", "w", encoding="utf-8") as f: f.write(mutant.mutant_code)
    try:
        res = subprocess.run(["pytest", "test_suite.py", "-q"], capture_output=True, timeout=5)
        survived = (res.returncode == 0)
    except: survived = False
    with open("implementation.py", "w", encoding="utf-8") as f: f.write(original_impl)
    
    if survived:
        print(f"   >>> ❌ Mutant Survived! ({mutant.mutation_description})")
        return {"feedback": f"Survived: {mutant.mutation_description}", "next_action": "retry_test", "mutation_logs": state["mutation_logs"] + ["Survived"]}
    else:
        print("   >>> ✅ Mutant Killed!")
        return {"feedback": "Passed", "next_action": "finish", "mutation_logs": state["mutation_logs"] + ["Killed"]}


# --- 4. グラフ構築 ---

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("planner_po", node_planner_po)
workflow.add_node("planner_architect", node_planner_architect)
workflow.add_node("planner_critic", node_planner_critic)
workflow.add_node("planner_reviser", node_planner_reviser)
workflow.add_node("tester", node_tester)
workflow.add_node("coder", node_coder)
workflow.add_node("executor", node_executor)
workflow.add_node("reflector", node_reflector)
workflow.add_node("mutation_tester", node_mutation_tester)

# Edges
workflow.set_entry_point("planner_po")
workflow.add_edge("planner_po", "planner_architect")
workflow.add_edge("planner_architect", "planner_critic")
workflow.add_edge("planner_critic", "planner_reviser")

# Reviser Conditional Edge (Timeout Logic Included)
def router_reviser(state: AgentState):
    if state["next_action"] == "replan_internal":
        return "planner_architect"
    return "tester"

workflow.add_conditional_edges("planner_reviser", router_reviser, {
    "planner_architect": "planner_architect",
    "tester": "tester"
})

workflow.add_edge("tester", "coder")
workflow.add_edge("coder", "executor")
workflow.add_edge("executor", "reflector")

# Reflector Conditional Edge
def router_reflector(state: AgentState):
    act = state["next_action"]
    if act == "retry_code": return "coder"
    elif act == "retry_test": return "tester"
    elif act == "replan": return "planner_architect"
    elif act == "mutation_check": return "mutation_tester"
    elif act == "finish": return END
    return END

workflow.add_conditional_edges("reflector", router_reflector, {
    "coder": "coder", "tester": "tester", "planner_architect": "planner_architect", 
    "mutation_tester": "mutation_tester", END: END
})

# Mutation Conditional Edge
def router_mutation(state: AgentState):
    if state["next_action"] == "retry_test": return "tester"
    return END

workflow.add_conditional_edges("mutation_tester", router_mutation, {"tester": "tester", END: END})

app = workflow.compile()

# --- 5. 実行 ---
if __name__ == "__main__":
    print("\n=======================================================")
    print(" 🚀 Multi-Agent Coder with Mutation Testing and Timeout (Full) 🚀")
    print("=======================================================")
    print("構成: PO -> Architect -> Critic -> Reviser -> Tester -> Coder -> QA(Mutation)")
    
    # ユーザー入力
    user_task = input("\n>>> 開発タスクを入力してください: ")
    if not user_task.strip():
        # デフォルトタスク
        user_task = "CSVファイル(data.csv)を読み込み、'score'列の平均値を計算する関数。ファイル欠損、空データ、列不足のエラーハンドリングを行うこと。"
        print(f"(入力なしのため、デフォルトタスクを実行します: {user_task})")
    
    initial = {
        "requirements": user_task,
        "po_output": "", "architect_output": "", "critic_output": "", "design_plan": {},
        "replan_count": 0, # 初期化
        "test_code": "", "impl_code": "", "test_result": "", "feedback": "",
        "iteration": 0, "mutation_logs": [], "current_phase": "dev", "next_action": ""
    }
    
    try:
        for s in app.stream(initial): pass
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nDone.")
    if os.path.exists("implementation.py"):
        print("--- implementation.py ---")
        with open("implementation.py", "r", encoding="utf-8") as f: print(f.read())
