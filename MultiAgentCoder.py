import os
import json
import subprocess
import operator
from typing import Annotated, List, TypedDict, Union, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 設定: API Keyなど ---
os.environ["OPENAI_API_KEY"] = "sk-..."  # ここにキーを設定するか、環境変数を使用

# モデル設定 (議論の質を高めるためGPT-4o推奨)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- 1. Stateの定義 (エージェント間で共有するメモリ) ---
class AgentState(TypedDict):
    requirements: str            # ユーザーの要求
    discussion_log: str          # Role Aの議論ログ (可視化用)
    design_plan: Dict            # Role Aが決めた仕様書 (JSON)
    test_code: str               # Role Bが書いたテストコード
    impl_code: str               # Role Cが書いた実装コード
    test_result: str             # テスト実行ログ
    feedback: str                # Role Dからの修正指示
    iteration: int               # ループ回数 (無限ループ防止)

# --- 2. 出力スキーマ定義 (Structured Output) ---

# Role Aの出力構造
class PlanOutput(BaseModel):
    discussion_log: str = Field(description="PO, Architect, Criticによる対話形式の議論ログ")
    final_plan: Dict = Field(description="確定した仕様、技術スタック、考慮されたエッジケース")

# Role Dの出力構造
class ReflectionOutput(BaseModel):
    analysis: str = Field(description="失敗原因の分析")
    action: str = Field(description="次のアクション: 'retry_code', 'retry_test', 'replan', 'finish'")
    feedback: str = Field(description="担当エージェントへの具体的な指示")

# --- 3. ノード関数 (各エージェントの処理) ---

# 前回のimport文やAgentState定義はそのまま使います
# node_planner を削除し、以下の関数群とグラフ定義を追加・変更してください

# --- Role A Sub-Agents Definitions ---

def node_planner_po(state: AgentState):
    """Sub-Agent 1: Product Owner (要件定義)"""
    print("\n--- [Role A-1] Product Owner Defining Requirements ---")
    
    requirements = state.get("requirements")
    feedback = state.get("feedback", "")
    
    prompt = f"""
    あなたはプロダクトオーナーです。
    ユーザーの要望: "{requirements}"
    
    以前のフィードバック: "{feedback}"
    
    あなたの仕事:
    ユーザーの要望を分析し、実現すべき「機能要件リスト」を作成してください。
    技術的な詳細はArchitectに任せ、あなたは「ユーザーにとっての価値」と「必須機能」に集中してください。
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"discussion_log": f"[PO]:\n{msg.content}\n\n"}

def node_planner_architect(state: AgentState):
    """Sub-Agent 2: Architect (技術設計)"""
    print("\n--- [Role A-2] Architect Designing System ---")
    
    current_log = state.get("discussion_log", "")
    
    prompt = f"""
    あなたはシステムアーキテクトです。
    これまでの議論（POの提案）を確認し、具体的な「技術設計」を行ってください。
    
    # 議論ログ:
    {current_log}
    
    あなたの仕事:
    1. Pythonでの実装方針（ライブラリ選定、クラス構造）
    2. データ構造の定義
    3. アルゴリズムの概略
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    # ログを追記していく
    return {"discussion_log": f"{current_log}[Architect]:\n{msg.content}\n\n"}

def node_planner_critic(state: AgentState):
    """Sub-Agent 3: Devil's Advocate (批判的レビュー)"""
    print("\n--- [Role A-3] Critic Reviewing Plan ---")
    
    current_log = state.get("discussion_log", "")
    
    prompt = f"""
    あなたは批判的レビュアー（Devil's Advocate）です。
    POとArchitectの提案に対して、意図的に「欠陥」や「リスク」を指摘してください。
    同意は不要です。最悪のケースを想定してください。
    
    # 議論ログ:
    {current_log}
    
    チェックポイント:
    - エッジケース（空入力、巨大データ、型違い）
    - セキュリティリスク
    - 実装の複雑さ
    - 仕様の矛盾
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    return {"discussion_log": f"{current_log}[Critic]:\n{msg.content}\n\n"}

def node_planner_reviser(state: AgentState):
    """Sub-Agent 4: Reviser (最終化)"""
    print("\n--- [Role A-4] Reviser Finalizing Plan ---")
    
    discussion_log = state.get("discussion_log", "")
    
    prompt = f"""
    あなたは議論のモデレーターです。
    PO、Architect、Criticの議論を踏まえて、「最終的な実装仕様書」をJSON形式でまとめてください。
    
    特にCriticの指摘事項（エッジケース等）を必ず仕様またはテスト要件に盛り込んでください。
    
    # 全議論ログ:
    {discussion_log}
    
    出力は以下のJSONフォーマットのみにしてください:
    {{
        "final_plan": {{
            "requirements": [...],
            "tech_stack": [...],
            "architecture": "...",
            "edge_cases_considered": [...]
        }},
        "discussion_summary": "議論の要約..."
    }}
    """
    
    # JSON構造で出力させる
    structured_llm = llm.with_structured_output(PlanOutput)
    response = structured_llm.invoke([HumanMessage(content=prompt)])
    
    # discussion_logはこれまでの履歴をそのまま残す（可視化のため）
    return {
        "design_plan": response.final_plan,
        # ここで discussion_log を上書きせず、これまでの流れを維持するか、要約を追加するか選べます。
        # 今回は可視化用にフルのログを残します。
    }

def node_tester(state: AgentState):
    """Role B: テストコード生成 (TDD)"""
    print("\n--- [Role B] QA Engineer Generating Tests ---")
    
    plan = state["design_plan"]
    discussion = state["discussion_log"]
    
    prompt = f"""
    あなたはQAエンジニアです。以下の仕様と議論ログに基づいて、Pythonの`unittest`コードを作成してください。
    特に議論ログで指摘された「エッジケース」を網羅するテストケースを含めてください。
    
    # 仕様:
    {json.dumps(plan, ensure_ascii=False)}
    
    # 議論ログ（懸念点）:
    {discussion}
    
    出力はPythonコードブロック(```python ... ```)のみにしてください。
    ファイル名は `test_solution.py` とし、実装ファイル `solution.py` からクラスや関数をimportする前提で書いてください。
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    code = msg.content.replace("```python", "").replace("```", "").strip()
    
    return {"test_code": code}

def node_coder(state: AgentState):
    """Role C: 実装コード生成"""
    print("\n--- [Role C] Developer Implementing Code ---")
    
    plan = state["design_plan"]
    test_code = state["test_code"]
    feedback = state.get("feedback", "")
    
    prompt = f"""
    あなたはシニアエンジニアです。以下の仕様とテストコードを満たす実装コード(`solution.py`)を書いてください。
    
    # 仕様:
    {json.dumps(plan, ensure_ascii=False)}
    
    # テストコード（これをパスする必要があります）:
    {test_code}
    
    # 前回の修正指示:
    {feedback}
    
    出力はPythonコードブロックのみにしてください。
    """
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    code = msg.content.replace("```python", "").replace("```", "").strip()
    
    return {"impl_code": code}

def node_executor(state: AgentState):
    """Tool: コード実行 (ローカルサンドボックスの代用)"""
    print("\n--- [Tool] Executing Tests ---")
    
    # 一時ファイルに書き出し
    with open("solution.py", "w", encoding="utf-8") as f:
        f.write(state["impl_code"])
    with open("test_solution.py", "w", encoding="utf-8") as f:
        f.write(state["test_code"])
        
    # テスト実行 (unittest)
    try:
        # solution.pyをimportできるようにPYTHONPATHを設定しないといけないが、
        # 同じディレクトリなら通常そのままで動く
        result = subprocess.run(
            ["python", "-m", "unittest", "test_solution.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout + result.stderr
    except Exception as e:
        output = str(e)
        
    print(f"[Execution Result] Length: {len(output)} chars")
    # print(output) # デバッグ用
    
    return {"test_result": output}

def node_reflector(state: AgentState):
    """Role D: 結果分析と分岐判断"""
    print("\n--- [Role D] Analyzing Results ---")
    
    result = state["test_result"]
    impl_code = state["impl_code"]
    iteration = state["iteration"]
    
    # 最大ループ回数チェック
    if iteration > 3:
        print("Max iterations reached. Stopping.")
        return {"feedback": "Max iterations reached", "action": "finish"}
    
    if "OK" in result and "FAILED" not in result:
        print(">>> Tests Passed! <<<")
        return {"action": "finish", "feedback": "Success"}
    
    # 失敗した場合の分析
    prompt = f"""
    あなたはCI分析官です。テストが失敗しました。以下の情報を元に、次に行うべきアクションを決定してください。
    
    # 実装コード:
    {impl_code}
    
    # テスト実行ログ:
    {result}
    
    アクションは以下のいずれかです:
    - "retry_code": 実装のケアレスミス。Role Cに修正させる。
    - "retry_test": テストコード自体が間違っている。Role Bに修正させる。
    - "replan": 仕様に根本的な無理がある。Role Aに議論し直させる。
    
    JSON形式で `{{ "analysis": "...", "action": "...", "feedback": "..." }}` を出力してください。
    """
    
    structured_llm = llm.with_structured_output(ReflectionOutput)
    response = structured_llm.invoke([HumanMessage(content=prompt)])
    
    print(f"[Reflexion] Decision: {response.action} | Feedback: {response.feedback}")
    
    return {
        "action": response.action, # 条件分岐用の一時ステート
        "feedback": response.feedback
    }

# --- 4. グラフの構築 (Workflow Definition) ---

def router(state: AgentState):
    """Role Dの決定に基づいて次のノードを決める関数"""
    action = state.get("action") # node_reflectorでセットされた値
    if action == "finish":
        return END
    elif action == "retry_code":
        return "coder"
    elif action == "retry_test":
        return "tester"
    elif action == "replan":
        return "planner"
    else:
        return END

# --- グラフ定義の変更 ---

workflow = StateGraph(AgentState)

# ノード追加 (Role Aを分割)
workflow.add_node("planner_po", node_planner_po)
workflow.add_node("planner_arch", node_planner_architect)
workflow.add_node("planner_critic", node_planner_critic)
workflow.add_node("planner_reviser", node_planner_reviser)

# 他のノードは前回と同じ
workflow.add_node("tester", node_tester)
workflow.add_node("coder", node_coder)
workflow.add_node("executor", node_executor)
workflow.add_node("reflector", node_reflector)

# エッジ（流れ）定義
# Role A (Internal Loop)
workflow.set_entry_point("planner_po")
workflow.add_edge("planner_po", "planner_arch")
workflow.add_edge("planner_arch", "planner_critic")
workflow.add_edge("planner_critic", "planner_reviser")

# Role A -> Role B -> ...
workflow.add_edge("planner_reviser", "tester") # Role A完了後にTesterへ
workflow.add_edge("tester", "coder")
workflow.add_edge("coder", "executor")
workflow.add_edge("executor", "reflector")

# 条件付きエッジ (Reflectorからの分岐)
def router(state: AgentState):
    action = state.get("action")
    if action == "finish": return END
    elif action == "retry_code": return "coder"
    elif action == "retry_test": return "tester"
    elif action == "replan": return "planner_po" # 再計画時はPOからやり直す
    else: return END

workflow.add_conditional_edges("reflector", router)

app = workflow.compile()

# --- 5. 実行例 ---

if __name__ == "__main__":
    # テスト用タスク: 典型的な「エッジケース考慮漏れ」が起きやすい課題
    user_request = "CSVファイルの文字列データを読み込んで、数値に変換して平均値を計算するPythonクラスを作ってください。"
    
    print(f"Starting Task: {user_request}")
    
    inputs = {"requirements": user_request, "iteration": 0}
    
    for output in app.stream(inputs):
        # ストリーム出力を見たい場合はここでprint
        pass
        
    print("\n=== Workflow Completed ===")
