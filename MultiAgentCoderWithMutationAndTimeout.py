import os
import json
import subprocess
import sys
from typing import List, TypedDict, Dict, Any, Optional

# 必要なライブラリのインポート
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    from langgraph.graph import StateGraph, END
except ImportError:
    print("エラー: 必要なライブラリが見つかりません。")
    print("pip install langchain-openai langchain-core pydantic langgraph")
    sys.exit(1)

# --- 0. 設定と準備 ---

# APIキーのチェック
if "OPENAI_API_KEY" not in os.environ:
    # os.environ["OPENAI_API_KEY"] = "sk-..." 
    # 実行時にAPIキーがない場合は警告
    print("⚠️ Warning: OPENAI_API_KEY is not set. Please set it via os.environ.")

# モデル設定 (GPT-4o推奨: 指示従順性が高いため)
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)


# --- 1. State (共有メモリ) の定義 ---

class AgentState(TypedDict):
    requirements: str            # ユーザーの要求
    
    # Role A (Planning)
    po_output: str
    architect_output: str
    critic_output: str
    design_plan: Dict
    replan_count: int            # Role A内での再設計回数カウンター
    
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
    function_name: str = Field(description="実装する関数の名前(Pythonのスネークケース, 例: calculate_average)")
    inputs: List[Dict[str, str]] = Field(description="引数のリスト。キーに名前、値に型ヒント(例: 'List[int]')")
    output_type: str = Field(description="返り値の型ヒント(例: 'float')")
    description: str = Field(description="関数の挙動概要とdocstring用の説明")
    requirements: List[str] = Field(description="実装すべき詳細な機能要件のリスト")
    edge_cases: List[str] = Field(description="考慮すべき具体的なエッジケース入力例のリスト")
    
    needs_replan: bool = Field(description="Criticの指摘が致命的で、Architectによる再設計が必要な場合はTrue")
    replan_reason: str = Field(description="再設計が必要な場合の理由")

class TestOutput(BaseModel):
    thought_process: str = Field(description="テスト設計の思考プロセス。どのエッジケースをどうカバーするか。")
    test_code: str = Field(description="pytestで実行可能な完全なPythonコード。")

class CodeOutput(BaseModel):
    thought_process: str = Field(description="実装の思考プロセス。")
    impl_code: str = Field(description="仕様を満たしテストを通す実装コード。")

class ReflectionOutput(BaseModel):
    analysis: str = Field(description="ログの分析結果。")
    action: str = Field(description="次のアクション: 'retry_code', 'retry_test', 'replan', 'mutation_check', 'finish'")
    feedback: str = Field(description="次の担当者への具体的な指示内容。")

class MutantOutput(BaseModel):
    mutant_code: str = Field(description="バグを埋め込んだコード全体。")
    mutation_description: str = Field(description="どのようなバグを埋め込んだかの説明。")


# --- 3. ノード関数 (Agentの実装) ---

# === Role A: Planning Squad ===

def node_planner_po(state: AgentState) -> AgentState:
    """[Role A1] Product Owner"""
    print("\n🔹 [Role A1] Product Owner Analyzing...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは熟練のプロダクトオーナー(PO)です。
ユーザーの曖昧な要求を分析し、開発チームが理解できる明確な「要件定義書」を作成してください。

以下の観点を含めてください：
1. **背景と目的**: なぜこの機能が必要なのか。
2. **主要なユーザーストーリー**: ユーザーは具体的にどう入力し、何を得たいのか。
3. **成功基準 (Acceptance Criteria)**: 何をもって「完成」とするか。

※ 技術的な実装詳細（ライブラリ選定やアルゴリズム）には踏み込まず、「What」に集中してください。"""),
        ("human", f"ユーザーの要求: {state['requirements']}")
    ])
    res = (prompt | llm).invoke({})
    
    return {
        "po_output": res.content,
        "iteration": 0,
        "replan_count": 0,
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
        ("system", """あなたは専門的なソフトウェアアーキテクトです。
POの要件定義に基づき、Pythonでの具体的な「実装設計」を行ってください。

以下の項目を設計してください：
1. **関数シグネチャ**: 関数名、引数、戻り値の型。
2. **アルゴリズム概要**: 処理の流れ。
3. **エラーハンドリング**: 無効な入力や例外に対する振る舞い。
4. **使用ライブラリ**: 標準ライブラリのみか、外部ライブラリが必要か。

※ もしフィードバックがある場合は、それを解決するように設計を修正してください。"""),
        ("human", f"""
        ユーザー要求: {state['requirements']}
        POの要件定義: {state['po_output']}
        
        過去のフィードバック/修正指示: {feedback}
        """)
    ])
    res = (prompt | llm).invoke({})
    return {"architect_output": res.content}

def node_planner_critic(state: AgentState) -> AgentState:
    """[Role A3] Critic"""
    print("\n🔹 [Role A3] Critic Reviewing...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは「悪魔の代弁者 (Devil's Advocate)」を務めるシニアエンジニアです。
Architectの設計案に対して、意地悪な視点から厳しくレビューを行ってください。

特に以下の「エッジケース」を徹底的に指摘してください：
1. **境界値**: 空リスト, 0, 負数, 極端に大きな数。
2. **不正な型**: 数値期待の場所に文字列, None, 欠損データ。
3. **外部要因**: ファイルが存在しない, 権限がない, 文字コードエラー。
4. **セキュリティ**: インジェクション攻撃やリソース枯渇の可能性。

良い点を褒める必要はありません。リスクの列挙に集中してください。"""),
        ("human", f"Architectの設計案:\n{state['architect_output']}")
    ])
    res = (prompt | llm).invoke({})
    return {"critic_output": res.content}

def node_planner_reviser(state: AgentState) -> AgentState:
    """[Role A4] Reviser (仕様書作成)"""
    print("\n🔹 [Role A4] Reviser Compiling...")
    replan_cnt = state.get("replan_count", 0)
    MAX_REPLANS = 3
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""あなたは開発チームのリードエンジニア(Reviser)です。
PO、Architect、Criticの議論を統合し、開発者(Coder)とテスト担当(Tester)に渡すための**「最終仕様書(JSON)」**を作成してください。

現在の再設計回数: {replan_cnt} / {MAX_REPLANS}

**重要指示**:
1. Criticが指摘した「エッジケース」は、必ず `edge_cases` リストに含めてください。
2. Architectの設計に致命的な欠陥（実現不可能、矛盾）がある場合のみ、 `needs_replan=True` としてください。
3. 軽微な修正で済む場合は、仕様書内で修正を指示し、`needs_replan=False` として開発フェーズへ進めてください。
"""),
        ("human", f"""
        [PO 要件]: {state['po_output']}
        [Architect 設計]: {state['architect_output']}
        [Critic 指摘]: {state['critic_output']}
        """)
    ])
    
    chain = prompt | llm.with_structured_output(FinalSpec)
    result = chain.invoke({})
    
    # タイムアウト制御ロジック
    if result.needs_replan:
        if replan_cnt >= MAX_REPLANS:
            print(f"   ⚠️ Replan Limit Reached ({MAX_REPLANS}). Forcing proceed.")
            return {
                "design_plan": result.model_dump(),
                "feedback": f"Warning: Spec finalized after {MAX_REPLANS} replans. Issues may remain.",
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


# === Role B, C, D, E: Dev & QA Squad ===

def node_tester(state: AgentState) -> AgentState:
    """[Role B] Tester"""
    print("\n🔹 [Role B] Tester Running...")
    spec = state["design_plan"]
    fb = state.get("feedback", "")
    phase = state.get("current_phase", "dev")
    existing_test = state.get("test_code", "")
    
    # フェーズに応じた指示の出し分け
    if phase == "mutation":
        instruction = f"""
        【重要：Mutation Test Fix Phase】
        Role E (Mutation Tester) からの指摘:
        "{fb}"
        
        あなたのテストスイートは、このバグを見逃しました（Mutant Survived）。
        既存のテストケースは**絶対に削除せず**、この特定のバグを検知してFailさせる新しいテストケースを追加してください。
        アサーションを厳格にしてください。
        """
    else:
        instruction = f"""
        【新規開発フェーズ】
        仕様書に基づき、pytest用のテストコードを作成してください。
        これまでのフィードバック: {fb}
        
        以下の要件を満たすこと：
        1. **Happy Path**: 正常系のテスト。
        2. **Edge Cases**: 仕様書の `edge_cases` ({spec.get('edge_cases')}) を網羅するテスト。
        3. `pytest.mark.parametrize` を活用し、簡潔かつ網羅的に記述する。
        4. 実装ファイル名は `implementation.py` と仮定してインポートする。
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは品質保証(QA)のエキスパートです。
Pythonの `pytest` フレームワークを使用した、高品質で堅牢なテストコードを作成してください。
出力はPythonコードブロックのみを含めてください。"""),
        ("human", f"""
        仕様書JSON: {json.dumps(spec, ensure_ascii=False)}
        
        現在のテストコード(あれば):
        {existing_test}
        
        具体的な指示:
        {instruction}
        """)
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
        ("system", """あなたはGoogleスタイルのコーディング規約を遵守するPythonエンジニアです。
提供された「テストコード」をすべてパス(Green)させる実装コードを作成してください。

遵守事項:
1. **Type Hints**: 引数と戻り値には必ず型ヒントを付ける。
2. **Docstring**: 関数には挙動、引数、戻り値の説明を書く。
3. **Error Handling**: 仕様書にあるエッジケースでは適切に例外を投げるか処理する。
4. **Minimalism**: テストを通すために必要なコードだけを書く (KISS原則)。
"""),
        ("human", f"""
        仕様書: {json.dumps(spec, ensure_ascii=False)}
        
        テストコード(これをパスさせる):
        {test}
        
        現在の実装:
        {impl}
        
        フィードバック/エラーログ:
        {fb}
        """)
    ])
    
    res = (prompt | llm.with_structured_output(CodeOutput)).invoke({})
    return {"impl_code": res.impl_code}

def node_executor(state: AgentState) -> AgentState:
    """[Role D] Executor"""
    print("\n🔹 [Role D] Executor Running...")
    
    # ファイル保存
    with open("implementation.py", "w", encoding="utf-8") as f:
        f.write(state["impl_code"])
    with open("test_suite.py", "w", encoding="utf-8") as f:
        f.write(state["test_code"])
    
    # pytest実行
    try:
        # タイムアウトを少し長めに設定
        res = subprocess.run(
            ["pytest", "test_suite.py", "-v", "--tb=short"],
            capture_output=True, text=True, timeout=15
        )
        return {"test_result": res.stdout + res.stderr}
    except Exception as e:
        return {"test_result": f"Execution Error: {str(e)}"}

def node_reflector(state: AgentState) -> AgentState:
    """[Role D] Reflector"""
    print("\n🔹 [Role D] Reflector Running...")
    res = state["test_result"]
    phase = state["current_phase"]
    itr = state["iteration"]
    
    # 全体ループリミット
    if itr > 20:
        return {"next_action": "finish", "feedback": "Global Loop Limit reached."}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはCI/CDパイプラインの管理者です。
テスト実行ログを分析し、次のアクションを決定してください。

**判断基準**:
1. `FAILED` (エラーあり):
   - 実装ロジックのミス -> `retry_code` (Coderへ)
   - テストコードのミス/仕様との不整合 -> `retry_test` (Testerへ)
   - **仕様自体が実現不可能/矛盾している -> `replan` (Architectへ戻す)**

2. `PASSED` (全テスト合格):
   - 現在が 'dev' フェーズ -> `mutation_check` (Role Eへ進む)
   - 現在が 'mutation' フェーズ -> `finish` (完了)
"""),
        ("human", f"Current Phase: {phase}\n\nExecution Log:\n{res}")
    ])
    
    decision = (prompt | llm.with_structured_output(ReflectionOutput)).invoke({})
    print(f"   -> Decision: {decision.action}")
    
    new_state = {
        "feedback": decision.feedback,
        "next_action": decision.action,
        "iteration": itr + 1
    }
    if decision.action == "mutation_check":
        new_state["current_phase"] = "mutation"
        
    return new_state

def node_mutation_tester(state: AgentState) -> AgentState:
    """[Role E] Mutation Tester"""
    print("\n🔹 [Role E] Mutation Tester Running...")
    original_impl = state["impl_code"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは意地悪なミューテーションテスト・エンジニアです。
提供されたPythonコードに対して、**「文法エラー(SyntaxError)は起こさないが、論理的振る舞いが変わるバグ」**を1つだけ埋め込んだコード（ミュータント）を作成してください。

**やってはいけないこと (NG)**:
- インデントを崩す、閉じ括弧を消す等のSyntax Error。
- 関数名や引数名を変える（テストが動かなくなるため）。

**推奨される変更 (OK)**:
- 比較演算子の変更 (`<` → `<=`, `==` → `!=`)
- 算術演算子の変更 (`+` → `-`)
- 条件分岐の論理反転 (`if x:` → `if not x:`)
- 定数の変更 (`return 0` → `return 1`)
- 配列のインデックス変更 (`arr[0]` → `arr[1]`)
"""),
        ("human", f"元のコード:\n{original_impl}")
    ])
    
    mutant = (prompt | llm.with_structured_output(MutantOutput)).invoke({})
    
    # ファイル書き換え
    with open("implementation.py", "w", encoding="utf-8") as f:
        f.write(mutant.mutant_code)
        
    # テスト実行 (Quietモード)
    try:
        res = subprocess.run(["pytest", "test_suite.py", "-q"], capture_output=True, timeout=10)
        # return_code == 0 (Pass) -> バグがあるのにテストが通った -> Mutant Survived (検知失敗)
        mutant_survived = (res.returncode == 0)
    except:
        # エラーで落ちたなら検知できたとみなす
        mutant_survived = False
        
    # 元に戻す
    with open("implementation.py", "w", encoding="utf-8") as f:
        f.write(original_impl)
        
    if mutant_survived:
        print(f"   >>> ❌ Mutant Survived! ({mutant.mutation_description})")
        return {
            "feedback": f"ミューテーションテスト失敗: あなたのテストはバグ『{mutant.mutation_description}』を見逃しました。これを検知できるテストケースを追加してください。",
            "next_action": "retry_test",
            "mutation_logs": state["mutation_logs"] + ["Survived"]
        }
    else:
        print("   >>> ✅ Mutant Killed! (Test is robust)")
        return {
            "feedback": "Passed.",
            "next_action": "finish",
            "mutation_logs": state["mutation_logs"] + ["Killed"]
        }


# --- 4. グラフ構築 (LangGraph) ---

workflow = StateGraph(AgentState)

# ノード登録
workflow.add_node("planner_po", node_planner_po)
workflow.add_node("planner_architect", node_planner_architect)
workflow.add_node("planner_critic", node_planner_critic)
workflow.add_node("planner_reviser", node_planner_reviser)
workflow.add_node("tester", node_tester)
workflow.add_node("coder", node_coder)
workflow.add_node("executor", node_executor)
workflow.add_node("reflector", node_reflector)
workflow.add_node("mutation_tester", node_mutation_tester)

# エッジ接続: Role Aの流れ
workflow.set_entry_point("planner_po")
workflow.add_edge("planner_po", "planner_architect")
workflow.add_edge("planner_architect", "planner_critic")
workflow.add_edge("planner_critic", "planner_reviser")

# Reviserの分岐 (Replan or Proceed)
def router_reviser(state: AgentState):
    if state["next_action"] == "replan_internal":
        return "planner_architect"
    return "tester"

workflow.add_conditional_edges(
    "planner_reviser",
    router_reviser,
    {
        "planner_architect": "planner_architect",
        "tester": "tester"
    }
)

# 開発フェーズの流れ
workflow.add_edge("tester", "coder")
workflow.add_edge("coder", "executor")
workflow.add_edge("executor", "reflector")

# Reflectorの分岐 (全方位)
def router_reflector(state: AgentState):
    act = state["next_action"]
    if act == "retry_code": return "coder"
    elif act == "retry_test": return "tester"
    elif act == "replan": return "planner_architect"
    elif act == "mutation_check": return "mutation_tester"
    elif act == "finish": return END
    return END

workflow.add_conditional_edges(
    "reflector",
    router_reflector,
    {
        "coder": "coder",
        "tester": "tester",
        "planner_architect": "planner_architect",
        "mutation_tester": "mutation_tester",
        END: END
    }
)

# Mutation Testerの分岐
def router_mutation(state: AgentState):
    if state["next_action"] == "retry_test":
        return "tester"
    return END

workflow.add_conditional_edges(
    "mutation_tester",
    router_mutation,
    {
        "tester": "tester",
        END: END
    }
)

app = workflow.compile()


# --- 5. メイン実行 ---

if __name__ == "__main__":
    print("\n=======================================================")
    print(" 🚀 Multi-Agent Coder v5 (Refined Prompts) 🚀")
    print("=======================================================")
    
    user_task = input("\n>>> タスクを入力してください: ")
    if not user_task.strip():
        user_task = "CSVファイルを読み込み、'score'カラムの平均値を計算する関数。ファイル欠損や不正データのエラーハンドリングを実装せよ。"
        print(f"(入力なしのためデフォルト実行: {user_task})")
    
    initial = {
        "requirements": user_task,
        "po_output": "", "architect_output": "", "critic_output": "", "design_plan": {},
        "replan_count": 0,
        "test_code": "", "impl_code": "", "test_result": "", "feedback": "",
        "iteration": 0, "mutation_logs": [], "current_phase": "dev", "next_action": ""
    }
    
    try:
        for s in app.stream(initial):
            pass
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        
    print("\n=======================================================")
    print(" 🎉 Workflow Completed!")
    
    # 成果物の表示
    if os.path.exists("implementation.py"):
        print("\n--- 最終成果物: implementation.py ---")
        with open("implementation.py", "r", encoding="utf-8") as f:
            print(f.read())
            
    if os.path.exists("test_suite.py"):
        print("\n--- 最終テスト: test_suite.py ---")
        with open("test_suite.py", "r", encoding="utf-8") as f:
            print(f.read())
