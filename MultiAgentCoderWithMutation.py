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

# APIキーの設定 (環境変数に設定されていない場合はここで入力してください)
if "OPENAI_API_KEY" not in os.environ:
    # os.environ["OPENAI_API_KEY"] = "sk-..." 
    print("⚠️ Warning: OPENAI_API_KEY is not set in environment variables.")

# モデル設定 (議論とコード生成にはGPT-4oクラスを強く推奨)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# --- 1. State (共有メモリ) の定義 ---

class AgentState(TypedDict):
    requirements: str            # ユーザーの要求
    
    # Role A (Planning) の成果物
    po_output: str               # POの要件定義
    architect_output: str        # Architectの設計案
    critic_output: str           # Criticの指摘事項
    design_plan: Dict            # Reviserがまとめた最終仕様書(JSON)
    
    # Devフェーズの成果物
    test_code: str               # Role Bが書いたテストコード
    impl_code: str               # Role Cが書いた実装コード
    test_result: str             # Role D (Executor) の実行ログ
    feedback: str                # 各Roleからの修正指示・フィードバック
    
    # 制御用フラグ
    iteration: int               # 無限ループ防止用のカウンター
    mutation_logs: List[str]     # Role Eの試行結果ログ
    current_phase: str           # "dev" (開発) or "mutation" (品質保証)
    next_action: str             # ルーターが遷移先を決定するための識別子


# --- 2. 出力スキーマ (Pydantic Models) ---

# Role A4 (Reviser) 用: 仕様書構造
class FinalSpec(BaseModel):
    function_name: str = Field(description="実装する関数の名前(スネークケース)")
    inputs: List[Dict[str, str]] = Field(description="引数のリスト(名前と型)")
    output_type: str = Field(description="返り値の型")
    description: str = Field(description="関数の概要")
    requirements: List[str] = Field(description="詳細な機能要件リスト")
    edge_cases: List[str] = Field(description="考慮すべきエッジケース・異常系のリスト")
    
    # 内部Replan判定用
    needs_replan: bool = Field(description="Criticの指摘が致命的で、Architectによる再設計が必要な場合はTrue")
    replan_reason: str = Field(description="再設計が必要な場合の理由")

# Role B (Tester) 用
class TestOutput(BaseModel):
    thought_process: str = Field(description="テスト設計の意図、エッジケースの網羅方法")
    test_code: str = Field(description="pytestで実行可能な完全なPythonコード")

# Role C (Coder) 用
class CodeOutput(BaseModel):
    thought_process: str = Field(description="実装の方針")
    impl_code: str = Field(description="仕様を満たしテストを通すPython実装コード")

# Role D (Reflector) 用
class ReflectionOutput(BaseModel):
    analysis: str = Field(description="ログの分析結果")
    action: str = Field(description="次のアクション: 'retry_code', 'retry_test', 'replan', 'mutation_check', 'finish'")
    feedback: str = Field(description="次の担当者への具体的な指示")

# Role E (Mutation Tester) 用
class MutantOutput(BaseModel):
    mutant_code: str = Field(description="バグを埋め込んだ実装コード")
    mutation_description: str = Field(description="埋め込んだバグの説明 (例: '<' を '<=' に変更)")


# --- 3. ノード関数 (Agentの実装) ---

# === Role A: Planning Squad ===

def node_planner_po(state: AgentState) -> AgentState:
    """[Role A1] Product Owner: 要求分析"""
    print("\n🔹 [Role A1] Product Owner Analyzing...")
    req = state["requirements"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは優秀なプロダクトオーナーです。
ユーザーの要望を分析し、開発すべき機能の「目的」「背景」「主要なユーザーストーリー」を明確に定義してください。
技術的な詳細（どう実装するか）には踏み込まず、「何を作るか（What）」に集中してください。"""),
        ("human", f"User Request: {req}")
    ])
    response = (prompt | llm).invoke({})
    
    return {
        "po_output": response.content,
        "iteration": 0,
        "mutation_logs": [],
        "current_phase": "dev",
        "feedback": ""
    }

def node_planner_architect(state: AgentState) -> AgentState:
    """[Role A2] Architect: 技術設計"""
    # ReflectorやReviserから戻ってきた場合のフィードバックを取得
    feedback = state.get("feedback", "")
    print(f"\n🔹 [Role A2] Architect Designing... (Feedback: {feedback})")
    
    req = state["requirements"]
    po_out = state["po_output"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは熟練のソフトウェアアーキテクトです。
POの定義に基づき、Pythonでの実装方針（関数構成、利用ライブラリ、アルゴリズム概要）を設計してください。
もしフィードバック（手戻り理由）がある場合は、それを解消するように設計を見直してください。"""),
        ("human", f"User Request: {req}\n\nPO Definition:\n{po_out}\n\nFeedback/Issues:\n{feedback}")
    ])
    response = (prompt | llm).invoke({})
    
    return {"architect_output": response.content}

def node_planner_critic(state: AgentState) -> AgentState:
    """[Role A3] Critic (Devil's Advocate): 設計批判"""
    print("\n🔹 [Role A3] Critic Reviewing...")
    arch_out = state["architect_output"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは「悪魔の代弁者（Devil's Advocate）」を務めるシニアエンジニアです。
Architectの設計案に対して、以下の観点で厳しく指摘を行ってください。
1. **エッジケース**: 空入力、巨大な数値、不正な型、ファイル欠損など。
2. **論理的欠陥**: アルゴリズムの不備や無限ループの可能性。
3. **セキュリティ**: 脆弱性の可能性。

褒める必要はありません。リスクを列挙してください。"""),
        ("human", f"Architect Design:\n{arch_out}")
    ])
    response = (prompt | llm).invoke({})
    
    return {"critic_output": response.content}

def node_planner_reviser(state: AgentState) -> AgentState:
    """[Role A4] Reviser: 仕様書作成と再設計判定"""
    print("\n🔹 [Role A4] Reviser Compiling Spec...")
    context = f"""
    [User Request]: {state['requirements']}
    [PO]: {state['po_output']}
    [Architect]: {state['architect_output']}
    [Critic]: {state['critic_output']}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは議論をまとめるリバイザーです。
これまでの議論を統合し、TesterとDeveloperに渡すための**最終的な仕様書(JSON)**を作成してください。

重要な指示:
1. Criticが指摘した「エッジケース」は必ず `edge_cases` リストに含めてください。
2. もしCriticの指摘が致命的で、現在のArchitect案では修正不可能（根本的な設計ミス）と判断した場合は、
   `needs_replan` を True にし、`replan_reason` に理由を記述してください。
"""),
        ("human", context)
    ])
    
    chain = prompt | llm.with_structured_output(FinalSpec)
    result = chain.invoke({})
    
    # 再設計が必要な場合
    if result.needs_replan:
        print(f"   ⚠️ Reviser Requesting Replan: {result.replan_reason}")
        return {
            "design_plan": {}, # 空にする
            "feedback": result.replan_reason,
            "next_action": "replan_internal" # Role A内ループフラグ
        }
    
    print(f"   -> Spec Finalized: {result.function_name}")
    return {
        "design_plan": result.model_dump(),
        "feedback": "", 
        "next_action": "proceed" # Role Bへ進むフラグ
    }


# === Role B, C, D, E: Development & QA Squad ===

def node_tester(state: AgentState) -> AgentState:
    """[Role B] Test Architect: テスト作成"""
    print("\n🔹 [Role B] Test Architect Running...")
    spec = state["design_plan"]
    feedback = state.get("feedback", "")
    current_phase = state.get("current_phase", "dev")
    existing_test = state.get("test_code", "")
    
    # フェーズによる指示の切り替え
    if current_phase == "mutation":
        instruction = f"""
        【品質保証フェーズ (Mutation Check)】
        Role E (Mutation Tester) からの指摘: "{feedback}"
        
        これは「実装に意図的なバグを埋め込んでもテストが合格してしまった（見逃した）」ことを意味します。
        既存のテストスイート（正常系・エッジケース）は維持したまま、
        **この特定のバグを検知してFailさせる新しいテストケース**を追加してください。
        """
    else:
        instruction = f"""
        【開発フェーズ (Initial TDD)】
        仕様書に基づいてテストコードを作成してください。
        これまでのフィードバック: {feedback}
        
        以下の2つを網羅すること:
        1. **Happy Path**: 正常動作確認。
        2. **Edge Cases**: 仕様書の `edge_cases` リスト ({spec.get('edge_cases')}) にある異常系処理。
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはpytestのエキスパートです。以下のルールを守ってください。
- `import pytest` を必ず含める。
- 実装コードは `implementation.py` にあると仮定し、`from implementation import *` を行う。
- 全てのテスト関数は `test_` で始める。
- Pythonコードブロックのみを出力する。"""),
        ("human", f"仕様書: {json.dumps(spec, ensure_ascii=False)}\n既存テスト: {existing_test}\n\n指示: {instruction}")
    ])
    
    chain = prompt | llm.with_structured_output(TestOutput)
    result = chain.invoke({})
    
    print(f"   -> Role B Thought: {result.thought_process}")
    return {"test_code": result.test_code, "feedback": ""}

def node_coder(state: AgentState) -> AgentState:
    """[Role C] Developer: 実装"""
    print("\n🔹 [Role C] Developer Running...")
    spec = state["design_plan"]
    test_code = state["test_code"]
    feedback = state.get("feedback", "")
    current_impl = state.get("impl_code", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはPythonエンジニアです。
提供された「テストコード」をすべてパス(Green)することだけを目標に実装してください。
YAGNI原則に従い、テストを通すための最小限の実装を行ってください。"""),
        ("human", f"""
        仕様: {json.dumps(spec, ensure_ascii=False)}
        テストコード: {test_code}
        現在の実装: {current_impl}
        エラー/フィードバック: {feedback}
        """)
    ])
    
    chain = prompt | llm.with_structured_output(CodeOutput)
    result = chain.invoke({})
    
    print(f"   -> Role C Thought: {result.thought_process}")
    return {"impl_code": result.impl_code}

def node_executor(state: AgentState) -> AgentState:
    """[Role D] CI Runner: 実行"""
    print("\n🔹 [Role D] CI Runner Running...")
    
    # ファイル書き出し
    with open("implementation.py", "w", encoding="utf-8") as f:
        f.write(state["impl_code"])
    with open("test_suite.py", "w", encoding="utf-8") as f:
        f.write(state["test_code"])
        
    # pytest実行
    try:
        # -v: 詳細, --tb=short: トレースバック短縮
        result = subprocess.run(
            ["pytest", "test_suite.py", "-v", "--tb=short"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        return_code = result.returncode
    except Exception as e:
        output = str(e)
        return_code = 1
        
    print(f"   -> Pytest Return Code: {return_code}")
    return {"test_result": output}

def node_reflector(state: AgentState) -> AgentState:
    """[Role D] Reflector: 判定と振り分け"""
    print("\n🔹 [Role D] Reflector Running...")
    output = state["test_result"]
    current_phase = state.get("current_phase", "dev")
    iteration = state["iteration"]
    
    # ループ制限 (安全装置)
    if iteration > 20:
        return {"feedback": "Loop limit reached.", "next_action": "finish", "iteration": iteration + 1}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはCIログを分析するリードエンジニアです。
状況に応じて次のアクション(`action`)を決定してください。

1. **テスト失敗 (Error/Fail)**:
   - 実装ミスと思われる場合 -> `retry_code`
   - テストコード自体や想定が誤っている場合 -> `retry_test`
   - **仕様自体に矛盾や無理がある場合** -> `replan` (Architectに戻す)

2. **テスト成功 (Pass)**:
   - 現在が 'dev' フェーズ -> `mutation_check` (品質保証へ)
   - 現在が 'mutation' フェーズ -> `finish` (完了)
"""),
        ("human", f"Current Phase: {current_phase}\nLog:\n{output}")
    ])
    
    chain = prompt | llm.with_structured_output(ReflectionOutput)
    result = chain.invoke({})
    
    print(f"   -> Decision: {result.action} ({result.analysis})")
    
    new_state = {
        "feedback": result.feedback,
        "next_action": result.action,
        "iteration": iteration + 1
    }
    
    if result.action == "mutation_check":
        new_state["current_phase"] = "mutation"
        
    return new_state

def node_mutation_tester(state: AgentState) -> AgentState:
    """[Role E] Mutation Tester: 品質監査"""
    print("\n🔹 [Role E] Mutation Tester Running...")
    original_impl = state["impl_code"]
    
    # 1. ミュータント生成
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは意地悪なQAエンジニアです。
提供された正常なコードに対し、**「文法は正しいが論理が微妙に間違っているバグ」を1つだけ埋め込んだコード**（ミュータント）を作成してください。
目的: 現在のテストスイートがこのバグを検知できるか試すこと。
例: 境界値の変更 (<= を < に)、条件反転、+1の削除など。"""),
        ("human", f"Code:\n{original_impl}")
    ])
    chain = prompt | llm.with_structured_output(MutantOutput)
    mutant = chain.invoke({})
    print(f"   -> Generated Mutant: {mutant.mutation_description}")
    
    # 2. ミュータント適用
    with open("implementation.py", "w", encoding="utf-8") as f:
        f.write(mutant.mutant_code)
        
    # 3. テスト実行
    try:
        # -q: 結果のみ表示
        res = subprocess.run(["pytest", "test_suite.py", "-q"], capture_output=True, timeout=5)
        # return_code == 0 (Pass) -> バグがあるのにテストが通った -> Mutant Survived (検知失敗)
        mutant_survived = (res.returncode == 0)
    except:
        mutant_survived = False
        
    # 4. 元に戻す (重要)
    with open("implementation.py", "w", encoding="utf-8") as f:
        f.write(original_impl)
        
    # 5. 結果判定
    if mutant_survived:
        print("   >>> ❌ Mutant Survived! (Tests need improvement)")
        feedback = f"ミューテーションテスト失敗: バグ『{mutant.mutation_description}』が埋め込まれましたが、テストはPassしてしまいました。このバグを検知できるテストを追加してください。"
        return {
            "feedback": feedback,
            "next_action": "retry_test", # Role Bに戻す
            "mutation_logs": state["mutation_logs"] + [f"Survived: {mutant.mutation_description}"]
        }
    else:
        print("   >>> ✅ Mutant Killed! (Tests are robust)")
        return {
            "feedback": "Passed.",
            "next_action": "finish",
            "mutation_logs": state["mutation_logs"] + [f"Killed: {mutant.mutation_description}"]
        }


# --- 4. グラフの構築 (Workflow Definition) ---

workflow = StateGraph(AgentState)

# ノードの登録
workflow.add_node("planner_po", node_planner_po)
workflow.add_node("planner_architect", node_planner_architect)
workflow.add_node("planner_critic", node_planner_critic)
workflow.add_node("planner_reviser", node_planner_reviser)
workflow.add_node("tester", node_tester)
workflow.add_node("coder", node_coder)
workflow.add_node("executor", node_executor)
workflow.add_node("reflector", node_reflector)
workflow.add_node("mutation_tester", node_mutation_tester)

# エッジ: エントリーからRole Aの流れ
workflow.set_entry_point("planner_po")
workflow.add_edge("planner_po", "planner_architect")
workflow.add_edge("planner_architect", "planner_critic")
workflow.add_edge("planner_critic", "planner_reviser")

# 条件付きエッジ 1: Reviser -> (Architect 戻り or Tester 進み)
def router_reviser(state: AgentState):
    if state["next_action"] == "replan_internal":
        return "planner_architect" # 内部Replan
    return "tester" # 承認

workflow.add_conditional_edges(
    "planner_reviser",
    router_reviser,
    {
        "planner_architect": "planner_architect",
        "tester": "tester"
    }
)

# エッジ: Devフェーズのメインストリーム
workflow.add_edge("tester", "coder")
workflow.add_edge("coder", "executor")
workflow.add_edge("executor", "reflector")

# 条件付きエッジ 2: Reflectorの分岐ロジック (全方位対応)
def router_reflector(state: AgentState):
    action = state["next_action"]
    if action == "retry_code":
        return "coder"
    elif action == "retry_test":
        return "tester"
    elif action == "replan":
        return "planner_architect" # 指摘③: 仕様からの作り直し
    elif action == "mutation_check":
        return "mutation_tester"   # 指摘②: 品質保証へ
    elif action == "finish":
        return END
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

# 条件付きエッジ 3: Mutation Tester -> (Tester 戻り or 終了)
def router_mutation(state: AgentState):
    if state["next_action"] == "retry_test":
        return "tester" # 指摘④: 検知失敗ならBに戻る
    return END

workflow.add_conditional_edges(
    "mutation_tester",
    router_mutation,
    {
        "tester": "tester",
        END: END
    }
)

# グラフのコンパイル
app = workflow.compile()


# --- 5. 実行エントリーポイント ---

if __name__ == "__main__":
    print("\n=======================================================")
    print(" 🚀 Multi-Agent Coder with Mutation Testing (Full) 🚀")
    print("=======================================================")
    print("構成: PO -> Architect -> Critic -> Reviser -> Tester -> Coder -> QA(Mutation)")
    
    # ユーザー入力
    user_task = input("\n>>> 開発タスクを入力してください: ")
    if not user_task.strip():
        # デフォルトタスク
        user_task = "CSVファイル(data.csv)を読み込み、'score'列の平均値を計算する関数。ファイル欠損、空データ、列不足のエラーハンドリングを行うこと。"
        print(f"(入力なしのため、デフォルトタスクを実行します: {user_task})")
    
    # 初期状態
    initial_state = {
        "requirements": user_task,
        "po_output": "",
        "architect_output": "",
        "critic_output": "",
        "design_plan": {},
        "test_code": "",
        "impl_code": "",
        "test_result": "",
        "feedback": "",
        "iteration": 0,
        "mutation_logs": [],
        "current_phase": "dev",
        "next_action": ""
    }
    
    # ストリーミング実行
    try:
        for event in app.stream(initial_state):
            # LangGraphは各ステップの状態を出力しますが、
            # 詳細なログは各Node関数のprint文で行っています
            pass
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        
    print("\n=======================================================")
    print(" 🎉 Workflow Completed!")
    print("=======================================================")
    
    # 結果ファイルの確認
    if os.path.exists("implementation.py"):
        print("\n--- Final Implementation (implementation.py) ---")
        with open("implementation.py", "r", encoding="utf-8") as f:
            print(f.read())
            
    if os.path.exists("test_suite.py"):
        print("\n--- Final Test Suite (test_suite.py) ---")
        with open("test_suite.py", "r", encoding="utf-8") as f:
            print(f.read())
