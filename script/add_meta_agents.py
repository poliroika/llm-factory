#!/usr/bin/env python3
"""
Add Meta-Agents to all dataset folders

Generates essential orchestration/management agents for the 'meta' domain
using LLM, then saves meta.json to all dataset folders.

Each folder has its own generation settings:
- agents_eng: English, temperature 0.7
- agents_rus: Russian (persona, description, display_name), temperature 0.7
- agents_temp_03_big: English, temperature 0.3
"""

import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_API_KEY = "very-secure-key-sber1love"
DEFAULT_BASE_URL = "https://keyword-cameras-homework-analyze.trycloudflare.com/v1"
DEFAULT_MODEL = "gpt-oss"

MAX_RETRIES = 3

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
AGENTS_NORM_DIR = DATASET_DIR / "agents_norm"

META_DOMAIN = "meta"
META_FILENAME = "meta.json"

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetConfig:
    """Configuration for each dataset folder."""
    name: str
    path: Path
    language: str  # "en" or "ru"
    temperature: float


DATASET_CONFIGS = [
    DatasetConfig(
        name="agents_eng",
        path=AGENTS_NORM_DIR / "agents_eng",
        language="en",
        temperature=0.7
    ),
    DatasetConfig(
        name="agents_rus",
        path=AGENTS_NORM_DIR / "agents_rus",
        language="ru",
        temperature=0.7
    ),
    DatasetConfig(
        name="agents_temp_03_big",
        path=AGENTS_NORM_DIR / "agents_temp_03_big",
        language="en",
        temperature=0.3
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# META-AGENTS SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

META_AGENTS_SPEC_EN = """
Generate the following essential meta-agents for multi-agent orchestration systems.
Each agent must be in the 'meta' domain and have appropriate role_id.

## REQUIRED META-AGENTS TO GENERATE:

### 1. Orchestrator / Manager (role_id: coordinator)
- Decomposes tasks, selects agent subgraphs, decides call order, stops/restarts branches
- Holds the "contract" of the entire pipeline: what is the final answer, when is it sufficient
- Tools: none (pure reasoning)

### 2. Router / Selector (role_id: router)
- Fast selection of "which agent(s) to call" based on request and context (often top-k)
- Can be part of Orchestrator, but separate Router is useful for speed and simplicity
- Tools: none

### 3. Summarizer / Context Compressor (role_id: summarizer)
- Compresses history/intermediate results, makes brief state for next steps
- Two modes: "context compression" and "final extraction"
- Tools: none

### 4. Tool Executor / Tool Manager (role_id: tool_runner)
- Single layer for tool-calls: argument validation, retries, timeouts, result normalization
- Separates "agent decided to call tool" from "system executed call safely"
- Tools: sandbox_exec, http_request, fs, sql_query

### 5. Verifier / Critic (role_id: verifier)
- Checks result (logic, contradictions, compliance with requirements, format)
- Returns structured report: issues, severity, fix_suggestions
- Tools: none

### 6. Safety / Policy Guard (role_id: safety_guard)
- Filters unsafe requests/responses, monitors secret leaks, forbidden actions
- Works on input (triage) and output (red-team check)
- Tools: none

### 7. Memory Manager (role_id: memory_manager)
- Extracts relevant memory (RAG/profile), decides what to write to long-term memory, what to forget
- Handles multiple memory sources and privacy policy
- Tools: rag_query, cache

### 8. Planner (role_id: planner)
- Creates step plan (tasks, subtasks, completion criteria)
- Sometimes combined with Orchestrator, but separate Planner is easier to test/replace
- Tools: none

### 9. Evaluator / Scorer (role_id: evaluator)
- Scores intermediate candidates (quality score, confidence, coverage)
- Useful for "best-of-n", A/B, RL/auto-optimization of graph
- Tools: metrics

### 10. Recovery / Fallback Handler (role_id: recovery_handler)
- If branch failed/stuck: changes strategy (other agents, different format, fewer tools)
- Provides graceful degradation
- Tools: none

### 11. State Keeper / Logger (role_id: state_keeper)
- Normalizes and saves pipeline events: who said what, which tool-calls, which prompt/model versions
- Critical for debugging and reproducibility
- Tools: cache, metrics
"""

META_AGENTS_SPEC_RU = """
Сгенерируй следующие мета-агенты для систем оркестрации многоагентных систем.
Каждый агент должен быть в домене 'meta' и иметь соответствующий role_id.

ВАЖНО: Поля "display_name", "persona" и "description" должны быть на РУССКОМ языке!

## НЕОБХОДИМЫЕ МЕТА-АГЕНТЫ:

### 1. Оркестратор / Менеджер (role_id: coordinator)
- Декомпозирует задачи, выбирает подграфы агентов, определяет порядок вызовов, останавливает/перезапускает ветки
- Держит "контракт" всего пайплайна: что является финальным ответом, когда он достаточен
- Tools: none (чистое рассуждение)

### 2. Маршрутизатор / Селектор (role_id: router)
- Быстрый выбор "какого агента(ов) вызвать" на основе запроса и контекста (часто top-k)
- Может быть частью Оркестратора, но отдельный Маршрутизатор полезен для скорости
- Tools: none

### 3. Суммаризатор / Компрессор контекста (role_id: summarizer)
- Сжимает историю/промежуточные результаты, делает краткое состояние для следующих шагов
- Два режима: "сжатие контекста" и "финальная экстракция"
- Tools: none

### 4. Исполнитель инструментов / Менеджер инструментов (role_id: tool_runner)
- Единый слой для tool-calls: валидация аргументов, ретраи, таймауты, нормализация результатов
- Разделяет "агент решил вызвать tool" от "система безопасно выполнила вызов"
- Tools: sandbox_exec, http_request, fs, sql_query

### 5. Верификатор / Критик (role_id: verifier)
- Проверяет результат (логика, противоречия, соответствие требованиям, формат)
- Возвращает структурированный отчёт: проблемы, серьёзность, предложения по исправлению
- Tools: none

### 6. Охранник безопасности / Политик (role_id: safety_guard)
- Фильтрует небезопасные запросы/ответы, мониторит утечки секретов, запрещённые действия
- Работает на входе (triage) и выходе (red-team check)
- Tools: none

### 7. Менеджер памяти (role_id: memory_manager)
- Извлекает релевантную память (RAG/профиль), решает что записать в долгосрочную память, что забыть
- Работает с несколькими источниками памяти и политикой приватности
- Tools: rag_query, cache

### 8. Планировщик (role_id: planner)
- Создаёт пошаговый план (задачи, подзадачи, критерии завершения)
- Иногда совмещён с Оркестратором, но отдельный Планировщик проще тестировать/заменять
- Tools: none

### 9. Оценщик / Скорер (role_id: evaluator)
- Оценивает промежуточных кандидатов (оценка качества, уверенность, покрытие)
- Полезен для "best-of-n", A/B, RL/авто-оптимизации графа
- Tools: metrics

### 10. Обработчик восстановления / Fallback (role_id: recovery_handler)
- Если ветка сломалась/застряла: меняет стратегию (другие агенты, другой формат, меньше tools)
- Обеспечивает graceful degradation
- Tools: none

### 11. Хранитель состояния / Логгер (role_id: state_keeper)
- Нормализует и сохраняет события пайплайна: кто что сказал, какие tool-calls, какие версии промптов/моделей
- Критичен для дебага и воспроизводимости
- Tools: cache, metrics
"""


# ═══════════════════════════════════════════════════════════════════════════════
# AVAILABLE TOOLS (for reference in prompt)
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_TOOLS = [
    "calculator", "web_browse", "web_search", "rag_query", "http_request",
    "sandbox_exec", "fs", "vcs_git", "sql_query", "spreadsheet", "document",
    "pdf", "presentation", "image", "email", "calendar", "contacts", "cache", "metrics"
]


# ═══════════════════════════════════════════════════════════════════════════════
# LLM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_generation_prompt(language: str) -> tuple[str, str]:
    """Build system and user prompts for meta-agent generation."""
    
    if language == "ru":
        system_prompt = f"""Ты эксперт по многоагентным AI системам. Генерируй определения AgentSpec для мета-агентов.

## Схема AgentSpec

{{
  "agents": [
    {{
      "agent_id": "string",           // уникальный slug: meta_<role>_<specific_name> например "meta_coordinator_orchestrator"
      "display_name": "string",       // человекочитаемое имя НА РУССКОМ ЯЗЫКЕ
      "persona": "string",            // 1-3 предложения описывающие характер/экспертизу агента НА РУССКОМ ЯЗЫКЕ
      "description": "string",        // детальное описание что делает агент и ожидаемый формат вывода НА РУССКОМ ЯЗЫКЕ
      "role_id": "string",            // один из: coordinator, router, summarizer, tool_runner, verifier, safety_guard, memory_manager, planner, evaluator, recovery_handler, state_keeper
      "domain": "meta",               // ВСЕГДА "meta" для мета-агентов
      "tools": [],                    // массив ID инструментов из: {', '.join(AVAILABLE_TOOLS)}
      "input_schema": {{}},           // JSON Schema для входа (оставь пустым для гибкости)
      "output_schema": {{}},          // JSON Schema для выхода (оставь пустым для гибкости)
      "raw": {{}}                     // пусто
    }}
  ]
}}

## ПРАВИЛА

1. Сгенерируй ВСЕ 11 мета-агентов указанных в сообщении пользователя
2. Каждый agent_id должен начинаться с "meta_" и быть уникальным
3. role_id должен соответствовать указанной роли для каждого агента
4. domain ДОЛЖЕН быть "meta" для всех агентов
5. ВАЖНО: display_name, persona и description должны быть на РУССКОМ языке!
6. Добавляй tools только когда агенту реально нужны внешние возможности
7. Оставляй input_schema и output_schema пустыми (это гибкие мета-агенты)

## ФОРМАТ ВЫВОДА
Верни ТОЛЬКО валидный JSON с массивом agents. Без markdown, без пояснений вне JSON.
"""
        user_prompt = META_AGENTS_SPEC_RU + """

Сгенерируй все 11 мета-агентов сейчас. Верни только валидный JSON.
ПОМНИ: display_name, persona и description на РУССКОМ языке!
"""
    else:
        system_prompt = f"""You are an expert in multi-agent AI systems. Generate AgentSpec definitions for meta-agents.

## AgentSpec Schema

{{
  "agents": [
    {{
      "agent_id": "string",           // unique slug: meta_<role>_<specific_name> e.g. "meta_coordinator_orchestrator"
      "display_name": "string",       // human-readable name in English
      "persona": "string",            // 1-3 sentences describing the agent's character/expertise
      "description": "string",        // detailed description of what the agent does and expected output format
      "role_id": "string",            // one of: coordinator, router, summarizer, tool_runner, verifier, safety_guard, memory_manager, planner, evaluator, recovery_handler, state_keeper
      "domain": "meta",               // ALWAYS "meta" for meta-agents
      "tools": [],                    // array of tool IDs from: {', '.join(AVAILABLE_TOOLS)}
      "input_schema": {{}},           // JSON Schema for input (keep empty for flexibility)
      "output_schema": {{}},          // JSON Schema for output (keep empty for flexibility)
      "raw": {{}}                     // empty
    }}
  ]
}}

## RULES

1. Generate ALL 11 meta-agents specified in the user message
2. Each agent_id must start with "meta_" and be unique
3. role_id must match the specified role for each agent
4. domain MUST be "meta" for all agents
5. Provide detailed, professional persona and description
6. Only add tools when the agent genuinely needs external capabilities
7. Keep input_schema and output_schema empty (these are flexible meta-agents)

## OUTPUT FORMAT
Return ONLY valid JSON with the agents array. No markdown, no explanations outside JSON.
"""
        user_prompt = META_AGENTS_SPEC_EN + """

Generate all 11 meta-agents now. Return only valid JSON.
"""
    
    return system_prompt, user_prompt


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code block
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


def generate_meta_agents(
    config: DatasetConfig,
    api_key: str,
    base_url: str,
    model: str
) -> list[dict]:
    """Generate meta-agents using LLM with dataset-specific settings."""
    
    lang_label = "Russian" if config.language == "ru" else "English"
    console.print(f"\n[bold blue]🤖 Generating for {config.name} ({lang_label}, temp={config.temperature})...[/bold blue]")
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=config.temperature,
        max_tokens=8000
    )
    
    system_prompt, user_prompt = build_generation_prompt(config.language)
    
    for attempt in range(MAX_RETRIES):
        try:
            console.print(f"   Attempt {attempt + 1}/{MAX_RETRIES}...")
            
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            result = extract_json(response.content)
            
            if result and "agents" in result:
                agents = result["agents"]
                console.print(f"   [green]✓ Generated {len(agents)} meta-agents[/green]")
                return agents
            else:
                console.print(f"   [yellow]⚠ Could not parse JSON, retrying...[/yellow]")
                
        except Exception as e:
            console.print(f"   [red]✗ Error: {e}[/red]")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
    
    console.print(f"[red]✗ Failed to generate for {config.name} after all retries[/red]")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_agent(agent: dict) -> dict:
    """Validate and normalize a single agent."""
    # Ensure domain is 'meta'
    agent['domain'] = META_DOMAIN
    
    # Ensure agent_id starts with meta_
    if not agent.get('agent_id', '').startswith('meta_'):
        agent['agent_id'] = 'meta_' + agent.get('agent_id', 'unknown')
    
    # Ensure required fields exist
    agent.setdefault('display_name', agent['agent_id'].replace('_', ' ').title())
    agent.setdefault('persona', '')
    agent.setdefault('description', '')
    agent.setdefault('role_id', 'coordinator')
    agent.setdefault('tools', [])
    agent.setdefault('input_schema', {})
    agent.setdefault('output_schema', {})
    agent.setdefault('raw', {})
    
    # Filter tools to only valid ones
    agent['tools'] = [t for t in agent.get('tools', []) if t in AVAILABLE_TOOLS]
    
    return agent


def build_meta_data(agents: list[dict]) -> dict:
    """Build the meta.json data structure."""
    # Validate and normalize all agents
    validated_agents = [validate_agent(agent) for agent in agents]
    
    # Remove duplicates by agent_id
    seen_ids = set()
    unique_agents = []
    for agent in validated_agents:
        if agent['agent_id'] not in seen_ids:
            unique_agents.append(agent)
            seen_ids.add(agent['agent_id'])
    
    return {
        "domain": META_DOMAIN,
        "generated_at": datetime.now().isoformat(),
        "total_agents": len(unique_agents),
        "agents": unique_agents
    }


def save_meta_file(data: dict, folder: Path) -> Path | None:
    """Save meta.json to a dataset folder."""
    if not folder.exists():
        console.print(f"   [yellow]⚠ Creating folder: {folder.name}[/yellow]")
        folder.mkdir(parents=True, exist_ok=True)
    
    output_path = folder / META_FILENAME
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        console.print(f"   [green]✓ Saved: {folder.name}/{META_FILENAME}[/green]")
        return output_path
        
    except Exception as e:
        console.print(f"   [red]✗ Error saving {folder.name}: {e}[/red]")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate meta-agents and save to all dataset folders"
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API base URL"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name"
    )
    parser.add_argument(
        "--folder",
        choices=["agents_eng", "agents_rus", "agents_temp_03_big"],
        default=None,
        help="Process only specific folder (default: all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    console.print(Panel.fit(
        "[bold]Meta-Agent Generator[/bold]\n"
        "Generates orchestration agents with per-folder settings",
        border_style="blue"
    ))
    
    # Filter configs if specific folder requested
    configs_to_process = DATASET_CONFIGS
    if args.folder:
        configs_to_process = [c for c in DATASET_CONFIGS if c.name == args.folder]
    
    # Show target folders
    console.print("\n[bold]Target folders:[/bold]")
    for config in configs_to_process:
        exists = "✓" if config.path.exists() else "○"
        lang = "🇷🇺 Russian" if config.language == "ru" else "🇬🇧 English"
        console.print(f"   {exists} {config.name} ({lang}, temp={config.temperature})")
    
    results = []
    
    # Process each folder with its own settings
    for config in configs_to_process:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Processing: {config.name}[/bold]")
        console.print(f"{'='*60}")
        
        # Generate agents for this config
        agents = generate_meta_agents(
            config=config,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
        
        if not agents:
            console.print(f"[red]✗ No agents generated for {config.name}, skipping[/red]")
            results.append((config.name, 0, None))
            continue
        
        # Build and save data
        data = build_meta_data(agents)
        output_path = save_meta_file(data, config.path)
        
        results.append((config.name, len(agents), output_path))
    
    # Final stats
    console.print("\n" + "=" * 60)
    console.print("[bold]📊 SUMMARY[/bold]")
    console.print("=" * 60)
    
    for name, count, path in results:
        if path:
            console.print(f"   [green]✓[/green] {name}: {count} agents → {path.relative_to(BASE_DIR)}")
        else:
            console.print(f"   [red]✗[/red] {name}: failed")
    
    console.print(f"\n   Domain: {META_DOMAIN}")
    console.print("=" * 60)


if __name__ == "__main__":
    main()
