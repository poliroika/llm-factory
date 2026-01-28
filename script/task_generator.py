#!/usr/bin/env python3
"""
Task Generator - Generate training tasks for each agent

For each agent in the dataset, generates 4-5 unique tasks that this specific
agent would handle best. Creates a synthetic dataset for training.

Steps:
1. Merge all domain files from agents_norm into unified dataset files
2. For each agent, generate 4-5 tasks via LLM
3. Save results with progress tracking and resume capability
"""

import json
import asyncio
import argparse
import traceback as tb
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_API_KEY = "very-secure-key-sber1love"
DEFAULT_BASE_URL = "https://keyword-cameras-homework-analyze.trycloudflare.com/v1"
DEFAULT_MODEL = "gpt-oss"

TEMPERATURE = 0.7
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300
DEFAULT_PARALLEL = 10
TASKS_PER_AGENT = 11

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
AGENTS_NORM_DIR = DATASET_DIR / "agents_norm"
OUTPUT_DIR = DATASET_DIR / "agent_tasks"
PROGRESS_FILE = SCRIPT_DIR / ".task_generator_progress.json"
LOG_FILE = SCRIPT_DIR / "task_generator.log"

# Dataset folders to process
DATASET_FOLDERS = ["agents_eng", "agents_rus", "agents_temp_03_big"]

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    """Log message to file."""
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level}] {msg}\n")


def log_llm(msg: str):
    """Log LLM-specific debug info to file."""
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [LLM] {msg}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentEntry:
    """Represents an agent with its source dataset."""
    agent_id: str
    display_name: str
    persona: str
    description: str
    role_id: str
    domain: str
    tools: list[str]
    dataset: str  # which dataset folder this came from
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "persona": self.persona,
            "description": self.description,
            "role_id": self.role_id,
            "domain": self.domain,
            "tools": self.tools,
            "dataset": self.dataset
        }


@dataclass
class AgentTasks:
    """Tasks generated for an agent."""
    agent_id: str
    dataset: str
    tasks: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "dataset": self.dataset,
            "tasks": self.tasks
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_progress() -> dict:
    """Load progress from checkpoint file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "merged_datasets": [],  # list of dataset folder names already merged
        "completed_agents": {},  # dataset -> list of agent_ids with generated tasks
        "stats": {
            "total_agents": 0,
            "tasks_generated": 0,
            "errors": 0
        }
    }


def save_progress(progress: dict):
    """Save progress to checkpoint file."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def clear_progress():
    """Clear progress file."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def merge_dataset(folder_name: str) -> list[AgentEntry]:
    """Merge all domain files from a dataset folder into a list of agents."""
    folder_path = AGENTS_NORM_DIR / folder_name
    if not folder_path.exists():
        log(f"Folder '{folder_name}' not found", "WARN")
        return []
    
    agents = []
    json_files = sorted(folder_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            domain = data.get("domain", json_file.stem)
            for agent_data in data.get("agents", []):
                agent = AgentEntry(
                    agent_id=agent_data.get("agent_id", ""),
                    display_name=agent_data.get("display_name", ""),
                    persona=agent_data.get("persona", ""),
                    description=agent_data.get("description", ""),
                    role_id=agent_data.get("role_id", ""),
                    domain=domain,
                    tools=agent_data.get("tools", []),
                    dataset=folder_name
                )
                if agent.agent_id:
                    agents.append(agent)
        except Exception as e:
            log(f"Error reading {json_file}: {e}", "ERROR")
    
    log(f"Merged {len(agents)} agents from {folder_name} ({len(json_files)} files)")
    return agents


def save_merged_dataset(folder_name: str, agents: list[AgentEntry]) -> Path:
    """Save merged dataset to a JSON file."""
    dataset_dir = OUTPUT_DIR / folder_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "dataset.json"
    
    data = {
        "dataset": folder_name,
        "merged_at": datetime.now().isoformat(),
        "total_agents": len(agents),
        "agents": [a.to_dict() for a in agents]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    log(f"Saved merged dataset to {output_path}")
    return output_path


def load_merged_dataset(folder_name: str) -> list[AgentEntry]:
    """Load merged dataset from file."""
    dataset_path = OUTPUT_DIR / folder_name / "dataset.json"
    if not dataset_path.exists():
        return []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    agents = []
    for agent_data in data.get("agents", []):
        agent = AgentEntry(
            agent_id=agent_data.get("agent_id", ""),
            display_name=agent_data.get("display_name", ""),
            persona=agent_data.get("persona", ""),
            description=agent_data.get("description", ""),
            role_id=agent_data.get("role_id", ""),
            domain=agent_data.get("domain", ""),
            tools=agent_data.get("tools", []),
            dataset=agent_data.get("dataset", folder_name)
        )
        if agent.agent_id:
            agents.append(agent)
    
    return agents


# ═══════════════════════════════════════════════════════════════════════════════
# LLM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def create_llm(api_key: str, base_url: str, model: str) -> ChatOpenAI:
    """Create LLM client."""
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=TEMPERATURE,
        max_tokens=4000
    )


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove markdown code blocks
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    
    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    return None


async def call_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str, timeout: int) -> dict | None:
    """Call LLM and parse JSON response with retries and exponential backoff."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    req_id = f"{datetime.now().timestamp():.0f}"[-6:]

    for attempt in range(MAX_RETRIES + 1):
        start = datetime.now()
        try:
            log_llm(f"[{req_id}] REQUEST attempt={attempt}/{MAX_RETRIES} prompt_len={len(user_prompt)}")
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
            elapsed = (datetime.now() - start).total_seconds()
            log_llm(f"[{req_id}] RESPONSE elapsed={elapsed:.2f}s len={len(response.content)}")
            log_llm(f"[{req_id}] CONTENT: {response.content[:500]}...")
            
            result = extract_json(response.content)
            if result:
                log_llm(f"[{req_id}] PARSED OK: tasks={len(result.get('tasks', []))}")
                return result
            
            log_llm(f"[{req_id}] JSON parse failed, will retry")
            
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start).total_seconds()
            log_llm(f"[{req_id}] TIMEOUT attempt={attempt} after {elapsed:.0f}s")
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            
            if "524" in str(e) or "timeout" in str(e).lower():
                log_llm(f"[{req_id}] SERVER TIMEOUT (524) attempt={attempt} after {elapsed:.0f}s")
            elif "500" in str(e) or "502" in str(e) or "503" in str(e):
                log_llm(f"[{req_id}] SERVER ERROR attempt={attempt} after {elapsed:.0f}s: {error_msg}")
            else:
                log_llm(f"[{req_id}] ERROR attempt={attempt} elapsed={elapsed:.2f}s: {error_type}: {error_msg}")

        if attempt < MAX_RETRIES:
            wait_time = 2 ** (attempt + 1)
            log_llm(f"[{req_id}] Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)

    log_llm(f"[{req_id}] FAILED after {MAX_RETRIES + 1} attempts")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TASK GENERATION PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

TASK_GENERATION_SYSTEM_PROMPT_EN = """You create realistic user requests for AI agent training data.

Your goal: generate {num_tasks} diverse, realistic user messages that THIS SPECIFIC AGENT would handle better than any other agent.

## CRITICAL: WHY THIS AGENT?

Each task MUST be one where THIS SPECIFIC AGENT excels over all others.

Before writing each task, ask yourself:
- "Would a general assistant handle this equally well?" → If YES, task is BAD
- "Would a different specialist agent be better for this?" → If YES, task is BAD  
- "Does this task specifically need THIS agent's unique expertise?" → Must be YES

Read the agent's full profile — name, persona, description, tools.
The task should leverage THIS agent's specific knowledge, skills, and character.

## TASK TYPES — BE CREATIVE!

Tasks are NOT just questions! They can be:
- **Direct requests**: "Book me a hotel in Paris for next week", "Draw me a logo for my cafe"
- **Emotional support**: "I'm feeling overwhelmed, can we talk?", "I failed my exam and don't know what to do"
- **Action tasks**: "Schedule a meeting", "Find and compare prices", "Translate this document"
- **Creative generation**: "Write a poem about...", "Design a workout plan", "Compose a melody"
- **Complex problems with real data**: Full math problems with numbers, code debugging with actual code, legal cases with specifics
- **Roleplay/simulation**: "Pretend you're interviewing me for a job", "Let's practice a sales pitch"

For META-AGENTS (orchestrator, router, planner, etc.):
- "Optimize this multi-agent communication graph"
- "Which agents should handle this complex request?"
- "Plan execution order for these 5 subtasks"
- "The previous agent failed, decide on fallback strategy"

## DIVERSITY REQUIREMENTS ({num_tasks} tasks)

1-2: Simple beginner questions
3-4: Complex analysis or comparison
5-6: Action/execution requests (do something, not just explain)
7-8: Creative or emotional tasks
9-10: Real-world problems with specific details
11: **SUPER HARD** — a genuinely difficult problem that might stump even the LLM
     Examples: complex physics calculation with numbers, multi-step legal analysis, 
     intricate code optimization, advanced mathematical proof

## REALISTIC USER VOICE

Write as REAL users would — natural, messy, contextual:
- "hey can you help me with..." (casual)
- "I need this urgently for tomorrow..." (stressed)
- "So I've been thinking about this for a while..." (conversational)
- Include typos occasionally, incomplete thoughts, real emotions

## AVOID

❌ Generic: "Help me with music"
❌ Only questions — include ACTION requests
❌ Robotic: "Provide comprehensive analysis"
❌ Same type repeated
❌ Tasks any general assistant could do

## OUTPUT FORMAT
Return ONLY valid JSON with exactly {num_tasks} tasks:
{{
  "tasks": [
    "First user request...",
    "Second user request...",
    ...
  ]
}}

No markdown, no explanations outside JSON."""


TASK_GENERATION_SYSTEM_PROMPT_RU = """Ты создаёшь реалистичные пользовательские запросы для обучающих данных AI-агентов.

Цель: сгенерировать {num_tasks} разнообразных, реалистичных пользовательских сообщений, с которыми ИМЕННО ЭТОТ АГЕНТ справится лучше любого другого.

## КРИТИЧНО: ПОЧЕМУ ИМЕННО ЭТОТ АГЕНТ?

Каждая задача ОБЯЗАНА быть такой, где ИМЕННО ЭТОТ АГЕНТ справится лучше всех остальных.

Перед написанием каждой задачи спроси себя:
- "Справится ли обычный ассистент так же хорошо?" → Если ДА, задача ПЛОХАЯ
- "Справится ли другой специализированный агент лучше?" → Если ДА, задача ПЛОХАЯ
- "Нужна ли для этой задачи уникальная экспертиза ЭТОГО агента?" → Должно быть ДА

Прочитай полный профиль агента — имя, персону, описание, инструменты.
Задача должна использовать именно ЕГО уникальные знания, навыки и характер.

## ТИПЫ ЗАДАЧ — БУДЬ КРЕАТИВНЫМ!

Задачи — это НЕ только вопросы! Они могут быть:
- **Прямые просьбы**: "Забронируй мне отель в Париже на следующую неделю", "Нарисуй логотип для моего кафе"
- **Эмоциональная поддержка**: "Мне плохо, можешь со мной поговорить?", "Завалил экзамен и не знаю что делать"
- **Действия**: "Назначь встречу", "Найди и сравни цены", "Переведи этот документ"
- **Творческая генерация**: "Напиши стих про...", "Составь план тренировок", "Сочини мелодию"
- **Сложные задачи с реальными данными**: Полные математические задачи с числами, дебаг кода с реальным кодом, юридические кейсы с деталями
- **Ролевая игра/симуляция**: "Представь что ты проводишь со мной собеседование", "Давай отрепетируем презентацию"

Для МЕТА-АГЕНТОВ (orchestrator, router, planner и т.д.):
- "Оптимизируй этот граф коммуникации агентов"
- "Какие агенты должны обработать этот сложный запрос?"
- "Спланируй порядок выполнения этих 5 подзадач"
- "Предыдущий агент упал, реши стратегию fallback"

## ТРЕБОВАНИЯ К РАЗНООБРАЗИЮ ({num_tasks} задач)

1-2: Простые вопросы новичка
3-4: Сложный анализ или сравнение
5-6: Запросы на действие (сделай что-то, а не просто объясни)
7-8: Творческие или эмоциональные задачи
9-10: Реальные проблемы с конкретными деталями
11: **СУПЕР-СЛОЖНАЯ** — реально трудная задача, которая может поставить в тупик даже LLM
     Примеры: сложный физический расчёт с числами, многоступенчатый юридический анализ,
     хитрая оптимизация кода, продвинутое математическое доказательство

## РЕАЛИСТИЧНЫЙ ГОЛОС ПОЛЬЗОВАТЕЛЯ

Пиши как РЕАЛЬНЫЕ пользователи — естественно, небрежно, с контекстом:
- "слушай можешь помочь с..." (разговорно)
- "Срочно нужно к завтра..." (стресс)
- "Короче я тут думал давно уже..." (разговорный)
- Иногда опечатки, незаконченные мысли, реальные эмоции

## ИЗБЕГАЙ

❌ Общее: "Помоги с музыкой"
❌ Только вопросы — включай ДЕЙСТВИЯ
❌ Роботизированно: "Предоставь комплексный анализ"
❌ Один тип повторяется
❌ Задачи, с которыми справится любой ассистент

## ФОРМАТ ВЫВОДА
Верни ТОЛЬКО валидный JSON ровно с {num_tasks} задачами:
{{
  "tasks": [
    "Первый запрос пользователя...",
    "Второй запрос пользователя...",
    ...
  ]
}}

Без markdown, без пояснений вне JSON."""


def build_task_prompt(agent: AgentEntry, language: str) -> tuple[str, str]:
    """Build prompts for task generation."""
    
    num_tasks = TASKS_PER_AGENT
    
    # Describe tools capability
    if agent.tools:
        tools_desc_en = f"🔧 Tools available: {', '.join(agent.tools)}"
        tools_desc_ru = f"🔧 Доступные инструменты: {', '.join(agent.tools)}"
    else:
        tools_desc_en = "🔧 No external tools (reasoning and knowledge only)"
        tools_desc_ru = "🔧 Без внешних инструментов (только рассуждения и знания)"
    
    if language == "ru":
        system_prompt = TASK_GENERATION_SYSTEM_PROMPT_RU.format(num_tasks=num_tasks)
        user_prompt = f"""# Агент: {agent.display_name}

{agent.persona}

{agent.description}

{tools_desc_ru}

---

Сгенерируй {num_tasks} задач, где ЭТОТ агент ПРЕВЗОЙДЁТ любого другого.

Для каждой задачи спроси: "Почему я выбрал бы ЭТОГО агента из 1000 других AI-специалистов?"
Если нет чёткого ответа — перепиши задачу, чтобы она требовала уникальную экспертизу этого агента.

Задачи на РУССКОМ языке. Разные по сложности и типу. Естественный язык.
Верни только JSON."""
    else:
        system_prompt = TASK_GENERATION_SYSTEM_PROMPT_EN.format(num_tasks=num_tasks)
        user_prompt = f"""# Agent: {agent.display_name}

{agent.persona}

{agent.description}

{tools_desc_en}

---

Generate {num_tasks} tasks where THIS agent would OUTPERFORM any other agent.

For each task ask: "Why would I choose THIS agent over 1000 other AI specialists?"
If no clear answer — rewrite the task to be more specific to this agent's unique expertise.

Tasks in ENGLISH. Varying complexity and type. Natural language.
Return only JSON."""
    
    return system_prompt, user_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# TASK GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_tasks_for_agent(
    agent: AgentEntry,
    llm: ChatOpenAI,
    language: str,
    timeout: int
) -> AgentTasks | None:
    """Generate tasks for a single agent."""
    
    system_prompt, user_prompt = build_task_prompt(agent, language)
    
    result = await call_llm(llm, system_prompt, user_prompt, timeout)
    
    if not result:
        log(f"Failed to generate tasks for {agent.agent_id}", "WARN")
        return None
    
    tasks = result.get("tasks", [])
    if not tasks:
        log(f"No tasks in response for {agent.agent_id}", "WARN")
        return None
    
    # Validate tasks are strings with reasonable length
    valid_tasks = [t for t in tasks if isinstance(t, str) and len(t) > 20]
    
    if len(valid_tasks) < 5:
        log(f"Too few valid tasks for {agent.agent_id}: {len(valid_tasks)}", "WARN")
        return None
    
    return AgentTasks(
        agent_id=agent.agent_id,
        dataset=agent.dataset,
        tasks=valid_tasks[:TASKS_PER_AGENT]
    )


def get_tasks_file_path(folder_name: str) -> Path:
    """Get path to the unified tasks file for a dataset."""
    dataset_dir = OUTPUT_DIR / folder_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir / "tasks.json"


def load_existing_tasks(folder_name: str) -> dict[str, list[str]]:
    """Load existing tasks from the unified file."""
    tasks_file = get_tasks_file_path(folder_name)
    if not tasks_file.exists():
        return {}
    
    try:
        with open(tasks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["agent_id"]: item["tasks"] for item in data.get("agents", [])}
    except (json.JSONDecodeError, IOError, KeyError):
        return {}


def save_all_tasks(folder_name: str, all_tasks: dict[str, list[str]], agents: list[AgentEntry]):
    """Save all tasks to a single unified JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks_file = get_tasks_file_path(folder_name)
    
    # Build agents list with their tasks
    agents_with_tasks = []
    for agent in agents:
        if agent.agent_id in all_tasks:
            agents_with_tasks.append({
                "agent_id": agent.agent_id,
                "display_name": agent.display_name,
                "domain": agent.domain,
                "role_id": agent.role_id,
                "tasks": all_tasks[agent.agent_id]
            })
    
    data = {
        "dataset": folder_name,
        "generated_at": datetime.now().isoformat(),
        "total_agents": len(agents_with_tasks),
        "total_tasks": sum(len(a["tasks"]) for a in agents_with_tasks),
        "agents": agents_with_tasks
    }
    
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    log(f"Saved {len(agents_with_tasks)} agents with tasks to {tasks_file}")


def get_language_for_dataset(folder_name: str) -> str:
    """Determine language based on dataset folder name."""
    if "rus" in folder_name.lower():
        return "ru"
    return "en"


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

async def process_dataset(
    folder_name: str,
    llm: ChatOpenAI,
    progress: dict,
    parallel: int,
    timeout: int
) -> dict:
    """Process all agents in a dataset: merge and generate tasks."""
    
    language = get_language_for_dataset(folder_name)
    lang_label = "Russian" if language == "ru" else "English"
    
    # Step 1: Merge dataset if not already done
    if folder_name not in progress["merged_datasets"]:
        console.print(f"\n[bold cyan]📦 Merging {folder_name}...[/bold cyan]")
        agents = merge_dataset(folder_name)
        if not agents:
            console.print(f"[yellow]⚠ No agents found in {folder_name}, skipping[/yellow]")
            return progress["stats"]
        
        save_merged_dataset(folder_name, agents)
        progress["merged_datasets"].append(folder_name)
        save_progress(progress)
        console.print(f"[green]✓ Merged {len(agents)} agents from {folder_name}[/green]")
    else:
        console.print(f"\n[bold cyan]📦 Loading merged {folder_name}...[/bold cyan]")
        agents = load_merged_dataset(folder_name)
        console.print(f"[green]✓ Loaded {len(agents)} agents[/green]")
    
    if not agents:
        return progress["stats"]
    
    # Load existing tasks from unified file
    all_tasks = load_existing_tasks(folder_name)
    
    # Get completed agents for this dataset
    completed = set(progress["completed_agents"].get(folder_name, []))
    pending_agents = [a for a in agents if a.agent_id not in completed]
    
    if not pending_agents:
        console.print(f"[green]✓ All {len(agents)} agents already processed[/green]")
        return progress["stats"]
    
    console.print(f"[cyan]🤖 Generating tasks for {len(pending_agents)} agents ({lang_label})...[/cyan]")
    
    stats = progress["stats"]
    semaphore = asyncio.Semaphore(parallel)
    lock = asyncio.Lock()
    processed_count = len(completed)
    total_count = len(agents)
    save_counter = 0  # Counter for periodic saves
    
    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2
    )
    
    async def process_agent(agent: AgentEntry, task_id) -> bool:
        nonlocal processed_count, save_counter
        async with semaphore:
            agent_tasks = await generate_tasks_for_agent(agent, llm, language, timeout)
            
            async with lock:
                if agent_tasks:
                    all_tasks[agent.agent_id] = agent_tasks.tasks
                    stats["tasks_generated"] += len(agent_tasks.tasks)
                    log(f"Generated {len(agent_tasks.tasks)} tasks for {agent.agent_id}")
                else:
                    stats["errors"] += 1
                
                if folder_name not in progress["completed_agents"]:
                    progress["completed_agents"][folder_name] = []
                progress["completed_agents"][folder_name].append(agent.agent_id)
                
                processed_count += 1
                save_counter += 1
                stats["total_agents"] = processed_count
                progress_bar.update(task_id, completed=processed_count)
                
                # Save progress and tasks file every 50 agents
                if save_counter >= 50:
                    save_progress(progress)
                    save_all_tasks(folder_name, all_tasks, agents)
                    save_counter = 0
                else:
                    save_progress(progress)
            
            return agent_tasks is not None
    
    try:
        with progress_bar:
            task_id = progress_bar.add_task(
                f"[cyan]{folder_name} ({lang_label})[/cyan]",
                total=total_count,
                completed=len(completed)
            )
            await asyncio.gather(*[process_agent(a, task_id) for a in pending_agents])
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted! Saving progress...[/yellow]")
        save_all_tasks(folder_name, all_tasks, agents)
        raise
    
    # Final save of all tasks
    save_all_tasks(folder_name, all_tasks, agents)
    
    console.print(f"[bold green]✓ Dataset '{folder_name}' complete: {processed_count}/{total_count} agents[/bold green]")
    console.print(f"[green]   → Saved to {get_tasks_file_path(folder_name)}[/green]")
    log(f"Dataset '{folder_name}' complete: {processed_count}/{total_count} agents")
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_all_tasks_async(
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    parallel: int = DEFAULT_PARALLEL,
    resume: bool = True,
    folders: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = MAX_RETRIES
):
    """Main async function to generate tasks for all agents."""
    global MAX_RETRIES
    MAX_RETRIES = retries
    
    console.print(Panel.fit(
        "[bold cyan]🎯 Task Generator[/bold cyan]\n"
        "Generate training tasks for each agent",
        border_style="cyan"
    ))
    
    # Show config
    table = Table(title="Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input", str(AGENTS_NORM_DIR))
    table.add_row("Output", str(OUTPUT_DIR))
    table.add_row("Model", model)
    table.add_row("Parallel", str(parallel))
    table.add_row("Timeout", f"{timeout}s")
    table.add_row("Retries", str(retries))
    table.add_row("Tasks per agent", str(TASKS_PER_AGENT))
    console.print(table)
    
    # Load or initialize progress
    if resume:
        progress = load_progress()
        total_completed = sum(len(v) for v in progress["completed_agents"].values())
        if total_completed > 0:
            console.print(f"\n[green]✓ Resuming: {total_completed} agents already processed[/green]")
    else:
        clear_progress()
        progress = load_progress()
    
    # Create LLM client
    llm = create_llm(api_key, base_url, model)
    
    # Process datasets
    folders_to_process = folders if folders else DATASET_FOLDERS
    
    log("=" * 80)
    log(f"Starting task generation: folders={folders_to_process}, parallel={parallel}, model={model}")
    log("=" * 80)
    
    try:
        for folder_name in folders_to_process:
            await process_dataset(folder_name, llm, progress, parallel, timeout)
    except KeyboardInterrupt:
        console.print("\n[yellow]Saving progress and exiting...[/yellow]")
        save_progress(progress)
        console.print("[green]Progress saved. Run again to resume.[/green]")
        return
    
    # Final stats
    stats = progress["stats"]
    
    console.print()
    stats_table = Table(title="Task Generation Complete", show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Agents processed", str(stats["total_agents"]))
    stats_table.add_row("Tasks generated", f"[green]{stats['tasks_generated']}[/green]")
    stats_table.add_row("Errors", f"[red]{stats['errors']}[/red]" if stats["errors"] > 0 else "0")
    stats_table.add_row("Output", str(OUTPUT_DIR))
    console.print(stats_table)
    
    # Check if all complete
    all_complete = True
    for folder in folders_to_process:
        agents = load_merged_dataset(folder)
        completed = set(progress["completed_agents"].get(folder, []))
        if len(completed) < len(agents):
            all_complete = False
            break
    
    if all_complete:
        clear_progress()
        console.print("\n[bold green]✓ All datasets processed successfully![/bold green]")
    
    log("=" * 80)
    log(f"Task generation complete: agents={stats['total_agents']} tasks={stats['tasks_generated']} errors={stats['errors']}")
    log("=" * 80)


def generate_all_tasks(
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    parallel: int = DEFAULT_PARALLEL,
    resume: bool = True,
    folders: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = MAX_RETRIES
):
    """Synchronous wrapper."""
    asyncio.run(generate_all_tasks_async(api_key, base_url, model, parallel, resume, folders, timeout, retries))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate training tasks for each agent in the dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (resumes if interrupted)
  python task_generator.py
  
  # Run with 20 parallel requests
  python task_generator.py -p 20
  
  # Start fresh (ignore previous progress)
  python task_generator.py --fresh
  
  # Process specific folders only
  python task_generator.py --folders agents_eng agents_rus
  
  # Custom API settings
  python task_generator.py --base-url http://localhost:8080/v1 --model my-model
  
  # Adjust timeout and retries
  python task_generator.py --timeout 120 --retries 5
        """
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="LLM API key"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="LLM API base URL"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LLM model identifier"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Number of parallel agent requests (default: {DEFAULT_PARALLEL})"
    )
    
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any previous progress"
    )
    
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help=f"Specific dataset folders to process (default: {DATASET_FOLDERS})"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per LLM request in seconds (default: {DEFAULT_TIMEOUT})"
    )
    
    parser.add_argument(
        "--retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Number of retries on failure (default: {MAX_RETRIES})"
    )
    
    args = parser.parse_args()
    
    try:
        generate_all_tasks(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            parallel=args.parallel,
            resume=not args.fresh,
            folders=args.folders,
            timeout=args.timeout,
            retries=args.retries
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        log(f"Fatal error: {e}\n{tb.format_exc()}", "FATAL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
