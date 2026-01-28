#!/usr/bin/env python3
"""
LLM Agent Factory v2 - Simplified
Generates AgentSpec definitions by iterating through domains with parallel processing.
For each domain: process each role separately, then refine the whole domain.
"""

import json
import re
import asyncio
import argparse
import traceback as tb
from pathlib import Path
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

# Configuration
DEFAULT_API_KEY = "very-secure-key-sber1love"
DEFAULT_BASE_URL = "https://keyword-cameras-homework-analyze.trycloudflare.com/v1"
DEFAULT_MODEL = "gpt-oss"

MAX_AGENTS_PER_COMBO = 3
MAX_RETRIES = 3
TEMPERATURE = 0.3
REQUEST_DELAY = 0.5
DEFAULT_TIMEOUT = 1800  # seconds per LLM request

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
CONFIG_DIR = BASE_DIR / "config"
DATASET_DIR = BASE_DIR / "dataset"
AGENTS_DIR = DATASET_DIR / "agents"
PROGRESS_FILE = SCRIPT_DIR / ".generation_progress.json"
LLM_LOG_FILE = SCRIPT_DIR / "llm_debug.log"

console = Console()

def log_llm(msg: str):
    """Log LLM debug info to file."""
    with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def load_catalogs():
    """Load domain, role, and tool catalogs from JSON files."""
    with open(CONFIG_DIR / "domain.json", "r", encoding="utf-8") as f:
        domains = json.load(f).get("domain", [])
    with open(CONFIG_DIR / "role_id.json", "r", encoding="utf-8") as f:
        roles = json.load(f).get("roles", [])
    with open(CONFIG_DIR / "tool.json", "r", encoding="utf-8") as f:
        tools = json.load(f).get("tools", [])
    return domains, roles, tools


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_combo_prompt(domain: str, role: str, all_tools: list[str], max_agents: int = MAX_AGENTS_PER_COMBO) -> tuple[str, str]:
    """Build prompts for generating agents for ONE domain + ONE role combination."""

    system_prompt = f"""You evaluate if a domain+role combination is logical and create agents ONLY if it makes real practical sense.

## OUTPUT FORMAT
{{
  "agents": [...],  // array of agents OR empty array if combination is invalid
  "notes": "string" // explanation of your decision
}}

## Agent Schema (when creating)
{{
  "agent_id": "string",      // [a-z0-9_]+ slug
  "display_name": "string",  // human name
  "persona": "string",       // 2-3 sentences
  "description": "string",   // what agent does
  "role_id": "{role}",       // FIXED
  "domain": "{domain}",      // FIXED
  "tools": [],               // usually empty - see rules
  "input_schema": {{}},      // usually empty - see rules
  "output_schema": {{}},     // usually empty - see rules
  "raw": {{}}
}}

## RULE 1: SOME COMBINATIONS ARE INVALID → 0 AGENTS

Not all domain+role combinations make sense. Return empty array when combination is illogical.

INVALID examples:
- coder + Cooking → no code to write
- debugger + Philosophy → nothing to debug
- devops + Music → no infrastructure
- architect + Sports → no systems to design
- data_scientist + Folklore → no data to analyze

Technical roles (coder, debugger, architect, devops, code_reviewer) are only valid for technical domains like Computing, Programming, Software, Engineering.

ASK: "Would someone hire a {role} specifically for {domain}?"
If NO → return {{"agents": [], "notes": "Invalid: <reason>"}}

## RULE 2: tools — use when agent needs external capabilities

Available tools:
- "web_search" — for agents that need current information from internet
- "code_interpreter" — for agents that execute code or calculations
- "file_search" — for agents that search through documents
- "vector_search" — for semantic search in knowledge bases
- "image_generation" — for agents that create images
- "shell" — for agents that run system commands
- "computer_use" — for agents that interact with computer UI
- "apply_patch" — for agents that modify code files
- "function_calling" — for agents that call external APIs
- "remote_mcp_servers" — for connecting to external tool servers

Most agents work with tools: [] because they process provided information.
Add tools only when the agent's job requires external actions or data:

EXAMPLES when to use tools:
- Research assistant needing latest news → "web_search"
- Code debugger that runs code → "code_interpreter"
- File analyzer searching documents → "file_search"
- API integrator calling services → "function_calling"

EXAMPLES when NOT to use tools:
- Writing assistant explaining concepts → tools: []
- Advisor giving recommendations → tools: []
- Teacher explaining topics → tools: []

## RULE 3: input_schema = {{}} and output_schema = {{}} unless agent returns pure data

Schemas are only for agents that return structured data with NO explanatory text.

NEVER add schemas for agents that explain, advise, write, analyze, teach — their output is prose.

ONLY add output_schema for agents returning pure data:
- Calculator returning only numbers
- Classifier returning only category
- Extractor returning only JSON fields

If agent writes any explanatory text → schemas: {{}}

## RULE 4: Maximum {max_agents} agents, prefer 0 or 1 or more

Only create multiple agents if they serve completely different purposes.

Return ONLY valid JSON."""

    user_prompt = f"""Domain: "{domain}"
Role: "{role}"

Is this combination logical? Would someone hire a "{role}" for "{domain}"?

If NO → return {{"agents": [], "notes": "Invalid because..."}}
If YES → create 1 agent. Use tools: [] unless the agent needs external capabilities for its job."""

    return system_prompt, user_prompt




def create_llm_client(api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create LLM client."""
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=TEMPERATURE)

async def call_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> dict:
    """Call LLM and parse JSON response with retries."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    req_id = f"{datetime.now().timestamp():.0f}"[-6:]  # short request id

    for attempt in range(MAX_RETRIES + 1):
        start = datetime.now()
        try:
            log_llm(f"[{req_id}] REQUEST attempt={attempt} prompt={user_prompt[:80]!r}")
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=DEFAULT_TIMEOUT)
            elapsed = (datetime.now() - start).total_seconds()
            log_llm(f"[{req_id}] RESPONSE elapsed={elapsed:.2f}s len={len(response.content)}")
            log_llm(f"[{req_id}] CONTENT: {response.content}")
            json_text = extract_json_from_response(response.content)
            data = json.loads(json_text)
            if isinstance(data, dict):
                return data
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            log_llm(f"[{req_id}] ERROR attempt={attempt} elapsed={elapsed:.2f}s type={type(e).__name__} error={e}")
            log_llm(f"[{req_id}] TRACEBACK: {tb.format_exc()}")

        if attempt < MAX_RETRIES:
            await asyncio.sleep(1)

    return {"agents": []}  # Return valid empty structure after all retries

def extract_json_from_response(response: str) -> str:
    """Extract JSON from response."""
    text = response.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text


def validate_agent(agent: dict, domain: str, role: str, tools: list[str]) -> dict | None:
    """Validate and normalize a single agent."""
    if not isinstance(agent, dict) or "agent_id" not in agent:
        return None

    agent_id = agent["agent_id"]
    if not re.match(r"^[a-z0-9_]+$", agent_id):
        return None

    # Normalize agent
    agent["domain"] = domain
    agent["role_id"] = role
    agent["tools"] = [t for t in agent.get("tools", []) if t in tools]
    agent.setdefault("input_schema", {})
    agent.setdefault("output_schema", {})
    agent.setdefault("raw", {})

    return agent

def load_progress() -> set:
    """Load progress data."""
    if PROGRESS_FILE.exists():
        data = json.load(open(PROGRESS_FILE, encoding="utf-8"))
        return set(data.get("completed_domains", []))
    return set()

def save_progress(completed_domains: set):
    """Save progress."""
    json.dump({
        "completed_domains": list(completed_domains)
    }, open(PROGRESS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def clear_progress():
    """Clear progress file."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def get_domain_file_path(domain: str) -> Path:
    """Get the file path for a domain's agents."""
    filename = re.sub(r'[^\w\-]', '_', domain) + ".json"
    return AGENTS_DIR / filename


def save_domain_agents(domain: str, agents: list[dict]) -> Path:
    """Save agents for a domain to its own JSON file."""
    AGENTS_DIR.mkdir(exist_ok=True)
    output_path = get_domain_file_path(domain)
    
    result = {
        "domain": domain,
        "generated_at": datetime.now().isoformat(),
        "total_agents": len(agents),
        "agents": agents
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# COMBO PROCESSING (domain + role)
# ═══════════════════════════════════════════════════════════════════════════════

async def process_combo(domain: str, role: str, tools: list[str], llm: ChatOpenAI) -> list[dict]:
    """Process domain+role combination."""
    system_prompt, user_prompt = build_combo_prompt(domain, role, tools)

    for attempt in range(MAX_RETRIES + 1):
        try:
            data = await call_llm(llm, system_prompt, user_prompt)

            if not isinstance(data.get("agents"), list):
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(0.1)  # Faster retry
                    continue
                return []

            valid_agents = []
            for agent in data["agents"]:
                if validated := validate_agent(agent, domain, role, tools):
                    valid_agents.append(validated)

            return valid_agents[:MAX_AGENTS_PER_COMBO]

        except Exception as e:
            if attempt < MAX_RETRIES:
                console.print(f"[yellow]Retry {attempt+1}/{MAX_RETRIES} {domain}+{role}: {e}[/yellow]")
                await asyncio.sleep(0.5)
            else:
                console.print(f"[red]Failed {domain}+{role} after {MAX_RETRIES+1} attempts: {e}[/red]")
                return []

    return []


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

async def process_domain(domain: str, roles: list[str], tools: list[str], llm: ChatOpenAI) -> list[dict]:
    """Process all roles for a domain and save to file."""
    domain_agents = []

    # Process each role
    for role in roles:
        console.print(f"  [{domain}] Processing {role}...")
        agents = await process_combo(domain, role, tools, llm)
        domain_agents.extend(agents)
        await asyncio.sleep(REQUEST_DELAY)

    # Save domain agents to file
    output_path = save_domain_agents(domain, domain_agents)
    console.print(f"[bold green]  [{domain}] Complete: {len(domain_agents)} agents → {output_path.name}[/bold green]")
    return domain_agents


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_all_agents_async(api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, resume: bool = True, parallel: int = 1) -> int:
    """Main async function to generate agents for all domains."""
    console.print(Panel.fit("[bold cyan]LLM Agent Factory v2[/bold cyan]", border_style="cyan"))

    # Load catalogs
    domains, roles, tools = load_catalogs()
    total_combos = len(domains) * len(roles)

    # Show summary
    table = Table(title="Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Domains", str(len(domains)))
    table.add_row("Roles", str(len(roles)))
    table.add_row("Total Combinations", f"[bold]{total_combos}[/bold]")
    table.add_row("Tools", str(len(tools)))
    table.add_row("Output Dir", str(AGENTS_DIR))
    table.add_row("Model", model)
    table.add_row("Parallel", str(parallel))
    table.add_row("Timeout", f"{DEFAULT_TIMEOUT}s")
    console.print(table)
    console.print()

    # Load progress
    if resume:
        completed_domains = load_progress()
        if completed_domains:
            console.print(f"[green]Resuming: {len(completed_domains)}/{len(domains)} domains completed[/green]\n")
    else:
        clear_progress()
        completed_domains = set()

    llm = create_llm_client(api_key, base_url, model)
    
    # Domain is pending if not completed OR file is missing
    def is_domain_pending(domain: str) -> bool:
        if domain not in completed_domains:
            return True
        return not get_domain_file_path(domain).exists()
    
    pending_domains = [d for d in domains if is_domain_pending(d)]

    if not pending_domains:
        console.print("[green]All domains already processed![/green]")
        return len(completed_domains)

    total_agents = 0
    semaphore = asyncio.Semaphore(parallel)
    lock = asyncio.Lock()

    async def process_domain_with_semaphore(domain: str, progress_task) -> int:
        nonlocal total_agents, completed_domains
        async with semaphore:
            try:
                # Retry domain if 0 agents (likely a bug)
                for domain_attempt in range(MAX_RETRIES + 1):
                    agents = await process_domain(domain, roles, tools, llm)
                    if len(agents) > 0:
                        break
                    if domain_attempt < MAX_RETRIES:
                        console.print(f"[yellow]  [{domain}] 0 agents generated, retrying ({domain_attempt + 1}/{MAX_RETRIES})...[/yellow]")
                        await asyncio.sleep(1)
                    else:
                        console.print(f"[red]  [{domain}] 0 agents after {MAX_RETRIES + 1} attempts, skipping[/red]")
                
                async with lock:
                    total_agents += len(agents)
                    completed_domains.add(domain)
                    progress.update(progress_task, advance=1, description=f"[green]Total: {total_agents}[/green]")
                    save_progress(completed_domains)
                return len(agents)
            except Exception as e:
                console.print(f"[red]  [{domain}] Error: {e}[/red]")
                return 0

    # Process with progress bar
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=40), TaskProgressColumn(), TextColumn("•"), TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(), console=console, refresh_per_second=2) as progress:
        task = progress.add_task("[cyan]Processing domains...", total=len(domains), completed=len(completed_domains))

        try:
            await asyncio.gather(*[process_domain_with_semaphore(domain, task) for domain in pending_domains])
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted! Saving progress...[/yellow]")
            save_progress(completed_domains)
            console.print("[green]Progress saved. Run again to resume.[/green]")
            raise

    # Final stats
    stats_table = Table(title="Generation Complete", show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Domains Processed", str(len(completed_domains)))
    stats_table.add_row("Total Agents", f"[bold]{total_agents}[/bold]")
    stats_table.add_row("Output Directory", str(AGENTS_DIR))
    console.print(stats_table)

    if len(completed_domains) == len(domains):
        clear_progress()
        console.print("\n[green]All domains processed successfully![/green]")

    return total_agents


def generate_all_agents(api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, resume: bool = True, parallel: int = 1) -> int:
    """Synchronous wrapper."""
    return asyncio.run(generate_all_agents_async(api_key, base_url, model, resume, parallel))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate LLM Agents for all domain+role combinations.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LLM API key")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model identifier")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignoring any previous progress")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Number of domains to process in parallel (default: 1)")

    args = parser.parse_args()

    try:
        generate_all_agents(args.api_key, args.base_url, args.model, not args.fresh, args.parallel)
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
