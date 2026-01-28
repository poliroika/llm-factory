#!/usr/bin/env python3
"""
Dataset Curator - LLM-powered agent normalization and curation

Processes agent datasets from multiple folders, analyzes domain completeness,
creates missing agents, removes/merges duplicates, outputs normalized dataset.
"""

import json
import re
import asyncio
import argparse
import traceback as tb
from pathlib import Path
from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_API_KEY = "very-secure-key-sber1love"
DEFAULT_BASE_URL = "https://keyword-cameras-homework-analyze.trycloudflare.com/v1"
DEFAULT_MODEL = "gpt-oss"

TEMPERATURE = 0.3
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 1800
DEFAULT_PARALLEL = 10

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
CONFIG_DIR = BASE_DIR / "config"
DATASET_DIR = BASE_DIR / "dataset"
AGENTS_DIR = DATASET_DIR / "agents"
OUTPUT_DIR = DATASET_DIR / "agents_norm"
PROGRESS_FILE = SCRIPT_DIR / ".curator_progress.json"
LOG_FILE = SCRIPT_DIR / "curator.log"

# Dataset folders to process (in order)
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
# CATALOGS
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalogs() -> tuple[list[str], list[str], list[str]]:
    """Load domain, role, and tool catalogs."""
    with open(CONFIG_DIR / "domain.json", "r", encoding="utf-8") as f:
        domains = json.load(f).get("domain", [])
    with open(CONFIG_DIR / "role_id.json", "r", encoding="utf-8") as f:
        roles = json.load(f).get("roles", [])
    with open(CONFIG_DIR / "tool.json", "r", encoding="utf-8") as f:
        tools = json.load(f).get("tools", [])
    return domains, roles, tools


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
        "completed": {},  # folder -> list of completed domain files
        "stats": {
            "processed_domains": 0,
            "agents_added": 0,
            "agents_removed": 0,
            "agents_merged": 0,
            "agents_edited": 0,
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
# LLM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def create_llm(api_key: str, base_url: str, model: str) -> ChatOpenAI:
    """Create LLM client."""
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=TEMPERATURE,
        max_tokens=8000
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


async def call_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> dict | None:
    """Call LLM and parse JSON response with retries and exponential backoff."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    req_id = f"{datetime.now().timestamp():.0f}"[-6:]

    for attempt in range(MAX_RETRIES + 1):
        start = datetime.now()
        try:
            log_llm(f"[{req_id}] REQUEST attempt={attempt}/{MAX_RETRIES} prompt_len={len(user_prompt)}")
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=DEFAULT_TIMEOUT)
            elapsed = (datetime.now() - start).total_seconds()
            log_llm(f"[{req_id}] RESPONSE elapsed={elapsed:.2f}s len={len(response.content)}")
            log_llm(f"[{req_id}] CONTENT: {response.content[:500]}...")
            
            result = extract_json(response.content)
            if result:
                log_llm(f"[{req_id}] PARSED OK: missing={len(result.get('missing_agents', []))} dups={len(result.get('duplicate_groups', []))} edits={len(result.get('agents_to_edit', []))}")
                return result
            
            log_llm(f"[{req_id}] JSON parse failed, will retry")
            
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start).total_seconds()
            log_llm(f"[{req_id}] TIMEOUT attempt={attempt} after {elapsed:.0f}s")
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            error_type = type(e).__name__
            error_msg = str(e)[:200]  # Truncate long error messages
            
            # Check for common server errors
            if "524" in str(e) or "timeout" in str(e).lower():
                log_llm(f"[{req_id}] SERVER TIMEOUT (524) attempt={attempt} after {elapsed:.0f}s - server overloaded")
            elif "500" in str(e) or "502" in str(e) or "503" in str(e):
                log_llm(f"[{req_id}] SERVER ERROR attempt={attempt} after {elapsed:.0f}s: {error_msg}")
            else:
                log_llm(f"[{req_id}] ERROR attempt={attempt} elapsed={elapsed:.2f}s: {error_type}: {error_msg}")

        if attempt < MAX_RETRIES:
            # Exponential backoff: 2s, 4s, 8s, 16s...
            wait_time = 2 ** (attempt + 1)
            log_llm(f"[{req_id}] Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)

    log_llm(f"[{req_id}] FAILED after {MAX_RETRIES + 1} attempts")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_ANALYSIS_SYSTEM_PROMPT = """You are an expert curator improving agent collections. Your job is to make each domain BETTER.

## YOUR ACTIONS

You MUST actively improve the domain by:

### 1. ADDING missing agents
ADD a new agent when:
- Domain lacks a specialist for an important sub-area (e.g., Music domain missing "music_composer" or "music_theory_expert")
- No agent covers a common use case in this domain
- Domain has only generic agents but lacks domain-specific experts

Example: For "Cooking" domain, if there's only "cooking_general" but no "pastry_chef", "nutritionist_chef", "recipe_developer" — ADD them!

### 2. REMOVING duplicates/similar agents — BE AGGRESSIVE!
REMOVE an agent when:
- Two agents have similar names (e.g., "music_advisor" vs "music_consultant")
- Two agents have overlapping descriptions (doing similar things)
- Agent's role doesn't make sense for this domain
- Agent is too generic when a more specific one exists
- Multiple agents with same role_id exist (keep only the best one!)

LOOK FOR THESE PATTERNS TO REMOVE:
- "{domain}_advisor" vs "{domain}_consultant" vs "{domain}_expert_advisor" — keep ONE
- "{domain}_helper" vs "{domain}_assistant" — keep ONE
- "{domain}_general" duplicating "{domain}_triage" — often redundant
- Generic agents when domain-specific ones exist

Example: If domain has "music_advisor", "music_consultant", "music_expert" — REMOVE TWO, keep the best one!

### 3. EDITING agents to improve quality — ALWAYS LOOK FOR ISSUES!
EDIT when (check EVERY agent for these):
- "persona" is weak/generic/short — MUST be 2-3 vivid sentences describing personality
- "persona" just repeats the role — make it unique and interesting
- "description" is vague like "helps with X" — make it specific: what exactly does it do?
- "tools" are WRONG (see tool rules below)

COMMON ISSUES TO FIX:
- Persona like "An expert in X" → too generic, add personality traits!
- Description like "Helps users with domain tasks" → too vague, be specific!
- researcher/retriever with tools: [] → WRONG, add ["web_search"]
- advisor/tutor with tools: ["web_search"] → usually WRONG, remove tools

### 4. FIXING tools — CHECK EVERY AGENT!
Go through each agent and verify tools are correct:

MUST HAVE tools (if missing → EDIT to add):
- role_id="researcher" → ["web_search"]
- role_id="fact_checker" → ["web_search"]  
- role_id="retriever" → ["web_search"] or ["file_search"]
- role_id="librarian" → ["web_search", "vector_search"] or ["file_search"]
- role_id="coder" → ["code_interpreter"]
- role_id="debugger" → ["code_interpreter"]
- role_id="executor" → ["code_interpreter", "shell"]
- role_id="tool_runner" → multiple tools

SHOULD NOT have tools (if present → EDIT to remove):
- role_id="tutor" → tools: []
- role_id="advisor" → tools: []
- role_id="expert_advisor" → tools: []
- role_id="critic" → tools: []
- role_id="planner" → tools: []
- role_id="summarizer" → tools: []
- role_id="editor" → tools: []
- role_id="translator" → tools: []

### 5. Domain-specific uniqueness
Each domain should have UNIQUE agents specific to that domain, not just generic roles!

BAD: Music domain with only "music_general", "music_tutor", "music_advisor" (too generic)
GOOD: Music domain with "composer_assistant", "music_theory_expert", "instrument_teacher", "audio_mixing_advisor", "songwriting_coach"

## WHAT NOT TO DO
- Don't change agent_id, role_id, or domain fields
- Don't rewrite everything — improve what exists
- Don't add agents with roles invalid for the domain (no "coder" in "Cooking")

## AVAILABLE ROLES AND TOOLS

Roles are from role_id.json — you will see the list in the prompt.
You may create a custom role_id if domain truly needs something special (use lowercase_with_underscores).

Tools are from tool.json:
- "web_search" — current info from internet
- "code_interpreter" — execute code
- "file_search" — search documents
- "vector_search" — semantic search
- "image_generation" — create images
- "shell" — system commands
- "computer_use" — UI interaction
- "apply_patch" — modify code files
- "function_calling" — external APIs
- "remote_mcp_servers" — external tool servers

## AGENT SCHEMA (for new agents only)
{
  "agent_id": "string",      // [a-z0-9_]+ slug, unique within domain
  "display_name": "string",  // human-readable name
  "persona": "string",       // 2-3 sentences describing personality
  "description": "string",   // what agent does
  "role_id": "string",       // from available roles or custom
  "domain": "string",        // FIXED to current domain
  "tools": [],               // usually empty - see tool rules
  "input_schema": {},        // usually empty
  "output_schema": {},       // usually empty
  "raw": {}
}

## ROLE VALIDITY RULES

Technical roles (coder, debugger, architect, devops, code_reviewer, data_scientist) are ONLY valid for technical domains (Computing, Programming, Software, Engineering).

For non-technical domains (Music, Cooking, Sports, Art, etc.) — REMOVE these technical roles if present!

## OUTPUT FORMAT

Return JSON:
{
  "domain": "string",
  "is_complete": boolean,
  "missing_agents": [...],    // Array of NEW agents to add (full agent objects, only if truly missing)
  "duplicate_groups": [...],  // Array: {"keep": "agent_id", "remove": ["agent_id", ...], "reason": "..."}
  "agents_to_edit": [...],    // Array: {"agent_id": "...", "changes": {"field": "new_value"}, "reason": "..."}
  "notes": "string"
}

For agents_to_edit, include changed fields in "changes". REAL EXAMPLES:

FIXING MISSING TOOLS (very common!):
- {"agent_id": "music_researcher", "changes": {"tools": ["web_search"]}, "reason": "Researcher must have web_search"}
- {"agent_id": "cooking_fact_checker", "changes": {"tools": ["web_search"]}, "reason": "Fact checker needs web_search"}
- {"agent_id": "computing_coder", "changes": {"tools": ["code_interpreter"]}, "reason": "Coder must have code_interpreter"}
- {"agent_id": "art_librarian", "changes": {"tools": ["web_search", "file_search"]}, "reason": "Librarian needs search tools"}

REMOVING WRONG TOOLS:
- {"agent_id": "philosophy_tutor", "changes": {"tools": []}, "reason": "Tutor doesn't need tools, teaches through conversation"}
- {"agent_id": "music_advisor", "changes": {"tools": []}, "reason": "Advisor gives advice, doesn't need external tools"}

ADDING SCHEMAS (for structured output agents):
- {"agent_id": "data_classifier", "changes": {"output_schema": {"type": "object", "properties": {"category": {"type": "string"}, "confidence": {"type": "number"}}}}, "reason": "Classifier returns structured categories"}
- {"agent_id": "sentiment_analyzer", "changes": {"output_schema": {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}, "score": {"type": "number"}}}}, "reason": "Returns structured sentiment data"}

IMPROVING WEAK DESCRIPTIONS:
- {"agent_id": "music_general", "changes": {"persona": "A passionate music enthusiast with deep knowledge of various genres, instruments, and music history. Patient and encouraging when explaining concepts.", "description": "Answers questions about music theory, recommends songs and artists, explains musical concepts, and helps users explore different genres and styles."}, "reason": "Expanded weak persona and vague description"}

## IMPORTANT: ALWAYS FIND IMPROVEMENTS!

A perfect domain is RARE. If you return empty arrays for everything, you're probably not looking hard enough!

CHECKLIST before returning:
1. Did you check for duplicate/similar agents? (advisor vs consultant vs expert)
2. Did you verify ALL tools are correct for each role?
3. Did you check personas are 2-3 sentences, not just "An expert in X"?
4. Did you check descriptions are specific, not vague?
5. Did you add domain-specific specialists (not just generic roles)?

If duplicate_groups: [] — double-check for similar agent names!
If agents_to_edit: [] — double-check tool assignments and weak personas!
If missing_agents: [] — think about what domain-specific experts are missing!

## CRITICAL: OUTPUT FORMAT

Return ONLY valid JSON. No explanations, no markdown, no text before or after.
Your entire response must be a single valid JSON object that can be parsed directly.

Example of CORRECT output:
{"domain": "Music", "is_complete": true, "missing_agents": [], "duplicate_groups": [], "agents_to_edit": [], "notes": "Domain is balanced"}

Example of WRONG output:
Here is my analysis:
```json
{"domain": "Music", ...}
```
This domain looks good because...

WRONG! Return ONLY the JSON object, nothing else."""


def build_domain_analysis_prompt(domain: str, agents: list[dict], roles: list[str]) -> str:
    """Build user prompt for domain analysis."""
    agent_summaries = []
    for agent in agents:
        summary = {
            "agent_id": agent.get("agent_id"),
            "display_name": agent.get("display_name"),
            "role_id": agent.get("role_id"),
            "persona": agent.get("persona", "")[:150],
            "description": agent.get("description", "")[:200],
            "tools": agent.get("tools", [])
        }
        agent_summaries.append(summary)
    
    return f"""Domain: "{domain}"

## Available roles (from role_id.json):
{json.dumps(roles)}

Note: You may use these roles or create a custom role_id if the domain truly requires something special.

## Current agents in this domain ({len(agents)} total):

{json.dumps(agent_summaries, indent=2, ensure_ascii=False)}

## Your task — FIND AND FIX ISSUES:

### Step 1: SCAN FOR DUPLICATES
Look at agent names above. Are there similar ones like:
- domain_advisor + domain_consultant + domain_expert → REMOVE extras!
- domain_helper + domain_assistant → REMOVE one!
- Multiple agents with same role_id → KEEP only the best!

### Step 2: CHECK EVERY AGENT'S TOOLS (most common fix!)
Go through the agent list above and check EACH agent's tools field:

MISSING TOOLS? Add them via agents_to_edit:
- researcher with tools: [] → needs ["web_search"]
- fact_checker with tools: [] → needs ["web_search"]
- retriever with tools: [] → needs ["web_search"]
- librarian with tools: [] → needs ["web_search", "file_search"]
- coder with tools: [] → needs ["code_interpreter"]
- debugger with tools: [] → needs ["code_interpreter"]
- executor with tools: [] → needs ["code_interpreter", "shell"]

UNNECESSARY TOOLS? Remove them via agents_to_edit:
- tutor with tools → should be []
- advisor with tools → should be []

This is the MOST COMMON fix needed — check every agent!

### Step 3: CHECK PERSONAS AND DESCRIPTIONS
- Persona just says "An expert in X" → EDIT to add personality traits, style!
- Description is vague like "helps with tasks" → EDIT to be specific about what it does!

### Step 4: CHECK IF SCHEMAS NEEDED
Some agents return structured data and need output_schema:
- "classifier" agents → need output_schema with categories
- "analyzer" agents returning scores → need output_schema
- "extractor" agents pulling specific fields → need output_schema
- "evaluator" agents giving scores → may need output_schema

Most agents (advisors, tutors, writers) output prose → schemas should be empty {{}}

### Step 5: ADD MISSING SPECIALISTS
What unique experts work in "{domain}" that aren't here?

**IMPORTANT:** Return at least SOME improvements! Empty arrays = you didn't look hard enough.

Return ONLY valid JSON. No explanations, no markdown, no other text."""


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_agent(agent: dict, domain: str, tools: list[str]) -> dict | None:
    """Validate and normalize a single agent."""
    if not isinstance(agent, dict) or "agent_id" not in agent:
        return None

    agent_id = agent.get("agent_id", "")
    if not re.match(r"^[a-z0-9_]+$", agent_id):
        return None

    # Normalize
    agent["domain"] = domain
    agent["tools"] = [t for t in agent.get("tools", []) if t in tools]
    agent.setdefault("input_schema", {})
    agent.setdefault("output_schema", {})
    agent.setdefault("raw", {})
    agent.setdefault("display_name", agent_id.replace("_", " ").title())
    agent.setdefault("persona", "")
    agent.setdefault("description", "")

    return agent


def calculate_similarity(agent1: dict, agent2: dict) -> float:
    """Calculate simple similarity between two agents based on text overlap."""
    def get_text(a: dict) -> str:
        return f"{a.get('display_name', '')} {a.get('persona', '')} {a.get('description', '')}".lower()
    
    text1 = set(get_text(agent1).split())
    text2 = set(get_text(agent2).split())
    
    if not text1 or not text2:
        return 0.0
    
    intersection = len(text1 & text2)
    union = len(text1 | text2)
    
    return intersection / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

async def process_domain_file(
    input_path: Path,
    output_path: Path,
    llm: ChatOpenAI,
    roles: list[str],
    tools: list[str],
    stats: dict
) -> bool:
    """Process a single domain file: analyze, add missing, remove duplicates."""
    
    domain_name = input_path.stem
    log(f"Processing domain: {domain_name}")
    
    try:
        # Load domain data
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        domain = data.get("domain", domain_name)
        agents = data.get("agents", [])
        original_count = len(agents)
        
        log(f"Domain '{domain}': {original_count} agents loaded")
        
        # Call LLM for analysis
        system_prompt = DOMAIN_ANALYSIS_SYSTEM_PROMPT
        user_prompt = build_domain_analysis_prompt(domain, agents, roles)
        
        analysis = await call_llm(llm, system_prompt, user_prompt)
        
        if not analysis:
            log(f"Domain '{domain}': LLM analysis failed, copying as-is", "WARN")
            # Copy file as-is if analysis fails
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            stats["errors"] += 1
            return True
        
        # Log what LLM returned
        missing_count = len(analysis.get("missing_agents", []))
        dup_count = len(analysis.get("duplicate_groups", []))
        edit_count = len(analysis.get("agents_to_edit", []))
        log(f"Domain '{domain}': LLM returned - missing={missing_count}, duplicates={dup_count}, edits={edit_count}, notes={analysis.get('notes', '')[:100]}")
        
        # Create agent lookup by id
        agents_by_id = {a.get("agent_id"): a for a in agents}
        
        # Process duplicate groups - remove duplicates
        duplicate_groups = analysis.get("duplicate_groups", [])
        removed_ids = set()
        
        for group in duplicate_groups:
            keep_id = group.get("keep")
            remove_ids = group.get("remove", [])
            reason = group.get("reason", "")
            
            for rid in remove_ids:
                if rid in agents_by_id and rid != keep_id:
                    removed_ids.add(rid)
                    log(f"Domain '{domain}': Removing duplicate '{rid}' (keeping '{keep_id}'): {reason}")
                    stats["agents_removed"] += 1
        
        # Filter out removed agents
        agents = [a for a in agents if a.get("agent_id") not in removed_ids]
        agents_by_id = {a.get("agent_id"): a for a in agents}
        
        # Process edits
        editable_fields = ["display_name", "persona", "description", "tools", "input_schema", "output_schema"]
        for edit in analysis.get("agents_to_edit", []):
            agent_id = edit.get("agent_id")
            changes = edit.get("changes", {})
            reason = edit.get("reason", "")
            
            if agent_id in agents_by_id:
                edited_fields = []
                for key, value in changes.items():
                    if key in editable_fields:
                        agents_by_id[agent_id][key] = value
                        edited_fields.append(key)
                if edited_fields:
                    log(f"Domain '{domain}': Edited '{agent_id}' - fields={edited_fields} reason={reason}")
                    stats["agents_edited"] += 1
        
        # Add missing agents
        missing_agents = analysis.get("missing_agents", [])
        for new_agent in missing_agents:
            validated = validate_agent(new_agent, domain, tools)
            if validated:
                agent_id = validated.get("agent_id")
                if agent_id not in agents_by_id:
                    agents.append(validated)
                    agents_by_id[agent_id] = validated
                    role = validated.get("role_id", "?")
                    log(f"Domain '{domain}': Added new agent '{agent_id}' (role={role})")
                    stats["agents_added"] += 1
        
        # Additional local deduplication pass (similarity-based)
        # Group by role_id and check for high similarity within groups
        by_role = {}
        for agent in agents:
            role = agent.get("role_id", "unknown")
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(agent)
        
        final_agents = []
        merged_ids = set()
        
        for role, role_agents in by_role.items():
            if len(role_agents) <= 1:
                final_agents.extend(role_agents)
                continue
            
            # Check pairs for high similarity
            kept = []
            for i, agent in enumerate(role_agents):
                if agent.get("agent_id") in merged_ids:
                    continue
                
                should_keep = True
                for j, other in enumerate(role_agents):
                    if i >= j or other.get("agent_id") in merged_ids:
                        continue
                    
                    sim = calculate_similarity(agent, other)
                    if sim > 0.7:
                        # Keep the one with longer description
                        if len(other.get("description", "")) > len(agent.get("description", "")):
                            merged_ids.add(agent.get("agent_id"))
                            should_keep = False
                            log(f"Domain '{domain}': Merged '{agent.get('agent_id')}' into '{other.get('agent_id')}' (similarity={sim:.2f})")
                            stats["agents_merged"] += 1
                            break
                        else:
                            merged_ids.add(other.get("agent_id"))
                            log(f"Domain '{domain}': Merged '{other.get('agent_id')}' into '{agent.get('agent_id')}' (similarity={sim:.2f})")
                            stats["agents_merged"] += 1
                
                if should_keep:
                    kept.append(agent)
            
            final_agents.extend(kept)
        
        # Rebuild output - same format as input, no curation metadata
        result = {
            "domain": domain,
            "generated_at": data.get("generated_at"),
            "total_agents": len(final_agents),
            "agents": final_agents
        }
        
        # Log curation info (not in file)
        log(f"Domain '{domain}': Curation stats - original={original_count}, final={len(final_agents)}, added={len(missing_agents)}, removed={len(removed_ids)}, merged={len(merged_ids)}")
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        log(f"Domain '{domain}': Saved {len(final_agents)} agents (was {original_count})")
        stats["processed_domains"] += 1
        
        return True
        
    except Exception as e:
        log(f"Domain '{domain_name}': Error - {e}\n{tb.format_exc()}", "ERROR")
        stats["errors"] += 1
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# FOLDER PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

async def process_folder(
    folder_name: str,
    llm: ChatOpenAI,
    roles: list[str],
    tools: list[str],
    progress: dict,
    parallel: int
) -> dict:
    """Process all domain files in a folder."""
    
    input_folder = AGENTS_DIR / folder_name
    output_folder = OUTPUT_DIR / folder_name
    
    if not input_folder.exists():
        log(f"Folder '{folder_name}' not found, skipping", "WARN")
        console.print(f"[yellow]⚠ Folder '{folder_name}' not found, skipping[/yellow]")
        return progress["stats"]
    
    # Get all JSON files
    all_files = sorted(input_folder.glob("*.json"))
    
    # Get completed files for this folder
    completed = set(progress["completed"].get(folder_name, []))
    pending_files = [f for f in all_files if f.name not in completed]
    
    if not pending_files:
        console.print(f"[green]✓ Folder '{folder_name}': all {len(all_files)} domains already processed[/green]")
        log(f"Folder '{folder_name}': all {len(all_files)} domains already processed")
        return progress["stats"]
    
    log(f"Folder '{folder_name}': starting - {len(pending_files)} pending / {len(all_files)} total")
    console.print(f"\n[bold cyan]📁 {folder_name}[/bold cyan] ({len(pending_files)} pending / {len(all_files)} total)")
    
    stats = progress["stats"]
    semaphore = asyncio.Semaphore(parallel)
    lock = asyncio.Lock()
    processed_count = len(completed)
    total_count = len(all_files)
    
    # Create progress bar
    progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=2
    )
    
    async def process_with_semaphore(input_path: Path, task_id) -> bool:
        nonlocal processed_count
        async with semaphore:
            output_path = output_folder / input_path.name
            success = await process_domain_file(input_path, output_path, llm, roles, tools, stats)
            
            async with lock:
                if folder_name not in progress["completed"]:
                    progress["completed"][folder_name] = []
                progress["completed"][folder_name].append(input_path.name)
                processed_count += 1
                progress_bar.update(task_id, completed=processed_count)
                save_progress(progress)
            
            return success
    
    try:
        with progress_bar:
            task_id = progress_bar.add_task(
                f"[cyan]{folder_name}[/cyan]",
                total=total_count,
                completed=len(completed)
            )
            await asyncio.gather(*[process_with_semaphore(f, task_id) for f in pending_files])
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted! Progress saved.[/yellow]")
        raise
    
    # Folder complete summary
    console.print(f"[bold green]✓ Folder '{folder_name}' complete: {processed_count}/{total_count} domains[/bold green]")
    log(f"Folder '{folder_name}' complete: {processed_count}/{total_count} domains")
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def curate_all_async(
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    parallel: int = DEFAULT_PARALLEL,
    resume: bool = True,
    folders: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = MAX_RETRIES
):
    """Main async curation function."""
    global DEFAULT_TIMEOUT, MAX_RETRIES
    DEFAULT_TIMEOUT = timeout
    MAX_RETRIES = retries
    
    console.print(Panel.fit(
        "[bold cyan]🔬 Dataset Curator[/bold cyan]\n"
        "LLM-powered agent normalization and curation",
        border_style="cyan"
    ))
    
    # Load catalogs
    domains, roles, tools = load_catalogs()
    
    # Show config
    table = Table(title="Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input", str(AGENTS_DIR))
    table.add_row("Output", str(OUTPUT_DIR))
    table.add_row("Model", model)
    table.add_row("Parallel", str(parallel))
    table.add_row("Timeout", f"{timeout}s")
    table.add_row("Retries", str(retries))
    table.add_row("Roles", str(len(roles)))
    table.add_row("Tools", str(len(tools)))
    console.print(table)
    
    # Load or initialize progress
    if resume:
        progress = load_progress()
        total_completed = sum(len(v) for v in progress["completed"].values())
        if total_completed > 0:
            console.print(f"\n[green]✓ Resuming: {total_completed} domains already processed[/green]")
    else:
        clear_progress()
        progress = load_progress()
    
    # Create LLM client
    llm = create_llm(api_key, base_url, model)
    
    # Process folders
    folders_to_process = folders if folders else DATASET_FOLDERS
    
    log(f"=" * 80)
    log(f"Starting curation: folders={folders_to_process}, parallel={parallel}, model={model}")
    log(f"=" * 80)
    
    try:
        for folder_name in folders_to_process:
            await process_folder(folder_name, llm, roles, tools, progress, parallel)
    except KeyboardInterrupt:
        console.print("\n[yellow]Saving progress and exiting...[/yellow]")
        save_progress(progress)
        console.print("[green]Progress saved. Run again to resume.[/green]")
        return
    
    # Final stats
    stats = progress["stats"]
    
    console.print()
    stats_table = Table(title="Curation Complete", show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Domains processed", str(stats["processed_domains"]))
    stats_table.add_row("Agents added", f"[green]+{stats['agents_added']}[/green]")
    stats_table.add_row("Agents removed", f"[red]-{stats['agents_removed']}[/red]")
    stats_table.add_row("Agents merged", f"[yellow]{stats['agents_merged']}[/yellow]")
    stats_table.add_row("Agents edited", f"[cyan]{stats['agents_edited']}[/cyan]")
    stats_table.add_row("Errors", str(stats["errors"]))
    stats_table.add_row("Output", str(OUTPUT_DIR))
    console.print(stats_table)
    
    # Clear progress on complete
    total_expected = 0
    for folder in folders_to_process:
        folder_path = AGENTS_DIR / folder
        if folder_path.exists():
            total_expected += len(list(folder_path.glob("*.json")))
    
    total_completed = sum(len(v) for v in progress["completed"].values())
    
    if total_completed >= total_expected:
        clear_progress()
        console.print("\n[bold green]✓ All folders processed successfully![/bold green]")
    
    log(f"=" * 80)
    log(f"Curation complete: domains={stats['processed_domains']} +{stats['agents_added']} -{stats['agents_removed']} ~{stats['agents_merged']} ✎{stats['agents_edited']} errors={stats['errors']}")
    log(f"=" * 80)


def curate_all(
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
    asyncio.run(curate_all_async(api_key, base_url, model, parallel, resume, folders, timeout, retries))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Curate and normalize agent datasets using LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run curation with default settings (resumes if interrupted)
  python dataset_curator.py
  
  # Run with 50 parallel requests
  python dataset_curator.py -p 50
  
  # Start fresh (ignore previous progress)
  python dataset_curator.py --fresh
  
  # Process specific folders only
  python dataset_curator.py --folders agents_eng agents_rus
  
  # Custom API settings
  python dataset_curator.py --base-url http://localhost:8080/v1 --model my-model
  
  # Adjust timeout and retries for slow servers
  python dataset_curator.py --timeout 600 --retries 5
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
        help=f"Number of parallel domain requests (default: {DEFAULT_PARALLEL})"
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
        help=f"Specific folders to process (default: {DATASET_FOLDERS})"
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
        curate_all(
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
