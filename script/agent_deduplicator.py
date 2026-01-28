#!/usr/bin/env python3
"""
Agent Deduplicator - Removes near-duplicate agents using semantic similarity.

Uses sentence embeddings + FAISS for efficient similarity search,
then applies union-find clustering to identify duplicate groups.

Features:
- Works with multiple agent directories: agents_rus, agents_eng, agents_big
- Processes each domain file separately within each directory
- Checkpoint support: can resume from where it stopped
- Reports statistics at the end
"""

import json
import os
import pickle
import shutil
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
from tqdm import tqdm


# ============ Configuration ============
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
AGENTS_DIR = DATASET_DIR / "agents"
LOG_FILE = SCRIPT_DIR / "dedup.log"
STATS_JSON_FILE = SCRIPT_DIR / "dedup_stats.json"

# Agent directories configuration
AGENT_DIRS = [
    ("agents_rus", AGENTS_DIR / "agents_rus"),
    ("agents_eng", AGENTS_DIR / "agents_eng"),
    ("agents_big", AGENTS_DIR / "agents_temp_03_big"),
]

CHECKPOINT_FILE = SCRIPT_DIR / ".dedup_checkpoint.pkl"
EMBEDDINGS_CACHE_DIR = SCRIPT_DIR / ".embeddings_cache"

# Similarity settings
TOP_K = 50  # Number of nearest neighbors to check
SIMILARITY_THRESHOLD = 0.87  # cosine_similarity > this = duplicate

# Model settings
EMBEDDING_MODEL = "all-mpnet-base-v2"


# ============ Logging Setup ============
def setup_logging():
    """Setup logging to both file and console."""
    logger = logging.getLogger("dedup")
    logger.setLevel(logging.INFO)
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    # Console handler - brief output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


log = setup_logging()


# ============ Union-Find Data Structure ============
class UnionFind:
    """Disjoint Set Union (Union-Find) for clustering duplicates."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_clusters(self) -> dict[int, list[int]]:
        """Returns {root: [members]} for all clusters."""
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


# ============ Checkpoint Management ============
@dataclass
class Checkpoint:
    """Saves progress so we can resume later."""
    processed_files: set = field(default_factory=set)  # Set of "dir_name:file_name"
    processed_dirs: set = field(default_factory=set)   # Set of completed directory names
    stats: dict = field(default_factory=lambda: {
        "total_original": 0,
        "total_removed": 0,
        "total_final": 0,
        "by_dir": {}  # Stats per directory
    })


def save_checkpoint(checkpoint: Checkpoint, path: str = CHECKPOINT_FILE):
    """Save checkpoint to disk."""
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: str = CHECKPOINT_FILE) -> Checkpoint:
    """Load checkpoint from disk or create new one."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        log.info(f"Loaded checkpoint: {len(checkpoint.processed_files)} files already processed")
        return checkpoint
    return Checkpoint()


# ============ Stats JSON Management ============
def load_stats_json() -> dict:
    """Load or create stats JSON file."""
    if STATS_JSON_FILE.exists():
        with open(STATS_JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "last_run": None,
        "summary": {
            "total_original": 0,
            "total_removed": 0,
            "total_remaining": 0,
            "removal_percentage": 0.0
        },
        "by_directory": {},
        "domains": []
    }


def save_stats_json(stats: dict):
    """Save stats to JSON file."""
    with open(STATS_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def update_domain_stats(stats: dict, dir_name: str, domain: str, original: int, removed: int):
    """Update stats for a single domain."""
    remaining = original - removed
    
    # Update domain entry
    domain_entry = {
        "directory": dir_name,
        "domain": domain,
        "original": original,
        "removed": removed,
        "remaining": remaining,
        "removal_percentage": round(removed / max(original, 1) * 100, 1)
    }
    
    # Find and update or append
    found = False
    for i, d in enumerate(stats["domains"]):
        if d["directory"] == dir_name and d["domain"] == domain:
            stats["domains"][i] = domain_entry
            found = True
            break
    if not found:
        stats["domains"].append(domain_entry)
    
    # Update directory totals
    if dir_name not in stats["by_directory"]:
        stats["by_directory"][dir_name] = {"original": 0, "removed": 0, "remaining": 0}
    
    # Recalculate directory totals from domains
    dir_domains = [d for d in stats["domains"] if d["directory"] == dir_name]
    stats["by_directory"][dir_name] = {
        "original": sum(d["original"] for d in dir_domains),
        "removed": sum(d["removed"] for d in dir_domains),
        "remaining": sum(d["remaining"] for d in dir_domains)
    }
    
    # Recalculate summary
    stats["summary"]["total_original"] = sum(d["original"] for d in stats["domains"])
    stats["summary"]["total_removed"] = sum(d["removed"] for d in stats["domains"])
    stats["summary"]["total_remaining"] = stats["summary"]["total_original"] - stats["summary"]["total_removed"]
    stats["summary"]["removal_percentage"] = round(
        stats["summary"]["total_removed"] / max(stats["summary"]["total_original"], 1) * 100, 1
    )
    stats["last_run"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    save_stats_json(stats)


# ============ Embedding Functions ============
def get_embedding_model():
    """Lazily load the sentence transformer model."""
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def create_agent_text(agent: dict) -> str:
    """Create text representation of agent for embedding."""
    parts = [
        agent.get('agent_id', ''),
        agent.get('description', ''),
        agent.get('persona', ''),
        ' '.join(agent.get('tools', []))
    ]
    return ' '.join(parts)


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    # Replace invalid Windows filename characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


def get_embeddings_for_agents(
    model, 
    agents: list[dict], 
    dir_name: str,
    domain: str,
    cache_dir: str = EMBEDDINGS_CACHE_DIR
) -> np.ndarray:
    """Get or compute embeddings for agents."""
    os.makedirs(cache_dir, exist_ok=True)
    # Sanitize domain name for use in filename
    safe_domain = sanitize_filename(domain)
    # Include dir_name in cache path to avoid conflicts
    cache_path = os.path.join(cache_dir, f"{dir_name}_{safe_domain}_embeddings.npy")
    ids_path = os.path.join(cache_dir, f"{dir_name}_{safe_domain}_ids.json")
    
    # Check cache
    if os.path.exists(cache_path) and os.path.exists(ids_path):
        with open(ids_path, 'r') as f:
            cached_ids = json.load(f)
        current_ids = [a['agent_id'] for a in agents]
        if cached_ids == current_ids:
            return np.load(cache_path)
    
    # Compute embeddings
    print(f"  Computing embeddings for {len(agents)} agents...")
    texts = [create_agent_text(a) for a in agents]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # Save to cache
    np.save(cache_path, embeddings)
    with open(ids_path, 'w') as f:
        json.dump([a['agent_id'] for a in agents], f)
    
    return embeddings


# ============ FAISS Index Functions ============
def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index for fast similarity search."""
    import faiss
    
    dimension = embeddings.shape[1]
    
    # Use Inner Product since embeddings are normalized (= cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    return index


def find_duplicates(
    agents: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD
) -> tuple[list[dict], int]:
    """
    Find and cluster duplicate agents.
    Returns (list of agents to keep, number of removed agents).
    """
    import faiss
    
    n = len(agents)
    if n <= 1:
        return agents, 0
    
    # Build index
    index = build_faiss_index(embeddings)
    
    # Search for nearest neighbors
    k = min(top_k + 1, n)  # +1 because each point finds itself
    
    # Search in batches to avoid memory issues
    batch_size = 1000
    all_distances = []
    all_indices = []
    
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_embeddings = embeddings[i:end].astype('float32')
        distances, indices = index.search(batch_embeddings, k)
        all_distances.append(distances)
        all_indices.append(indices)
    
    distances = np.vstack(all_distances)
    indices = np.vstack(all_indices)
    
    # Build Union-Find structure
    uf = UnionFind(n)
    duplicate_pairs = 0
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            if j == i:
                continue
            similarity = distances[i, j_idx]  # Already cosine similarity for normalized vectors
            if similarity > threshold:
                uf.union(i, j)
                duplicate_pairs += 1
    
    # Get clusters and select representatives
    clusters = uf.get_clusters()
    
    kept_agents = []
    removed = 0
    
    for root, members in clusters.items():
        if len(members) == 1:
            kept_agents.append(agents[members[0]])
        else:
            # Select representative: prefer the one with longest description
            best_idx = max(members, key=lambda idx: len(agents[idx].get('description', '')))
            kept_agents.append(agents[best_idx])
            removed += len(members) - 1
    
    return kept_agents, removed


# ============ File Operations ============
def get_domain_files(agents_dir: Path) -> list[Path]:
    """Get all domain JSON files from agents directory."""
    if not agents_dir.exists():
        log.error(f"Agents directory not found: {agents_dir}")
        return []
    
    files = sorted(agents_dir.glob("*.json"))
    return files


def load_domain_file(file_path: Path) -> dict:
    """Load a domain file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_domain_file(file_path: Path, data: dict):
    """Save a domain file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============ Main Processing ============
def process_domain_file(
    model,
    dir_name: str,
    file_path: Path,
    stats_json: dict,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD
) -> tuple[int, int]:
    """
    Process a single domain file.
    Returns (original_count, removed_count).
    """
    data = load_domain_file(file_path)
    domain = data.get('domain', file_path.stem)
    agents = data.get('agents', [])
    original_count = len(agents)
    
    if original_count <= 1:
        log.info(f"{dir_name}/{domain}: {original_count} agent(s) - skipped (too few)")
        update_domain_stats(stats_json, dir_name, domain, original_count, 0)
        return original_count, 0
    
    # Get embeddings
    embeddings = get_embeddings_for_agents(model, agents, dir_name, domain)
    
    # Find duplicates
    kept_agents, removed = find_duplicates(
        agents, 
        embeddings,
        top_k=top_k,
        threshold=threshold
    )
    
    # Update and save file
    data['agents'] = kept_agents
    data['total_agents'] = len(kept_agents)
    if 'deduplication_info' not in data:
        data['deduplication_info'] = {}
    data['deduplication_info'].update({
        'original_count': original_count,
        'removed_count': removed,
        'similarity_threshold': threshold
    })
    
    save_domain_file(file_path, data)
    
    # Log the result
    if removed > 0:
        log.info(f"{dir_name}/{domain}: {original_count} -> {len(kept_agents)} agents (removed {removed})")
    else:
        log.info(f"{dir_name}/{domain}: {original_count} agents - no duplicates found")
    
    # Update JSON stats
    update_domain_stats(stats_json, dir_name, domain, original_count, removed)
    
    return original_count, removed


def process_directory(
    model,
    dir_name: str,
    agents_dir: Path,
    checkpoint: Checkpoint,
    stats_json: dict,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    domain_filter: Optional[str] = None
) -> tuple[int, int]:
    """
    Process all domain files in a directory.
    Returns (total_original, total_removed) for this directory.
    """
    log.info(f"")
    log.info(f"{'='*50}")
    log.info(f"Processing directory: {dir_name}")
    log.info(f"Path: {agents_dir}")
    log.info(f"{'='*50}")
    
    # Get domain files
    domain_files = get_domain_files(agents_dir)
    
    if not domain_files:
        log.error(f"No domain files found in {agents_dir}!")
        return 0, 0
    
    # Filter to specific domain if requested
    if domain_filter:
        domain_name = domain_filter.replace('.json', '')
        domain_files = [f for f in domain_files if f.stem == domain_name]
        if not domain_files:
            log.error(f"Domain file not found: {domain_filter}")
            return 0, 0
    
    log.info(f"Found {len(domain_files)} domain files")
    
    dir_original = 0
    dir_removed = 0
    
    for file_path in tqdm(domain_files, desc=f"Processing {dir_name}"):
        file_key = f"{dir_name}:{file_path.name}"
        
        if file_key in checkpoint.processed_files:
            continue
        
        original, removed = process_domain_file(
            model, dir_name, file_path, stats_json,
            top_k=top_k,
            threshold=threshold
        )
        
        dir_original += original
        dir_removed += removed
        
        # Update checkpoint
        checkpoint.processed_files.add(file_key)
        checkpoint.stats["total_original"] += original
        checkpoint.stats["total_removed"] += removed
        checkpoint.stats["total_final"] = checkpoint.stats["total_original"] - checkpoint.stats["total_removed"]
        
        # Update per-directory stats
        if dir_name not in checkpoint.stats["by_dir"]:
            checkpoint.stats["by_dir"][dir_name] = {"original": 0, "removed": 0}
        checkpoint.stats["by_dir"][dir_name]["original"] += original
        checkpoint.stats["by_dir"][dir_name]["removed"] += removed
        
        save_checkpoint(checkpoint)
    
    return dir_original, dir_removed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate agents using semantic similarity"
    )
    parser.add_argument(
        "--clear-cache", 
        action="store_true",
        help="Clear embeddings cache and checkpoint before running"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for duplicates (default: {SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of nearest neighbors to check (default: {TOP_K})"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Process only specific domain file (e.g., 'Computing' or 'Computing.json')"
    )
    parser.add_argument(
        "--dir",
        type=str,
        choices=["agents_rus", "agents_eng", "agents_big"],
        default=None,
        help="Process only specific directory"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    log.info("=" * 50)
    log.info("Agent Deduplicator (Multi-directory)")
    log.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 50)
    
    # Clear cache if requested
    if args.clear_cache:
        log.info("Clearing cache and checkpoint...")
        if os.path.exists(EMBEDDINGS_CACHE_DIR):
            shutil.rmtree(EMBEDDINGS_CACHE_DIR)
            log.info(f"   Removed {EMBEDDINGS_CACHE_DIR}/")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            log.info(f"   Removed {CHECKPOINT_FILE}")
        if STATS_JSON_FILE.exists():
            os.remove(STATS_JSON_FILE)
            log.info(f"   Removed {STATS_JSON_FILE}")
    
    log.info(f"Configuration: model={EMBEDDING_MODEL}, threshold={args.threshold}, top_k={args.top_k}")
    log.info(f"Log file: {LOG_FILE}")
    log.info(f"Stats JSON: {STATS_JSON_FILE}")
    
    # Filter directories if specific one requested
    dirs_to_process = AGENT_DIRS
    if args.dir:
        dirs_to_process = [(name, path) for name, path in AGENT_DIRS if name == args.dir]
        if not dirs_to_process:
            log.error(f"Directory not found: {args.dir}")
            return
    
    log.info(f"Directories to process: {[name for name, _ in dirs_to_process]}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Load stats JSON
    stats_json = load_stats_json()
    
    # Load embedding model
    model = get_embedding_model()
    
    try:
        for dir_name, agents_dir in dirs_to_process:
            if dir_name in checkpoint.processed_dirs and not args.domain:
                log.info(f"Skipping directory {dir_name} (already fully processed)")
                continue
            
            if not agents_dir.exists():
                log.warning(f"Skipping {dir_name}: directory not found")
                continue
            
            dir_original, dir_removed = process_directory(
                model, dir_name, agents_dir, checkpoint, stats_json,
                top_k=args.top_k,
                threshold=args.threshold,
                domain_filter=args.domain
            )
            
            # Mark directory as completed (only if no domain filter)
            if not args.domain:
                checkpoint.processed_dirs.add(dir_name)
                save_checkpoint(checkpoint)
    
    except KeyboardInterrupt:
        log.warning("Interrupted! Progress has been saved. Run the script again to continue.")
        save_checkpoint(checkpoint)
        return
    
    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    # Print final statistics
    stats = checkpoint.stats
    total_original = stats.get("total_original", 0)
    total_removed = stats.get("total_removed", 0)
    total_final = total_original - total_removed
    
    log.info("")
    log.info("=" * 50)
    log.info("FINAL STATISTICS")
    log.info("=" * 50)
    
    # Per-directory stats
    for dir_name in ["agents_rus", "agents_eng", "agents_big"]:
        if dir_name in stats.get("by_dir", {}):
            dir_stats = stats["by_dir"][dir_name]
            orig = dir_stats.get("original", 0)
            rem = dir_stats.get("removed", 0)
            final = orig - rem
            pct = rem / max(orig, 1) * 100
            log.info(f"{dir_name}: {orig:,} -> {final:,} (removed {rem:,}, {pct:.1f}%)")
    
    log.info("-" * 50)
    log.info(f"TOTAL: {total_original:,} -> {total_final:,} (removed {total_removed:,}, {total_removed/max(total_original,1)*100:.1f}%)")
    log.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
