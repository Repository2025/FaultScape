import os
import json
from pairwise_rerank import pairwise_ranking
from config import TOP_K

def load_code_snippets_from_dir(dir_path, top_k=10):
    snippets = []
    files = sorted(os.listdir(dir_path))  
    for filename in files:
        if len(snippets) >= top_k:
            break
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    snippets.append(content)
    return snippets


def load_test_failure_info(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_failure_info", "")


if __name__ == "__main__":
    snippets_folder = "./FL/ranking_task/run_model/sus_pos_rerank_new"
    test_failure_info_path = "./pairwise_rerank/test_failure_info.json"

    print(f"Loading up to {TOP_K} code snippets from folder: {snippets_folder}")
    candidate_code_snippets = load_code_snippets_from_dir(snippets_folder, TOP_K)

    print(f"Loading test failure info from: {test_failure_info_path}")
    test_failure_info = load_test_failure_info(test_failure_info_path)

    print(f"Loaded {len(candidate_code_snippets)} code snippets and test failure info, start pairwise ranking...")

    ranking = pairwise_ranking(test_failure_info, candidate_code_snippets)

    print("\n=== Final Ranking ===")
    for rank_idx, (idx, score) in enumerate(ranking, 1):
        snippet_preview = candidate_code_snippets[idx]
        preview = snippet_preview.replace("\n", " ")[:80]
        print(f"Rank {rank_idx}: Snippet #{idx} - Score: {score:.4f} - Code Preview: {preview}")
