# pairwise_rerank.py

import openai
import itertools
import json
from config import API_KEY, API_BASE, MODEL_NAME, TOP_K

openai.api_key = API_KEY
openai.api_base = API_BASE

def build_prompt(test_failure_context, code_snippet_1, code_snippet_2):
    prompt = f"""
You are provided with a failing test context from a Java project and two code snippets.
Your task is to:
1. Compare the marked code lines in the two snippets.
2. Analyze which snippet is more likely to contain the bug causing the failing test,
and provide a confidence score for your decision
3. Return your analysis in JSON format with the following fields:
(1) Compare_BugCause
(2) Compare_Result
(3) Compare_Confidence

The "Failing Test Context" is:
{test_failure_context}

The "Two Code Snippets" are:
[Code Snippet 1]
{code_snippet_1}

[Code Snippet 2]
{code_snippet_2}
"""
    return prompt.strip()

def query_model(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert in fault localization of Java code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        output = response['choices'][0]['message']['content'].strip()
        return output
    except Exception as e:
        print(f"Model error: {e}")
        return None

def parse_response(response_text):
    """
     JSON：
    {
      "Compare_BugCause": "The first snippet does not handle null values properly...",
      "Compare_Result": "Snippet 1",
      "Compare_Confidence": 0.85
    }
    """
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
            parsed = json.loads(json_str)
            if all(k in parsed for k in ("Compare_BugCause", "Compare_Result", "Compare_Confidence")):
                return parsed
        print("Warning: Response JSON format invalid, fallback to default")
    except Exception as e:
        print(f"Parse error: {e}")

    return {
        "Compare_BugCause": "",
        "Compare_Result": None,
        "Compare_Confidence": 0.0
    }

def pairwise_ranking(test_failure_context, code_snippets):
    scores = [0.0] * len(code_snippets)
    counts = [0] * len(code_snippets)

    for i, j in itertools.combinations(range(len(code_snippets)), 2):
        prompt = build_prompt(test_failure_context, code_snippets[i], code_snippets[j])
        response = query_model(prompt)
        if response is None:
            continue
        parsed = parse_response(response)

        winner = parsed.get("Compare_Result")
        confidence = parsed.get("Compare_Confidence", 0.0)
        if winner == "Snippet 1":
            scores[i] += confidence
            counts[i] += 1
        elif winner == "Snippet 2":
            scores[j] += confidence
            counts[j] += 1
        else:
            pass

    final_scores = [s / c if c > 0 else 0.0 for s, c in zip(scores, counts)]
    ranked = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    return ranked
