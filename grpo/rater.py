def score_group(samples, tool_trace_checker, llm_judge=None, weights=(0.0, 1.0)):
    """Compose tool correctness and LLM judge signals for GRPO rewards."""
    scores = []
    for sample in samples:
        tool_score = tool_trace_checker(sample)
        llm_score = llm_judge(sample) if llm_judge else 0.0
        score = weights[0] * llm_score + weights[1] * tool_score
        scores.append(score)
    return scores
