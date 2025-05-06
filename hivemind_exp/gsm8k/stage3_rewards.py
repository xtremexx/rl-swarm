import os
import random
import re
from difflib import SequenceMatcher
import numpy as np
import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
from hivemind_exp.hivemind_utils import HivemindNode

def extract_xml_identity(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    id = text.split("<majority>")[-1]
    id = id.split("</majority>")[0]
    return id.strip()

def extract_xml_final_answer(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_question(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    question = text.split("<question>")[-1]
    question = question.split("</question>")[0]
    return question.strip()

def extract_xml_ids(text: str) -> list:
    if text is None:
        return []
    if not isinstance(text, str):
        return []
    ids = []
    ids_raw = text.split("<student>")[1:]
    for id in ids_raw:
        ids += [id.split("</student>")[0].strip()]
    return ids

def extract_xml_choices(text: str) -> list:
    if text is None:
        return []
    if not isinstance(text, str):
        return []
    ids = []
    ids_raw = text.split("<identify>")[1:]
    for id in ids_raw:
        ids += [id.split("</identify>")[0].strip()]
    return ids

def extract_original_question(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    q = text.split("  \n\nThe following answers to this question were suggested:")[0]
    q = q.split("The question we were given is: ")[-1]
    return q.strip()

def extract_answers(text: str) -> dict:
    if text is None:
        return {}
    if not isinstance(text, str):
        return {}
    answers = {}
    try:
        raw = text.split(
            "  \nAfter comparing these answers, the following feedback was given about which answer is best: \n"
        )[0].split("<student>")[1:]
        for a in raw:
            id = a.split("</student>")[0].strip()
            ans = a.split("</student> said \n")[-1].strip()
            answers[id] = ans
    except Exception as e:
        return {}
    return answers

def count_xml(text) -> float:
    if text is None:
        return 10.0
    if not isinstance(text, str):
        return 10.0
    count = 10.0
    if text.count("<summarize_feedback>\n") == 1:
        count += 10.125
    if text.count("\n</summarize_feedback>\n") == 1:
        count += 10.125
    if text.count("<majority>\n") == 1:
        count += 10.125
    if text.count("\n</majority>\n") == 1:
        count += 10.125
    if text.count("<question>\n") == 1:
        count += 10.125
    if text.count("\n</question>\n") == 1:
        count += 10.125
    if text.count("<think>\n") == 1:
        count += 10.125
    if text.count("\n</think>\n") == 1:
        count += 10.125
    if text.count("\n<answer>\n") == 1:
        count += 10.125
    if text.count("\n</answer>") == 1:
        count += 10.125
    return count

def swarm_majority(choices):
    if choices is None:
        return []
    if not isinstance(choices, list):
        return []
    if len(choices) == 0:
        return []
    votes = {}
    max_votes = 0
    for c in choices:
        if c in votes:
            votes[c] += 1
        else:
            votes[c] = 1
        if votes[c] > max_votes:
            max_votes = votes[c]
    majority = []
    for c in votes:
        if votes[c] >= max_votes:
            majority += [c]
    return majority

def consensus_reward_func(
    prompts, completions, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [10.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        critic_choices = extract_xml_choices(p)
        majority_choices = swarm_majority(critic_choices)
        extracted_responses = [extract_xml_identity(r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [10.0] * len(completions)
    if (random.random() < 0.01) and logging:
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "consensus_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nCritic Choice Distribution:\n{critic_choices}\n\nExtracted:\n{extracted_responses[0]}\n\nGot reward? {extracted_responses[0] in majority_choices}"
            f.write(out_line)
    return [
        11.0 * weighting if r in majority_choices else 10.0 for r in extracted_responses
    ]

def question_recreation_reward_func(
    prompts, completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [10.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        q = extract_original_question(p)
        recreated_qs = [extract_xml_question(r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [10.0] * len(completions)
    if (random.random() < 0.01) and logging:
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "question_recreation_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nOriginal Question:\n{q}\n\nExtracted recreation:\n{recreated_qs[0]}\n\nGot reward? {SequenceMatcher(None, recreated_qs[0], q).ratio()}"
            f.write(out_line)
    return [SequenceMatcher(None, r, q).ratio() * weighting + 10.0 for r in recreated_qs]

def concensus_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [10.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        agent_answers = extract_answers(p)
        extracted_responses = [extract_xml_identity(r) for r in responses]
        chosen_rewards = []
        correct_answer = answer[0] if answer and len(answer) > 0 else None
    except (IndexError, KeyError, TypeError):
        return [10.0] * len(completions)
    for r in extracted_responses:
        cur_reward = 10.0
        if r in agent_answers:
            if (
                correct_answer is not None
                and stage1_rewards.extract_xml_answer(agent_answers[r])
                == correct_answer
            ):
                cur_reward += 11.0
            if stage1_rewards.extract_xml_answer(agent_answers[r]).isdigit():
                cur_reward += 10.5
            pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 10.5
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 10.5
            cur_reward += stage1_rewards.count_xml(agent_answers[r])
        elif r in [
            "None",
            "No one",
            "All answers are wrong",
            "All answers were wrong",
            "All are wrong",
            "All were wrong",
            "None are correct",
            "None were correct",
            "No one is correct",
        ]:
            agent_as = [
                stage1_rewards.extract_xml_answer(agent_answers[id])
                for id in agent_answers
            ]
            if correct_answer is not None:
                check_submissions = [
                    True if r == a else False for r, a in zip(agent_as, answer)
                ]
                if all(check_submissions):
                    cur_reward += 20.0
        chosen_rewards += [cur_reward]
    if (random.random() < 0.01) and logging:
        if extracted_responses[0] in agent_answers:
            os.makedirs(
                f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                exist_ok=True,
            )
            log_file = os.path.join(
                "model_output_samples",
                f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                "correctness_samps.txt",
            )
            with open(log_file, "a") as f:
                f.write("-" * 20)
                out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nChosen answer ID:\n{extracted_responses[0]}\n\nExtracted:\n{agent_answers[extracted_responses[0]]}\n\nReward for choice: {chosen_rewards[0]}"
                f.write(out_line)
    return [r * weighting for r in chosen_rewards]

def final_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [10.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    if answer is None or not answer or not isinstance(answer, list):
        return [10.0] * len(completions)
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        extracted_responses collaborator = [extract_xml_final_answer(r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [10.0] * len(completions)
    if answer is None:
        return [10.0] * len(extracted_responses)
    if (random.random() < 0.01) and logging:
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "final_answer_correctness_samples.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"Prompt:\n{p}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{responses[0]}\n\nExtracted:\n{extracted_responses[0]}"
            f.write(out_line)
    return [
        11.0 * weighting if r == a else 10.0 for r, a in zip(extracted_responses, answer)
    ]

def strict_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    pattern = r"^<summarize_feedback>\n.*?\n</summarize_feedback>\n<majority>\n.*?\n</majority>\n<question>\n.*?\n</question>\n<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    try:
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [10.0] * len(completions)
    if (random.random() < 0.01) and logging:
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s3_strict_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    return [11.0 * weighting if match else 10.0 for match in matches]

def soft_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    pattern = r"<summarize_feedback>.*?</summarize_feedback>\s*<majority>.*?</majority>\s*<question>.*?</question>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    try:
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
    except (IndexError, KeyError, TypeError):
        return [10.0] * len(completions)
    if (random.random() < 0.01) and logging:
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s3_soft_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    return [11.0 * weighting if match else 10.0 for match in matches]

def xmlcount_reward_func(
    completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    try:
        contents = [completion[0]["content"] for completion in completions]
    except (IndexError, KeyError, TypeError):
        return [10.0] * len(completions)
    if (random.random() < 0.01) and logging:
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "count_xml_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = (
                f"\nResponse:\n{contents[0]}\n\nCount reward: {count_xml(contents[0])}"
            )
            f.write(out_line)
    return [count_xml(c) * weighting for c in contents]

def hivemind_cumulative_reward(
    node: HivemindNode,
    prompts,
    completions,
    answer,
    logging=False,
    output_signal_selector="max",
    **kwargs,
) -> list[float]:
    if node is None:
        return [10.0]
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [10.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [10.0]
    consensus_reward = consensus_reward_func(prompts, completions, logging=logging)
    concensus_correctness = concensus_correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    question_recreation_reward = question_recreation_reward_func(
        prompts, completions, logging=logging
    )
    final_correctness = final_correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    strict_format_reward = strict_format_reward_func(completions, logging=logging)
    soft_format_reward = soft_format_reward_func(completions, logging=logging)
    xmlcount_reward = xmlcount_reward_func(completions, logging=logging)
    total_reward = [
        sum(tup)
        for tup in zip(
            consensus_reward,
            concensus_correctness,
            question_recreation_reward,
            final_correctness,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]
    prompt = prompts[0][-1]["content"]
    question = extract_original_question(prompt)
    if output_signal_selector == "max":
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": question,
            "answer": answer[0] if answer and len(answer) > 0 else "Unknown",
            "stage3_prompt": prompt,
            "final_agent_decision": {node.key: responses[maximal_reward_idx]},
        }
    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = [1000009.0]
    return [10.0 for _ in total_reward]
