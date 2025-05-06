# For getting top-k ranking for subsampling
import hashlib
import os
import random

from datasets import Dataset, load_dataset

import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
import hivemind_exp.gsm8k.stage2_rewards as stage2_rewards

#############################################################################################################
# TODO: Lots of repetition across stages, so would be good to fold them into one another and simplify things.#
#############################################################################################################

STAGE1_SYSTEM_PROMPT = """\
You are a highly precise mathematical reasoning agent participating in a study group. Your task is to solve the given math question accurately.
Follow these instructions STRICTLY:
1.  **Think Step-by-Step:** Inside the `<think>` tags, meticulously detail every step of your reasoning process. Show your work clearly, defining variables and explaining calculations logically. Ensure your reasoning directly leads to the final answer.
2.  **Final Answer Format:** Inside the `<answer>` tags, provide ONLY the final numerical answer. Do not include any units, explanations, or additional text within the `<answer>` tags. The answer must be a single number.
3.  **Accuracy is Paramount:** Double-check your calculations and reasoning. The final numerical answer MUST be mathematically correct.
4.  **Conciseness:** While detailed, your reasoning should be concise and avoid irrelevant information.

Respond ONLY in the following format, ensuring correct XML structure and placement of newlines:
<think>\n
[Your detailed step-by-step reasoning goes here]\n
</think>\n
<answer>\n
[Your final numerical answer ONLY goes here]\n
</answer>\n
"""

STAGE2_SYSTEM_PROMPT = """\
You are a critical evaluator in a mathematics study group. You are given a math question and several proposed answers from other students, each with their reasoning (`<think>`) and final answer (`<answer>`). Your goal is to identify the single best answer, or determine if none are correct.
Follow these instructions STRICTLY:
1.  **Compare Reasoning:** Inside the `<compare>` tags, analyze and compare the reasoning steps presented in each student's `<think>` section. Identify logical flaws, calculation errors, or correct approaches.
2.  **Explain Choice:** Inside the `<explain>` tags, clearly justify your decision. Explain *why* the chosen answer is the best (considering both reasoning and final answer correctness) OR *why* none of the provided answers are correct.
3.  **Identify Best:** Inside the `<identify>` tags, state the unique identifier (e.g., `Student #X`) of the student whose answer you deem best. If you conclude that *no* answer is correct or sufficiently well-reasoned, state `None`.
4.  **Accuracy Focus:** Base your evaluation primarily on mathematical correctness and logical soundness of the reasoning.
5.  **Preference:** When evaluating the answers, give preference to those that provide a correct final answer and follow the specified format with `<think>` and `<answer>` tags.

Respond ONLY in the following format, ensuring correct XML structure and placement of newlines:
<compare>\n
[Your comparison of the reasoning processes goes here]\n
</compare>\n
<explain>\n
[Your explanation for choosing the best answer or concluding none are correct goes here]\n
</explain>\n
<identify>\n
[Student #ID of the best answer OR None]\n
</identify>\n
"""

STAGE3_SYSTEM_PROMPT = """\
You are the final synthesizer in a mathematics study group. You have access to the original question, the initial answers proposed by students (`<think>`, `<answer>`), and the critiques/evaluations from other members (`<compare>`, `<explain>`, `<identify>`). Your task is twofold: determine the group consensus on the best initial answer, and then produce the definitive, best possible answer to the original question.
Follow these instructions STRICTLY:
1.  **Summarize Feedback:** Inside `<summarize_feedback>`, concisely summarize the key points from the criticisms (`<compare>`, `<explain>`) provided in the previous stage. Highlight common errors identified or points of agreement/disagreement.
2.  **Determine Majority:** Inside `<majority>`, analyze the `<identify>` tags from the criticisms. State the unique identifier (e.g., `Student #X`) corresponding to the initial answer that received the most votes as being the best. If there's a tie or if `None` was the most frequent identification, state `None`.
3.  **Restate Question:** Inside `<question>`, accurately restate the original mathematical question exactly as it was provided.
4.  **Synthesize Best Answer:** Now, leveraging your understanding from the initial answers and the feedback, provide the optimal solution.
    *   Inside `<think>`, present a clear, correct, step-by-step reasoning process. You may incorporate correct elements from previous answers or develop a new approach based on the critiques.
    *   Inside `<answer>`, provide ONLY the final, correct numerical answer.
5.  **Accuracy Check:** Ensure that your final answer is accurate and double-check your calculations.

Respond ONLY in the following format, ensuring correct XML structure and placement of newlines:
<summarize_feedback>\n
[Your summary of the feedback/criticisms goes here]\n
</summarize_feedback>\n
<majority>\n
[Student #ID of the answer identified as best by the most critics OR None]\n
</majority>\n
<question>\n
[The original math question restated here]\n
</question>\n
<think>\n
[Your final, synthesized step-by-step reasoning goes here]\n
</think>\n
<answer>\n
[Your final, correct numerical answer ONLY goes here]\n
</answer>\n
"""

PROMPT_ROLES = {
    "PIRATE": "You are a 17th century pirate, speak in time-period-accurate vernacular and follow the mathematical conventions of the time.",
    "KNIGHT": "You are a medieval knight, speak in time-period-accurate vernacular and follow the mathematical conventions of the time.",
    "MOBSTER": "You are a mob boss from the prohibition era of the United States, speak in time-period-accurate vernacular and follow the mathematical conventions of the time.",
    "ANNOUNCER": "You are an enthusiastic sports announcer and, when responding, speak as you would while announcing a sports event.",
    "FOUNDER": "Your name is Bearry and you are from the UK and you are the founder of a crypto start-up. Speak as you would during an investor meeting.",
}

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def generate_system_prompt(default_sys_prompt):
    if os.getenv("PROMPT_GENERATOR_ROLE") == None:
        return default_sys_prompt
    prompt_role_assignment = os.getenv("PROMPT_GENERATOR_ROLE").upper()
    if prompt_role_assignment == "RANDOM":
        prompt_role_assignment = random.choice(list(PROMPT_ROLES.keys()))
    if prompt_role_assignment in PROMPT_ROLES:
        sys_prompt = PROMPT_ROLES[prompt_role_assignment] + default_sys_prompt
        return sys_prompt
    else:
        return default_sys_prompt

def stage2_generator(values):
    for val in values:
        output = {}
        for field in val:
            if field not in ["agent_answers"]:
                output[field] = val[field]
            else:
                for subfield in val[field]:
                    output[f"{field}_{subfield}"] = val[field][subfield]
        yield output

def stage3_generator(values):
    for val in values:
        output = {}
        for field in val:
            if field not in {"agent_answers", "agent_opinion"}:
                output[field] = val[field]
            else:
                for subfield in val[field]:
                    output[f"{field}_{subfield}"] = val[field][subfield]
        yield output

def sorted_agent_ids(cols, prefix):
    agent_ids = []
    for c in cols:
        if c.startswith(prefix):
            agent_ids.append(c[len(prefix) :])
    agent_ids.sort(reverse=False)
    return agent_ids

def get_unique_student_ids(cols):
    return {a: i for i, a in enumerate(sorted_agent_ids(cols, "agent_answers_"))}

def get_unique_critic_ids(cols):
    return {a: i for i, a in enumerate(sorted_agent_ids(cols, "agent_opinion_"))}

def pick_k_cols(cols, datum, current_stage, default_k=5, method="top_k"):
    if current_stage == 2:
        prefix = "agent_answers"
    elif current_stage == 3:
        prefix = "agent_opinion"
    valid_cols = [c for c in cols if c.startswith(prefix)]
    k = min(default_k, len(valid_cols))
    if method == "uniform_random":
        subsampled_cols = random.sample(valid_cols, k)
    elif method == "top_k":
        question, completions, answer = (
            [[{"content": datum["question"]}]],
            [[{"content": datum[c]}] for c in valid_cols],
            [datum["answer"] for _ in valid_cols],
        )
        if current_stage == 2:
            total_rewards = stage1_rewards.top_k_cumulative_reward(
                question, completions, answer
            )
        elif current_stage == 3:
            total_rewards = stage2_rewards.top_k_cumulative_reward(
                question, completions, answer
            )
        reward_per_col = {c: {} for c in valid_cols}
        for idx, c in enumerate(valid_cols):
            hash_fxn = hashlib.md5()
            hash_fxn.update(str.encode(c))
            reward_per_col[c]["tiebreaker"] = int(hash_fxn.hexdigest(), 16)
            reward_per_col[c]["reward"] = total_rewards[idx]
        to_sort = [
            (reward_per_col[c]["reward"], reward_per_col[c]["tiebreaker"], c)
            for c in reward_per_col
        ]
        to_sort.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, valid_cols = zip(*to_sort)
        subsampled_cols = valid_cols[-k:]
    return subsampled_cols

def generate_stage2_user_prompt(datum, cols):
    sp = []
    sp.append(f"The question we were given is: {datum['question']}" + "  \n\n")
    sp.append("The following answers to this question were suggested:" + " \n")
    subsampled_cols = pick_k_cols(cols, datum, 2)
    agentID_to_studentID = get_unique_student_ids(subsampled_cols)
    for agentID in agentID_to_studentID:
        feature = f"agent_answers_{agentID}"
        if feature in datum:
            sp.append(
                f"<student>Student #{agentID_to_studentID[agentID]}</student> said \n"
            )
            sp.append(datum[feature])
            sp.append("\n\n\n")
    return "".join(sp)

def generate_stage3_user_prompt(datum, cols):
    sp = []
    sp.append(f"{datum['stage2_prompt']}" + "  \n")
    sp.append(
        "After comparing these answers, the following feedback was given about which answer is best:"
        + " \n"
    )
    subsampled_cols = pick_k_cols(cols, datum, 3)
    agentID_to_criticID = get_unique_critic_ids(subsampled_cols)
    for agentID in agentID_to_criticID:
        feature = f"agent_opinion_{agentID}"
        if feature in datum:
            sp.append(
                f"<criticism>Criticism #{agentID_to_criticID[agentID]}</criticism> was \n"
            )
            sp.append(datum[feature])
            sp.append("\n\n\n")
    return "".join(sp)

def get_gsm8k_questions(data) -> Dataset:
    sys_prompt = generate_system_prompt(STAGE1_SYSTEM_PROMPT)
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data

def get_gsm8k_questions_with_stage1_answers(data) -> Dataset:
    sys_prompt = generate_system_prompt(STAGE2_SYSTEM_PROMPT)
    cols = data.column_names
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": generate_stage2_user_prompt(x, cols)},
            ],
            "answer": x["answer"],
        }
    )
    return data

def get_gsm8k_questions_with_stage1and2_answers(data) -> Dataset:
    sys_prompt = generate_system_prompt(STAGE3_SYSTEM_PROMPT)
    cols = data.column_names
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": generate_stage3_user_prompt(x, cols)},
            ],
            "answer": x["answer"],
        }
    )
    return data

def fill_unknown_answers_opinions(values):
    FILLED_FIELDS = ("agent_answers", "agent_opinion")
    agent_set = set()
    for val in values:
        for field in val:
            if field in FILLED_FIELDS:
                agent_set |= val[field].keys()
    for val in values:
        for field in val:
            if field in FILLED_FIELDS:
                diff_keys = agent_set - val[field].keys()
                for agent in diff_keys:
                    val[field].update({agent: "No answer received..."})

def get_stage1_samples():
    dataset_id = "openai/gsm8k"
    train_dataset = load_dataset(dataset_id, "main")["train"]
    test_dataset = load_dataset(dataset_id, "main")["test"]
    train_dataset = get_gsm8k_questions(train_dataset)
    test_dataset = get_gsm8k_questions(test_dataset)
    return train_dataset, test_dataset

def get_stage2_samples(values, test_size=0.1):
    fill_unknown_answers_opinions(values)
    dataset = Dataset.from_generator(stage2_generator, gen_kwargs={"values": values})
    dataset = get_gsm8k_questions_with_stage1_answers(dataset)
    return dataset, dataset

def get_stage3_samples(values, test_size=0.1):
    fill_unknown_answers_opinions(values)
    dataset = Dataset.from_generator(stage3_generator, gen_kwargs={"values": values})
    dataset = get_gsm8k_questions_with_stage1and2_answers(dataset)
    return dataset, dataset
