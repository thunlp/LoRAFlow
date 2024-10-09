import json
import os
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    results_file = sample_file.replace("reform.jsonl", f"results.txt")
    print(f"Writing results to {results_file}")
    with open(results_file, "w") as f:
        f.write(str(results))
    
def reformat(data, input_file):

    new_data = []
    for instance in data:
        new_instance = {}
        new_instance["task_id"] = instance["data"]["task_id"]
        new_instance["completion"] = instance["processed_outputs"][0].strip()
        new_instance["all_code"] = instance["raw_outputs"][0]
        new_data.append(new_instance)

    output_dir = os.path.dirname(input_file)
    output_file = os.path.join(output_dir, "reform.jsonl")
    with open(output_file, "w") as f:
        for instance in new_data:
            f.write(json.dumps(instance, ensure_ascii=False) + "\n")
    
    return output_file

def transform_humaneval(data, input_file):
    reformat_file = reformat(data, input_file)
    if os.path.exists(reformat_file):
        entry_point(reformat_file)
        
        with open(os.path.join(reformat_file.replace("reform.jsonl", ""), "results.txt"), "r") as f:
            result = eval(f.read())['pass@1']
    return result

if __name__ == "__main__":
    pass
    