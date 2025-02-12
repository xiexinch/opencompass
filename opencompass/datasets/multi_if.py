import json

import pandas as pd
from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset
from .IFEval.ifeval import (
    IFEvaluator,
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)


@LOAD_DATASET.register_module()
class MultiIFDataset(BaseDataset):

    def __init__(
        self,
        language_list=[
            "all_languages",
            "German",
            "Italian",
            "Vietnamese",
            "Spanish",
            "Hindi",
            "Portuguese",
            "English",
            "French",
            "Thai",
            "Chinese",
            "Russian",
        ],
        **kwargs,
    ):
        self.language_list = language_list
        super().__init__(**kwargs)

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=False)
        input_df = pd.read_csv(path, keep_default_na=False)
        datasets = []
        prompts_keys = [
            "turn_1_prompt",
            "turn_2_prompt",
            "turn_3_prompt",
        ]
        instruction_id_keys = [
            "turn_1_instruction_id_list",
            "turn_2_instruction_id_list",
            "turn_3_instruction_id_list",
        ]
        kwargs_keys = [
            "turn_1_kwargs",
            "turn_2_kwargs",
            "turn_3_kwargs",
        ]
        temp_count = 0
        for index, row in input_df.iterrows():
            temp_count += 1
            prompts = [
                json.loads(row[key])["content"]
                for key in prompts_keys
                if row[key] != None and row[key] != ""
            ]
            dialogues = []
            for prompt in prompts:
                dialogues.append({"role": "user", "content": prompt})
                dialogues.append({"role": "assistant", "content": ""})

            instruction_id_list = [
                json.loads(row[key])
                for key in instruction_id_keys
                if row[key] != None and row[key] != ""
            ]
            str_kwargs_list = [
                json.loads(row[key])
                for key in kwargs_keys
                if row[key] != None and row[key] != ""
            ]

            kwargs_list = []
            for kwargs in str_kwargs_list:
                kwargs_ = []
                for kwarg in kwargs:
                    kwargs_.append(json.loads(kwarg))
                kwargs_list.append(kwargs_)

            datasets.append(
                {
                    "dialogue": dialogues,
                    "references": {
                        "instruction_id_list": instruction_id_list,
                        "kwargs_list": kwargs_list,
                        "language": row["language"],
                        "key": row["key"],
                    },
                },
            )

        return Dataset.from_list(datasets)


class MultiIFEvaluator(IFEvaluator):

    def score(self, predictions, references, origin_prompt):

        prompt_strict_correct, prompt_strict_total = [0, 0, 0], [0, 0, 0]
        inst_strict_correct, inst_strict_total = [0, 0, 0], [0, 0, 0]
        prompt_loose_correct, prompt_loose_total = [0, 0, 0], [0, 0, 0]
        inst_loose_correct, inst_loose_total = [0, 0, 0], [0, 0, 0]
        details = {
            "turn_1": {},
            "turn_2": {},
            "turn_3": {},
        }

        for index, (preds, refers, prompts) in enumerate(
            zip(predictions, references, origin_prompt)
        ):
            for turn, (pred, prompt) in enumerate(zip(preds, prompts)):

                input = InputExample(
                    key=refers["key"],
                    instruction_id_list=refers["instruction_id_list"][turn],
                    prompt=prompt,
                    kwargs=refers["kwargs_list"][turn],
                )

                for kwarg in input.kwargs:
                    for k in list(kwarg.keys()):
                        if kwarg[k] is None:
                            kwarg.pop(k, None)

                # strict
                example = test_instruction_following_strict(input, pred)
                follow_instruction_list = example.follow_instruction_list
                instruction_id_list = example.instruction_id_list
                prompt_strict_total[turn] += 1
                is_strict_correct = all(follow_instruction_list)
                prompt_strict_correct[turn] += is_strict_correct
                inst_strict_total[turn] += len(instruction_id_list)
                inst_strict_correct[turn] += sum(follow_instruction_list)

                # loose
                example = test_instruction_following_loose(input, pred)
                follow_instruction_list = example.follow_instruction_list
                instruction_id_list = example.instruction_id_list
                prompt_loose_total[turn] += 1
                is_loose_correct = all(follow_instruction_list)
                prompt_loose_correct[turn] += is_loose_correct
                inst_loose_total[turn] += len(instruction_id_list)
                inst_loose_correct[turn] += sum(follow_instruction_list)

                if is_strict_correct:
                    grade = "strict"
                elif is_loose_correct:
                    grade = "loose"
                else:
                    grade = "none"

                details[f"turn_{turn+1}"][str(index)] = {
                    "prompt": prompt,
                    "pred": pred,
                    "refer": {
                        "instruction_id_list": instruction_id_list,
                        "kwargs": input.kwargs,
                        "key": input.key,
                    },
                    "grade": grade,
                    "is_strict_correct": is_strict_correct,
                    "is_loose_correct": is_loose_correct,
                    "is_correct": is_strict_correct,
                }

        results = {
            "turn_1_average": sum(
                [
                    prompt_strict_correct[0] / prompt_strict_total[0],
                    inst_strict_correct[0] / inst_strict_total[0],
                    prompt_loose_correct[0] / prompt_loose_total[0],
                    inst_loose_correct[0] / inst_loose_total[0],
                ]
            )
            / 4,
            "turn_2_average": sum(
                [
                    prompt_strict_correct[1] / prompt_strict_total[1],
                    inst_strict_correct[1] / inst_strict_total[1],
                    prompt_loose_correct[1] / prompt_loose_total[1],
                    inst_loose_correct[1] / inst_loose_total[1],
                ]
            )
            / 4,
            "turn_3_average": sum(
                [
                    prompt_strict_correct[2] / prompt_strict_total[2],
                    inst_strict_correct[2] / inst_strict_total[2],
                    prompt_loose_correct[2] / prompt_loose_total[2],
                    inst_loose_correct[2] / inst_loose_total[2],
                ]
            )
            / 4,
            "Prompt-level-strict-accuracy_turn_1": prompt_strict_correct[0]
            / prompt_strict_total[0],
            "Prompt-level-strict-accuracy_turn_2": prompt_strict_correct[1]
            / prompt_strict_total[1],
            "Prompt-level-strict-accuracy_turn_3": prompt_strict_correct[2]
            / prompt_strict_total[2],
            "Instruction-level-strict-accuracy_turn_1": inst_strict_correct[0]
            / inst_strict_total[0],
            "Instruction-level-strict-accuracy_turn_2": inst_strict_correct[1]
            / inst_strict_total[1],
            "Instruction-level-strict-accuracy_turn_3": inst_strict_correct[2]
            / inst_strict_total[2],
            "Prompt-level-loose-accuracy_turn_1": prompt_loose_correct[0]
            / prompt_loose_total[0],
            "Prompt-level-loose-accuracy_turn_2": prompt_loose_correct[1]
            / prompt_loose_total[1],
            "Prompt-level-loose-accuracy_turn_3": prompt_loose_correct[2]
            / prompt_loose_total[2],
            "Instruction-level-loose-accuracy_turn_1": inst_loose_correct[0]
            / inst_loose_total[0],
            "Instruction-level-loose-accuracy_turn_2": inst_loose_correct[1]
            / inst_loose_total[1],
            "Instruction-level-loose-accuracy_turn_3": inst_loose_correct[2]
            / inst_loose_total[2],
            "details": details,
        }
        return results
