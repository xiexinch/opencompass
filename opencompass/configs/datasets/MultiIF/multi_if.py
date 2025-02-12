from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BaseRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import MultiIFDataset, MultiIFEvaluator

multi_if_reader_cfg = dict(input_columns=["dialogue"], output_column="references")


multi_if_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{prompt}"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=ChatInferencer,
        max_seq_len=4096,
        max_out_len=1024,
        temperature=0,
        do_sample=False,
        infer_mode="every",
    ),
)

multi_if_eval_cfg = dict(evaluator=dict(type=MultiIFEvaluator))

multi_if_datasets = [
    dict(
        abbr="MultiIF",
        type=MultiIFDataset,
        path="opencompass/multi_if",
        reader_cfg=multi_if_reader_cfg,
        infer_cfg=multi_if_infer_cfg,
        eval_cfg=multi_if_eval_cfg,
    )
]
