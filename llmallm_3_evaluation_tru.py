
import pandas as pd
from trulens_eval import Tru

tru = Tru()

from evaluation_utils import get_prebuilt_trulens_recorder

SYS_EVAL_ID = "modeling_sentence_window"
SYS_EVAL_ID = "modeling_auto_merging"

q_dataset = {
    'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf':
    [
        ### Out of context question (overlap with common domain)
        # "How to prepare pizza ?",
        ### Keyword around question
        "What is the purpose of <Figure 2> ?",
        # "List citations where Ethan Perez is one of co-authors",
        ### Global summarization question
        # "Summarize Llama 2 - Open Foundation and Fine-Tuned Chat Models in 500 words",
        ### Typical research paper question
        # "What is the purpose of Red Teaming?",
        # "Prepare table of content ?",
        # "Does paper contain LLama2 comparision to other algorithms ?",
        # "How Llama 2 compared to other open source models in Table 6",
        # "How Llama 2 compared to other open source models in Table 3",
        ### Multi-document questions
        # "If RAG and LLM Fine tuning can be combined some way ?"
    ]
}


def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

tru_recorder = get_prebuilt_trulens_recorder(
    query_engine,
    app_id=SYS_EVAL_ID
)

eval_questions = q_dataset['Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf']
run_evals(eval_questions, tru_recorder=tru_recorder, query_engine=query_engine)

records, feedback = tru.get_records_and_feedback(app_ids=[])

pd.set_option("display.max_colwidth", None)
eval_dataset = records[["input", "output"] + feedback]

# tru.reset_database()
tru.run_dashboard()

