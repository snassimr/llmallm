from llama_index import Document

def create_figures_query_engine(document : Document, modeling_dir : str):
    
    import os
    import re
    from llama_index.text_splitter import SentenceSplitter

    s_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
    nodes = s_splitter.get_nodes_from_documents(document)

    figure_pattern = 'Figure \d{1,3}'
    nodes_figures = [i for i in nodes if re.search(figure_pattern, i.to_dict()['text'])]


    def get_figures_list(text):
        llm_figures_list_template = """
                        The text below contains mention for some figures.
                        Output figures and their numbers as Python list.
                        
                        EXAMPLES
                        -------- 
                        text : "Figure 2 and no other figures" , Answer : ["Figure 2"]
                        text : "Figure 2 and Figure 10 are present" , Answer : ["Figure 2", "Figure 10"]
                        test " "Figures are part of paper" , Answer : []
                        test " "Table 2 is present and Figure 2 also" , Answer : ["Figure 2"]
                        test " "Table 2 is present" , Answer : []
                        
                        TEXT
                        --------
                        {text}

                        ANSWER
                        --------
                        """
        llm_figures_list_prompt = llm_figures_list_template.replace("{text}", text)
        figures_list = llm.complete(llm_figures_list_prompt).text

        return figures_list

    import json
    from llama_index.llms import OpenAI
    llm = OpenAI(model = 'gpt-4', temperature = 0.0)
    # llm.additional_kwargs = {"top_k": 1}

    for n in nodes_figures:
        figures_list = get_figures_list(n.to_dict()['text'])
        figures_str = " and ".join(json.loads(figures_list))
        print(figures_list)
        n.metadata = {'figures_content' : f"Contains information about {figures_str}"}

    from llama_index.prompts.prompt_type import PromptType
    from llama_index import Prompt

    # Build vector index
    from llama_index import StorageContext
    from llama_index import VectorStoreIndex
    from llama_index import load_index_from_storage

    vector_index_save_dir=os.path.join(modeling_dir, "indexes/figures_index")

    if not os.path.exists(vector_index_save_dir):
        vector_index = VectorStoreIndex(nodes_figures)
        vector_index.storage_context.persist(persist_dir=vector_index_save_dir)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vector_index_save_dir),
        )

    # Define query engines
    from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
    from llama_index import Prompt

    SYS_QA_TEMPLATE = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)

    vector_query_engine = vector_index.as_query_engine(
        text_qa_template = SYS_QA_TEMPLATE,
        similarity_top_k=10
    )

    return vector_query_engine
