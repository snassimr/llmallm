
import pandas as pd
from llama_index import Response

def display_agent_history(files, questions_set, query_engine, document_agents):

    from IPython.display import display

    agent_history = []
    for file in files:
        for q in questions_set[file]:
            agent = document_agents[file]
            response = query_engine.query(q)
            agent_history.extend(
                [
                    {
                        'file' : file,
                        'question' : q,
                        'role': msg.role,
                        'content': msg.content,
                        'additional_kwargs': msg.additional_kwargs
                    } 
                    for msg in agent.chat_history
                ])
    
    agent_history = pd.DataFrame(agent_history)

    def style_content(val):
        """Style for the content column."""
        style = 'white-space: normal; word-wrap: break-word; border: 1px solid black; padding: 5px; text-align: left;'
        return style

    styled_df = agent_history.style.applymap(style_content, subset=['content'])
    
    display(styled_df)

    return agent_history

def prepare_output(files, questions_set, document_agents, query_engine):

    import pandas as pd

    qa_dataset = []
    agent_history = []

    for file in files:
        for q in questions_set[file]:

            # qa_dataset
            response = query_engine.query(q)
            answer   = response.response
            qa_dataset.append((file, q, answer))

            # agent_history
            agent = document_agents[file]
            agent_history.extend(
                [
                    {
                        'file' : file,
                        'question' : q,
                        'role': msg.role,
                        'content': msg.content,
                        'additional_kwargs': msg.additional_kwargs
                    } 
                    for msg in agent.chat_history
                ])
  
    qa_dataset = pd.DataFrame(qa_dataset, columns = ['file', 'question', 'answer'])
    agent_history = pd.DataFrame(agent_history)
 
    return qa_dataset, agent_history

# def prepare_output(files, questions_set, query_engine):

#     import pandas as pd

#     qa_dataset = []

#     for file in files:
#         for q in questions_set[file]:

#             # qa_dataset
#             response = query_engine.query(q)
#             answer   = response.response
#             qa_dataset.append((file, q, answer))

#     qa_dataset = pd.DataFrame(qa_dataset, columns = ['file', 'question', 'answer'])
 
#     return qa_dataset

def prepare_extracts(files, questions_set, vector_query_engines):

    import pandas as pd

    extracts = []

    for file in files:
        for q in questions_set[file]:
            vector_query_engine = vector_query_engines[file]
            vector_tool_response = vector_query_engine.query(q)
            vector_tool_sources  = vector_tool_response.source_nodes
            # vector_tool_files = [i.metadata['file_name'] for i in vector_tool_sources]
            # vector_tool_pages = [i.metadata['page_label'] for i in vector_tool_sources]

            extracts.extend([
                {
                        'file' : file,
                        'question' : q,
                        #   'file_name': vector_tool_files[i-1], 
                        #   'page_label': vector_tool_pages[i-1],
                        'text': i.node.get_content(),
                        'score': i.score,
                } 
                for i in vector_tool_sources])
        
    extracts = pd.DataFrame(extracts)

    return extracts
    
def display_output(df, subset):

    from IPython.display import display

    def style_content(val):
        """Style for the content column."""
        style = 'white-space: normal; word-wrap: break-word; border: 1px solid black; padding: 5px; text-align: left;'
        return style

    styled_df = df.style.map(style_content, subset = subset)
    
    display(styled_df)

def display_strings(strings_list):

    from IPython.display import display, HTML

    style = """
            <style>
            .my-string-style {
                white-space: normal;
                word-wrap: break-word;
                border: 1px solid black;
                padding: 5px;
                text-align: left;
                max-width: 500px;  /* You can set a max-width to ensure wrapping */
            }
            </style>
            """
    
    for string in strings_list:
        html_string = f"{style}<div class='my-string-style'>{string}</div>"

    # Display the styled string
    display(HTML(html_string))


def display_eval_df(query: str, response: Response, eval_result: str) -> None:
  
  from IPython.display import display

  eval_df = pd.DataFrame(
      {
          "Query": str(query),
          "Response": str(response),
          "Source": response.source_nodes[0].node.get_content()[:500] + "...",
          "Evaluation Result": eval_result.feedback
      },
      index=[0],
  )
  eval_df = eval_df.style.set_properties(
      **{
          "inline-size": "600px",
          "overflow-wrap": "break-word",
      },
      subset=["Response", "Source"]
  )
  
  display(eval_df)

  return eval_df
