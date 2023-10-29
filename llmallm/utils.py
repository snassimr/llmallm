
import pandas as pd
from llama_index import Response

def display_agent_history(agent):

    from IPython.display import display

    data = [{'role': msg.role, 
            'content': msg.content, 
            'additional_kwargs': msg.additional_kwargs} for msg in agent.chat_history]
    # Create DataFrame
    agent_history = pd.DataFrame(data)
    # Style

    def style_content(val):
        """Style for the content column."""
        style = 'white-space: normal; word-wrap: break-word; border: 1px solid black; padding: 5px; text-align: left;'
        return style

    styled_df = agent_history.style.applymap(style_content, subset=['content'])
    
    display(styled_df)

    return agent_history

def display_extracts(vector_tool_response):

    from IPython.display import display
    
    vector_tool_sources  = vector_tool_response.source_nodes
    # vector_tool_files = [i.metadata['file_name'] for i in vector_tool_sources]
    # vector_tool_pages = [i.metadata['page_label'] for i in vector_tool_sources]
    vector_tool_texts = [i.node.get_content() for i in vector_tool_sources]
    vector_tool_scores = [i.score for i in vector_tool_sources]

    vector_tool_data = [{
                #   'file_name': vector_tool_files[i-1], 
                #   'page_label': vector_tool_pages[i-1],
                'text': i.node.get_content(),
                'score': i.score,
                } for i in vector_tool_sources]
        
    vector_tool_df = pd.DataFrame(vector_tool_data)

    def style_content(val):
        """Style for the content column."""
        style = 'white-space: normal; word-wrap: break-word; border: 1px solid black; padding: 5px; text-align: left;'
        return style

    styled_df = vector_tool_df.style.applymap(style_content, subset=['text'])
    
    display(styled_df)

    return vector_tool_df
    



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
