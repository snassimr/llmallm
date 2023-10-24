
import pandas as pd
from llama_index import Response

# Define jupyter display function
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
