from dotenv import load_dotenv
_ = load_dotenv()

import os
import openai
import gradio as gr


openai.api_key=os.getenv('OPENAI_API_KEY')

def generate_openai_response(prompt):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "user", "content": prompt}
    ]
    )
    message = completions.choices[0].message.content
    return message.strip()

# def run_ui():
    
import random

with gr.Blocks() as demo:
    conversation = gr.Chatbot(label = "Conversation :",
                            elem_id = None,
                            layout = 'panel',
                            show_copy_button = False)
    
    btn_clear_conversation = gr.ClearButton([conversation])
    
    question = gr.Textbox(label = "Question :",
                        elem_id = None,
                        lines = 3, max_lines = 3)
    
    btn_clear_question = gr.ClearButton([question])

    saved = gr.Chatbot(label = "Saved :",
                        elem_id = None,
                        layout = 'panel',
                        show_copy_button = False)
    
    btn_clear_saved = gr.ClearButton([saved])

    def vote(data: gr.LikeData, conversation: gr.Chatbot, saved: gr.Chatbot):
        if data.liked:
            saved.append(conversation[data.index[0]])
            return conversation, saved
        else:
            del conversation[data.index[0]]
            return conversation, saved

    conversation.like(vote, [conversation, saved], [conversation, saved])  
    
    def respond_random(question, conversation):
        response = random.choice(["How are you?", "I love you", "I'm very hungry"])
        conversation.append((question, response))
        return "", conversation
    
    def respond_openai(question, conversation):
        response = generate_openai_response(question)
        conversation.append((question, response))
        return "", conversation
    
    respond = respond_openai

    question.submit(respond, [question, conversation], [question, conversation])

    uploaded_file = gr.File(file_types = [".pdf"])

    with gr.Row():
        btn_process_file = gr.Button("Process File")
        txtb_file_path   = gr.Textbox(visible = False)

    def upload_file(file):
        file_name = file.name
        return file_name

    def get_file(uploaded_file):
        import os
        temp_location = os.path.join("/tmp", uploaded_file.name)
        with open(temp_location, "wb") as f:
            f.write(uploaded_file.read())
        return temp_location

    def process_file(file_path):
        btn_process_file.interactive = False
        print(file_path)
        btn_process_file.interactive = True

    txtb_file_path.change(process_file, txtb_file_path)

    btn_process_file.click(get_file, inputs = uploaded_file , outputs = txtb_file_path)
    


demo.launch(
    show_api=False,
    # share = True
)
