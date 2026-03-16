from langchain_core.prompts.prompt import PromptTemplate


def get_prompt():
    return PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. Create a table that lists course schedule.
    The table should have three columns: name, time, teacher. Please align the entries to the left.
    Remember to respond with a table, and NOTHING else.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: 
    {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
