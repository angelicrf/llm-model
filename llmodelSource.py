from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import ast

def llmchain_invoke(question):

    llm = CTransformers(model="Models/pytorch_model.bin", model_type='llama')
    template = """Question: {question}

    Answer:"""
    prompt_template = PromptTemplate(template=template, input_variables=["question"])
    
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    input_data = {"question": question}
    output = llm_chain.invoke(input_data)
    
    return output


def get_llma_version():
    try:
        lama_version = pkg_resources.get_distribution("lama").version
        print("LAMA version:", lama_version)
        return lama_version
    except pkg_resources.DistributionNotFound:
        print("LAMA is not installed.")
        return "Not Valid"

def create_input_field():
    input_value = st.text_input("Enter your question:")
    return input_value

def validate_input(input_value):
    if input_value:
        return input_value
    else:
        return None

def generate_text_input():
    input_value = create_input_field()
    validated_value = validate_input(input_value)
    if validated_value != None:
        return validated_value
    else:
        return None 

def get_response(thisInput):
        question = {thisInput}
        if question:
            output = llmchain_invoke(question)
            print(output)
            text_value = output["text"]
            parsed_text = ast.literal_eval(text_value)
            output_text = parsed_text.pop() if isinstance(parsed_text, set) else parsed_text
            print(output_text)
            st.write("The validated input value is:", output_text)       
        else:
            print('No Data')
 

def main():
    getvalue = generate_text_input()
    if getvalue:
        get_response(getvalue)
    else: return None   
   



if __name__ == "__main__":
    main()


