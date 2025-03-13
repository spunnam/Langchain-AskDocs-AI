from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Langchain Doc Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if (

        "chat_answers_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_source_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    source_string = "sources:\n"
    for i, source in enumerate(sources_list):
        source_string += f"{i + 1}. {source}\n"
    return source_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        formatted_response = (
            f"{generated_response['result']}\n\n {create_source_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                              st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)
