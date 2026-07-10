import streamlit as st
import requests
from chatbot_ui.core.config import config

def api_call(method, url, **kwargs):
    def _show_error_popup(message):
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }
    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response formar from server"}
        
        if response.ok:
            return True, response_data
        
        return False, response_data
    
    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your internet connection and try again.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("Request timed out. Please try again.")
        return False, {"message": "Request timed out"}
    except Exception as e:
        _show_error_popup(f"An error occurred: {str(e)}")
        return False, {"message": str(e)}


if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant", "content":"Hello, how can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        output = api_call("post", f"{config.API_URL}/rag", json={"query": prompt})
        response_data = output[1]
        answer = response_data["answer"]
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})