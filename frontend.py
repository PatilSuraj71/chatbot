import os
import time
import uuid
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver


st.set_page_config(
    page_title="Astra • AI Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)


load_dotenv()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@st.cache_resource(show_spinner=False)
def build_chatbot():

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("groq_key"),
        temperature=0.7,
    )

    def prompt_message(state: ChatState):
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        messages = state["messages"]
        response = model.invoke([system_message] + messages)
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("prompt", prompt_message)
    graph.add_edge(START, "prompt")
    graph.add_edge("prompt", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


chatbot = build_chatbot()


def stream_chat(user_message: str, thread_id: str):
    """Yield Astra's reply token-by-token."""
    config = {"configurable": {"thread_id": thread_id}}
    for chunk, _metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
        stream_mode="messages",
    ):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content



st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background: radial-gradient(circle at 20% 0%, #1b1033 0%, #0d0c1a 45%, #08070f 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #150f26 0%, #0d0c1a 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .astra-hero {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 18px 22px;
            margin-bottom: 18px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(59,130,246,0.10));
            border: 1px solid rgba(255,255,255,0.08);
        }
        .astra-hero .logo {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            background: linear-gradient(135deg, #8b5cf6, #3b82f6);
            box-shadow: 0 6px 18px rgba(139,92,246,0.45);
        }
        .astra-hero h1 {
            font-size: 20px;
            font-weight: 700;
            margin: 0;
            color: #f4f2ff;
        }
        .astra-hero p {
            margin: 0;
            font-size: 13px;
            color: #a9a4c2;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #34d399;
            display: inline-block;
            margin-right: 6px;
            box-shadow: 0 0 8px #34d399;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 4px 2px;
            margin-bottom: 4px;
            background: transparent;
        }

        div[data-testid="stChatMessageContent"] {
            font-size: 15px;
            line-height: 1.55;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            background: linear-gradient(135deg, rgba(139,92,246,0.16), rgba(139,92,246,0.05));
            border: 1px solid rgba(139,92,246,0.25);
            border-radius: 16px 16px 4px 16px;
            padding: 10px 14px;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px 16px 16px 4px;
            padding: 10px 14px;
        }

        div[data-testid="stChatInput"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
        }

        div[data-testid="stChatInput"] textarea {
            color: #f4f2ff !important;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            color: #f4f2ff;
            font-weight: 500;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            border-color: #8b5cf6;
            background: rgba(139,92,246,0.15);
            color: #ffffff;
        }

        .thread-pill {
            font-size: 12px;
            color: #a9a4c2;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 6px 10px;
            margin-top: 6px;
            word-break: break-all;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.markdown("### ✨ Astra")
    st.caption("Your AI assistant, made by Suraj Patil")

    st.markdown("---")

    if st.button("🆕  New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Model**")
    st.caption("llama-3.3-70b-versatile via Groq")

    st.markdown("**Thread**")
    st.markdown(f"<div class='thread-pill'>🧵 {st.session_state.thread_id}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"💬 {len(st.session_state.chat_history)} messages in this session")

st.markdown(
    """
    <div class="astra-hero">
        <div class="logo">✨</div>
        <div>
            <h1>Astra</h1>
            <p><span class="status-dot"></span>Online · Fast, helpful, honest answers</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.chat_history:
    st.markdown("#### Ask me anything ✨")
    cols = st.columns(2)

for msg in st.session_state.chat_history:
    avatar = "🧑" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


user_input = st.chat_input("Message Astra...")

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="✨"):
        placeholder = st.empty()
        full_reply = ""
        try:
            for token in stream_chat(user_input, st.session_state.thread_id):
                full_reply += token
                placeholder.markdown(full_reply + "▌")
                time.sleep(0.005)
            placeholder.markdown(full_reply)
        except Exception as e:
            full_reply = f"⚠️ Something went wrong: `{e}`"
            placeholder.markdown(full_reply)

    st.session_state.chat_history.append({"role": "assistant", "content": full_reply})