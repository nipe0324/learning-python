import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage, # システムメッセージ
    HumanMessage, # ユーザーの質問
    AIMessage # ChatGPTの返答
)
from langchain.callbacks import get_openai_callback

def init_page():
    # ページ設定とヘッダー
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤖",
    )

    st.header("My Great ChatGPT 🤖")
    st.sidebar.title("オプション")

def init_messages():
    clear_button = st.sidebar.button("履歴をクリア", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            # SystemMessageは、ChatGTPの設定を決める
            SystemMessage(content="You are a helpful assistant."),
        ]
        st.session_state.costs = []
        st.session_state.tokens = []

def select_model():
    model = st.sidebar.radio("Select model", ["GPT-3.5", "GPT-4"])
    model_name = { "GPT-3.5": "gpt-3.5-turbo", "GPT-4": "gpt-4" }[model] or "gpt-3.5-turbo"

    # temperatureは0に近いほど予測可能で一貫性がでてくる。大きいほどランダム性が高くなる
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    return ChatOpenAI(model_name=model_name, temperature=temperature)

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost, cb.total_tokens

def main():
    init_page()
    init_messages()

    llm = select_model()

    # ユーザー入力
    if user_input := st.chat_input("聞きたいことを入力してね!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("GPTが考え中..."):
            answer, cost, tokens = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)
        st.session_state.tokens.append(tokens)

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    # コストの表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## コスト")
    st.sidebar.markdown(f"**合計: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

    # トークンの表示
    tokens = st.session_state.get('tokens', [])
    st.sidebar.markdown("## トークン")
    st.sidebar.markdown(f"**合計: {sum(tokens)}**")
    for token in tokens:
        st.sidebar.markdown(f"- {token}\n- ${token}")

if __name__ == "__main__":
    main()
