import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage, # システムメッセージ
    HumanMessage, # ユーザーの質問
    AIMessage # ChatGPTの返答
)

def main():
    # temperatureは0から1までの範囲で設定できる
    # 0に近いほど予測可能で一貫性がでてくる。1に近いほどランダム性が高くなる
    llm = ChatOpenAI(temperature=0.7)

    # ページ設定とヘッダー
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤖",
    )
    st.header("My Great ChatGPT 🤖")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = [
            # SystemMessageは、ChatGTPの設定を決める
            SystemMessage(content="You are a helpful assistant."),
        ]

    if user_input := st.chat_input("聞きたいことを入力してね!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("GPTが考え中..."):
            response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

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

if __name__ == "__main__":
    main()
