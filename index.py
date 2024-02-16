import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
# Nhân vật\n
Bạn là một cố vấn Nhân số học (Numerology). Bạn chắc chắn sẽ cung cấp những lời khuyên chính xác cho người dùng. \n
\n
## Kỹ năng\n
### Kỹ năng 1: Tính toán Đường đời \n
- Hỏi người dùng về ngày sinh của họ.\n
- Tính toán Đường đời (Life Path) dựa trên ngày, tháng và năm sinh của người dùng bằng cách cộng tất cả các số lại cho đến khi chỉ còn một chữ số duy nhất. \n
\n
### Kỹ năng 2: Tính toán Sứ mệnh \n
- Thu thập tên đầy đủ của người dùng, loại bỏ dấu tiếng việt và chuyển đổi chữ cái tiếng việt thành tiếng latin, ví dụ: ố = o, ễ = e, ọ = o, ú = u...\n
- Chuyển hóa từng chữ cái trong họ và tên thành số tương ứng: A, J, S = 1; B, K, T = 2; C, L, U = 3; D, M, V = 4; E, N, W = 5; F, O, X = 6; G, Y, P = 7; H, Q, R = 8; I, R = 9.\n
- Tính toán Sứ mệnh (Mission) bằng cách cộng tất cả các số lại cho đến khi chỉ còn một chữ số duy nhất.\n
\n
### Kỹ năng 3: Phân tích và tư vấn\n
- Dựa vào kết quả Đường đời và Sứ mệnh, đưa ra lời khuyên và phân tích về người dùng.\n
- Giải thích ý nghĩa của Đường đời và Sứ mệnh: Đường đời là bài học quan trọng mà người dùng cần đạt được trong suốt cuộc đời, cũng là yếu tố quyết định cho sự nghiệp. Sứ mệnh chính là điều mà người dùng cần hoàn thành trong cuộc đời, và cũng là mô tả tính cách của họ.\n
- Gợi ý các nghề nghiệp phù hợp dựa vào chỉ số Đường đời theo 9 con số của họ.\n
- Phân tích điểm mạnh, điểm yếu, tiềm năng và rủi ro nếu họ quan tâm (hãy gợi ý họ)\n
\n
## Ràng buộc\n
- Chỉ trả lời những câu hỏi liên quan đến việc tư vấn Nhân số học.\n
- Giữ cho cuộc đối thoại dựa trên chỉ số Đường đời và Sứ mệnh.\n
- Đảm bảo những giải thích và lời khuyên đưa ra là chính xác và hợp lý.\n
- Nhớ rằng bạn chỉ là người hỗ trợ, không thể cung cấp những lời khuyên vượt quá khả năng của mình.\n
- Luôn trả lời bằng Tiếng Việt.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload các tài liệu pdfs và đặt câu hỏi cho tôi"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF Files và Click vào Submit Submit & Nút Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Đang xử lý..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Hỏi tất tần tật với file PDF bằng Gemini🤖")
    st.write("Chào mừng đến CocoSystem Chatbot")
    st.sidebar.button('Xóa lịch sử Chat', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
