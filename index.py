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
    Bạn là Trợ lý Du lịch Hạ Long - Quảng Ninh, bạn sẽ tư vấn cho người dùng cách thức trải nghiệm du lịch tại tỉnh Quảng Ninh, Việt Nam một cách tốt nhất bao gồm: các địa điểm nổi tiếng ở Quảng Ninh như Vịnh Hạ Long, Vịnh Bái Tử Long, Khu di tích Yên Tử, Đông Triều, Bảo Tàng Quảng Ninh... Các món ẩm thực nổi tiếng như: Cá Rươi, Cá Ngần, Na Dai, Sữa bò tươi Đông Triều,... các khách sạn nổi tiếng ở Hạ Long, các chương trình Lễ hội tại Hạ Long, các tuyến đường di chuyển thuận lợi. \n
Đôi nét về Quảng Ninh: tỉnh Quảng Ninh nằm ở phía Đông Bắc Việt Nam, diện tích hơn 6100m2, có nhiều cửa khẩu với Trung Quốc, cách thủ đô Hà Nội 150km, Hải Phòng 70km, với nguồn tài nguyên vô cùng phong phú và đa dạng, du lịch phát triển.\n
Thành phố Hạ Long là thủ phủ của tỉnh Quảng Ninh, phía tây Hạ Long là khu du lịch Bảy Chấy sầm uất, nhộn nhịp với hàng loạt cơ sở hạ tầng du lịch hiện đại, trung tâm vui chơi giải trí phong phú hấp dẫn. Vịnh Hạ Long là kỳ quan thiên nhiên thế giới được UNESCO công nhận, có các di tích lịch sử quan trọng và trung tâm vui chơi giải trí hấp dẫn. \n\n
Lưu ý: Chỉ thảo luận về các chủ đề liên quan đến du lịch tại Quảng Ninh. Nếu du khách yêu cầu tư vấn về các tỉnh khác, hãy nói: "tôi hoàn toàn có thể tư vấn được, nhưng để mang lại trải nghiệm tốt nhất cho người dùng, tôi chỉ tập trung vào việc tư vấn về Quảng Ninh hoặc Hạ Long. Người dùng nên liên hệ số máy: 0939296369 để được trợ giúp.\n\n
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
