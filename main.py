import streamlit as st
import sqlite3
import hashlib
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import os
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from pypdf import PdfReader
import tempfile


load_dotenv()
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

gemini = genai.GenerativeModel("gemini-2.0-flash")

DB_NAME = "healthmate.db"
UPLOAD_DIR = "user_uploaded_files"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL                                         
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS files(
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """) 

    print("Database and Tables initialized")  


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def sign_up(first_name, last_name, date_of_birth, email, password):
    with sqlite3.connect(DB_NAME) as conn:
        try:
            conn.execute("""
            INSERT INTO users (first_name, last_name, date_of_birth, email, password)
            VALUES (?, ?, ?, ?, ?)                    
            """, (first_name, last_name, str(date_of_birth), email, hash_password(password)))
            conn.commit()
            return True, "Account created Successfully, Now you can login"
        
        except sqlite3.IntegrityError:
            return False, "This email is already registered, Try logging in."
        

def login(email, password):
    with sqlite3.connect(DB_NAME) as conn:
        user = conn.execute("""
        SELECT user_id, first_name, last_name FROM users WHERE email = ? AND password = ?                            
        """, (email, hash_password(password))).fetchone()
        return user if user else None
    

def save_file(user_id, file_name, file_path):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        INSERT INTO files (user_id, file_name, file_path)
        VALUES (?, ?, ?)
        """, (user_id, file_name, file_path))
        conn.commit()


def get_user_files(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        files = conn.execute("""
        SELECT file_name, file_path FROM files where user_id = ?
        """, (user_id,)).fetchall()
        return files
    

def delete_file(user_id, file_name):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        DELETE FROM files WHERE (user_id, file_name) = (?, ?)
        """, (user_id, file_name)).fetchall()
        conn.commit()

init_db()

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    db = FAISS.from_texts(chunks, embeddings)
    return db

def get_rel_text(user_query, db):
    rel_text = db.similarity_search(user_query, k=1)
    return rel_text[0].page_content if rel_text else "No relevant information found."

def bot_response(model, query, relevant_texts, history):
    context = " ".join(relevant_texts)
    prompt = f"""

    This is the context of the document
    Context : {context}
    And this is the User Query
    User : {query}
    And this is the history of the conversation
    History : {history}

    Please generate a response to the user query based on the context and the history of the conversation.
    The questions might be asked related to the provided context, and may also be in terms of the medical field, diseases, biology, etc.
    Answer the question with respect to the context provided, you can also use your additional knowledge too, but do not ignore the content of the provided medical report.
    Answer the following queries like a professional doctor, having a lot of knowledge on the basis of report context.

    Bot :

    """

    response = model.generate_parts(
        prompt,
        generation_config = genai.GenerationConfig(
            temperature = 0.68
        )
    )

    return response.text


model = genai.GenerativeModel(model_name="gemini-2.0-flash",
    system_instruction = """"

    Your name is "CuraBot" and you are a doctor who gives the medications and let the user know the disease he is suffering from based on the symptoms he provides.

    Your Roles:
    1) you are a healthbot, who is intelligent in finding the particular disease or list of diseases based on the symptoms provided .
    2) you are a doctor, you should let the user know through which he is suffering based on the symptoms provided.
    3) If possible you can also give the medications for that particular symptom which user is encountering
    4) the best and the most important part is that you should tell him what he is suffering from, based on the symptoms provided by him.

    Points to remember:
    1) You should engage with the user like a fellow doctor, and give the user proper reply for his queries
    2) The conentration and the gist of the conversation no need to be completely based on the symptoms or diagnosis, your flow of chat should be more like a human conversation.
    3) If the conversation goes way too out of medicine and healthcare or if the user input is abusive, let the user know that the content is abusive and we cannot tolerate suh inputs.
    4) The important part is that you should not anywhere mention, "You should consult a doctor for furthr diagnosis"

    """)

st.set_page_config(page_title="HealthMate", page_icon="🩺", layout="wide")

if 'messages' not in st.session_state:
    st.session_state.messages = {}

with st.sidebar:
    selected = option_menu(
        "Menu", ["Landing Page", "Login/Signup", "Consultation Bot", "Medical Record Bot"],
        icons = ["house", "person", "chat-dots", "file-medical"],
        menu_icon = "cast", default_index=0
    )

if selected == "Login/Signup":
    st.header("Login/ Signup")

    if "user_id" in st.session_state:
        st.info(f"You are logged in as {st.session_state['first_name']} {st.session_state['last_name']}")
        if st.button("Logout"):
            st.session_state.clear()
            st.success("Logged Out Successfully!")

    else:
        action = st.selectbox("Select an action", ['Login', 'Sign Up'])

        if action == "Sign Up":
            st.subheader("Sign Up")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            dob = st.date_input("Date Of Birth")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')

            if st.button("Sign Up"):
                success, msg = sign_up(first_name, last_name, dob, email, password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        
        elif action == "Login":
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            
            if st.button("Login"):
                user = login(email, password)
                if user:
                    st.session_state['user_id'], st.session_state['first_name'], st.session_state['last_name'] = user
                    st.success(f"Logged in as: {user[1]} {user[2]}!")
                    st.session_state.messages[st.session_state['user_id']] = []

                else:
                    st.error("Invalid Email or Password")


if selected == "Consultation Bot":
    st.subheader("Chat with CuraBot")
    if 'user_id' not in st.session_state:
        st.warning("Please login to access the CuraBot")

    else:
        st.info(f"Welcome {st.session_state['first_name']} !!")
        st.write("Describe your symptoms and get insights on possible conditions and treatments.")
        
        chat_history = st.session_state.messages.get(st.session_state['user_id'], [])

        chat_bot = model.start_chat(
            history = chat_history
        )

        for message in chat_history:
            # row = st.columns(2)
            if message['role'] == 'user':
                st.chat_message(message['role']).markdown(message['parts'])
            else:
                st.chat_message(message['role']).markdown(message['parts'])

        user_question = st.chat_input("Type your message here: ")

        if user_question:
            st.chat_message("user").markdown(user_question)
            chat_history.append(
                {'role': 'user',
                 'parts' : user_question}
            )

            with st.spinner("Thinking..."):
                response = chat_bot.send_message(user_question)

                st.chat_message("assistant").markdown(response.text)

                chat_history.append(
                    {
                        'role':'assistant',
                        'parts':response.text
                    }
                )

            st.session_state.messages[st.session_state['user_id']] = chat_history

               
# BUILD THE CHAT WITH MEDICAL REPORT BOT (RAG System)
if selected == "Medical Record Bot":
    st.subheader("Medical Record Reader")
    if 'user_id' not in st.session_state:
        st.warning("Please login to access the Medical Record Bot")
    else:
        st.info(f"Welcome {st.session_state['first_name']} !!")
        
        choice = None
        with st.expander("Select the feature", expanded=True):
            choice = st.radio(
                "Choose an action",
                options=["Upload Medical Report", "View File Parts", "Chat with Medical Report Bot"],
                key=f"radio_{st.session_state['user_id']}"
            )
        if choice == "Upload Medical Report":
            file = st.file_uploader(
                "Upload your medical report (PDF)",
                type='pdf',
                key=f"upload_{st.session_state['user_id']}"
            )
            if file:
                file_name = file.name
                file_path = os.path.join(UPLOAD_DIR, f"{st.session_state['user_id']}_{file_name}")
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(file.read())
                if st.button("Save file", key=f"save_{st.session_state['user_id']}"):
                    save_file(st.session_state['user_id'], file_name, file_path)
                    st.success(f"File {file_name} saved successfully!")
            st.subheader("Your Uploaded Files")
            files = get_user_files(st.session_state['user_id'])
            if files:
                for fname, fpath in files:
                    st.markdown(f"- {fname}")
                    if st.button(f"Delete {fname}", key=f"del_{fname}"):
                        delete_file(st.session_state['user_id'], fname)
                        if os.path.exists(fpath):
                            os.remove(fpath)
                        st.success(f"File {fname} deleted successfully!")
            else:
                st.info("No files uploaded yet.")
        elif choice == "View File Parts":
            files = get_user_files(st.session_state['user_id'])
            if not files:
                st.warning("Please upload at least one medical report first.")
            else:
                s_file = st.selectbox(
                    "Select the file to view",
                    options=[i for i, v in files],
                    key=f"select_view_{st.session_state['user_id']}"
                )
                def get_value(i, lst):
                    for pair in lst:
                        if pair[0] == i:
                            return pair[1]
                    return None
                if s_file:
                    file_path = get_value(s_file, files)
                    if st.button("View Parts", key=f"view_{s_file}"):
                        with st.spinner("Extracting Parts..."):
                            pdf_reader = PdfReader(file_path)
                            text = ''
                            for page in pdf_reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text
                            st.subheader(f"Parts of {s_file}")
                            st.write(text)
                            
        elif choice == "Chat with Medical Report Bot":
            files = get_user_files(st.session_state['user_id'])
            if not files:
                st.warning("Please upload at least one medical report first.")
            else:
                s_file = st.selectbox(
                    "Select the file to chat with",
                    options=[i for i, v in files],
                    key=f"select_chat_{st.session_state['user_id']}"
                )
                file_path = dict(files).get(s_file)
                if file_path and os.path.exists(file_path):
                    pdf_reader = PdfReader(file_path)
                    text = ''
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    if not text or not isinstance(text, str):
                        st.error("Failed to extract valid text from the PDF. Please ensure the PDF contains readable text.")
                        
                    try:
                        chunks = get_chunks(text)
                        db = get_vector_store(chunks)
                    except Exception as e:
                        st.error(f"Error processing report: {str(e)}")
                        
                    if st.session_state.get('messages') is None:
                        st.session_state['messages'] = {}
                    chat_history = st.session_state['messages'].get(st.session_state['user_id'], [])
                    for message in chat_history:
                        st.chat_message(message['role']).markdown(message['parts'])
                    user_question = st.chat_input("Ask a question about this report...")
                    if user_question:
                        st.chat_message("user").markdown(user_question)
                        chat_history.append({'role': 'user', 'parts': user_question})
                        with st.spinner("Analyzing report..."):
                            rel_text = get_rel_text(user_question, db)
                            response_text = bot_response(model, user_question, [rel_text], chat_history)
                            st.chat_message("assistant").markdown(response_text)
                            chat_history.append({'role': 'assistant', 'parts': response_text})
                        st.session_state['messages'][st.session_state['user_id']] = chat_history
                else:
                    st.error("Selected file not found. Please re-upload it.")

