import streamlit as st
from langchain_community.llms import Ollama
import os
import chromadb
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.callbacks.base import BaseCallbackHandler
import urllib.parse
import streamlit.components.v1 as components
from streamlit_pdf_viewer import pdf_viewer
import base64
import requests
from streamlit_pdf_reader import pdf_reader
import PyPDF2
import pandas as pd
import io


# Custom streamlit handler to display LLM outputs in stream mode
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:

        self.text+=token+"" 
        self.container.markdown(self.text)

# streamlit UI configuration
def setup_page():
    st.set_page_config(layout="wide")
    st.markdown("<h2 style='text-align: center; color: white;'>OxyBot </h2>" , unsafe_allow_html=True)
    col1, col2, col3= st.columns(3)
    with col2:
        st.markdown("""
            <div style="text-align: center;">
            <h5 style='color: white;'>Ce chatBot est une assisance, pour répondre à vos demande simple.</h5>
            <p>Vous pouvez poser des questions sur le SAV, Oxygene, Memsoft, Mobilités, etc.</p>
            <span>Attention les réponses peuvent être approximatives.</span>
            <span>Pour des questions plus complèxes, vous pouvez contacter les développeurs.</span>
            <a href="mailto:support@info-tec.fr">Contacter le support</a>
            </div>
            """, unsafe_allow_html=True)
    st.divider()

# get necessary environment variables for later use
def get_environment_variables():
    model = os.environ.get("MODEL", "mistral")
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L12-v2")
    persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
    return model, embeddings_model_name, persist_directory, target_source_chunks

# create knowledge base retriever
def create_knowledge_base(embeddings_model_name, persist_directory, target_source_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    return retriever

# def display_pdf_from_url(base64_pdf, page, index):
#     # Create an HTML string with the embedded PDF for the specified page
#     pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}#page={page}" width="700" height="1000" type="application/pdf" key={index}>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

def get_pdf_page_as_base64(url, page_number, index):
    # Télécharge le PDF
    response = requests.get(url)
    response.raise_for_status()

    # Ouvre le PDF téléchargé
    with io.BytesIO(response.content) as open_pdf_file:
        pdf_reader = PyPDF2.PdfReader(open_pdf_file)
        
        # Crée un nouveau PDF avec seulement la page souhaitée
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_number - 1])  # Les pages commencent à l'index 0

        # Sauvegarde la page unique dans un objet BytesIO
        pdf_output = io.BytesIO()
        pdf_writer.write(pdf_output)

        # Encode le PDF de la page unique en base64
        pdf_output.seek(0)
        base64_pdf = base64.b64encode(pdf_output.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf" key={index}>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        # return base64_pdf

def get_csv_excel(url, extension, select_rows):
    if extension == ".csv":
        try:
            df = pd.read_csv(url, encoding='utf-8', delimiter=';', quotechar='"')
        except UnicodeDecodeError:
            df = pd.read_csv(url, encoding='iso-8859-1', delimiter=';', quotechar='"')
        except pd.errors.ParserError:
            df = pd.read_csv(url, encoding='utf-8', delimiter=';', quotechar='"', error_bad_lines=False)
    else:
        df = pd.read_excel(url)

    # Filter the DataFrame to only include selected rows
    if select_rows:
        df = df.loc[df.index.isin(select_rows)]

    st.dataframe(df, key=f'df-{url}')
    # st.write(df)

def display_sources(grouped_metadata):
    # Déterminer le nombre de colonnes en fonction de la largeur de la fenêtre
    window_width = st.session_state.get('window_width', 1000)  # Valeur par défaut si non défini
    if window_width < 640:
        columns_per_page = 1  # Mobile
    elif 640 <= window_width < 1024:
        columns_per_page = 1  # Tablet
    else:
        columns_per_page = 2  # Desktop
    base_url = "http://localhost:5000/files/"

    for index, (url, data) in enumerate(grouped_metadata.items()):
        with st.expander(f"Source du document: {url}"):
            file_name = url.split('\\')[-1]  # Extraire le nom de fichier
            encoded_file_name = urllib.parse.quote(file_name)  # Encoder le nom de fichier pour URL
            full_url = f"{base_url}{encoded_file_name}"  # Construire l'URL complète
            response = requests.get(full_url)
            response.raise_for_status()
            pages_text = ", ".join(map(str, data["pages"]))
            rows_text = ", ".join(map(str, data["rows"]))

            extension = os.path.splitext(url)[1]
            if extension == ".pdf":
                st.markdown(f"**Pages:** {pages_text}")
                cols = st.columns(columns_per_page)
                for i, page in enumerate(data["pages"]):
                    with cols[i % columns_per_page]: 
                        get_pdf_page_as_base64(full_url, page, f"{index}_{page}")
            if extension == ".csv":
                st.markdown(f"**Lignes:** {rows_text}")
                get_csv_excel(full_url, extension, data["rows"])


def handle_query(query, model, retriever):
    agent_context = """Début des instructions
1. Rôle: Tu es un assistant.
2. Langue: Tu parles uniquement en Français.
3. Domaine de réponse: Tu vas répondre uniquement en parlant du logiciel.
4. Noms du logiciel: Le logiciel Desktop est connu sous plusieurs noms : Memsoft, SAV, Sav++, Oxygène.
5. Entreprise: L'entreprise qui commercialise ce logiciel s'appelle Infotec.
6. Solution logicielle: La solution comprend le logiciel desktop et un logiciel mobilité pour des PDA ou tablettes.
7. Style de réponse: Réponds le plus simplement possible, en utilisant des listes ou des étapes.
Type de questions: Les questions posées seront principalement des problèmes rencontrés ou des demandes de procédure.
9. Importance des instructions: Le respect des instructions est la chose la plus importante.
10. Contradiction: En aucun cas tu ne dois contredire les éléments indiqués dans les instructions.
Fin des instructions"""

    full_query = f"{agent_context} {query}"

    with st.chat_message('assistant'):
        with st.spinner("Recherche en cours..."):
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)  
            llm = Ollama(model=model, callbacks=[stream_handler], temperature=0.1, top_p=0.9, top_k=1)
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever, 
                return_source_documents=True)
            try:
                res = qa.invoke(full_query)
                grouped_metadata = {}
                #si res['source_documents'] n'est pas vide
                st.markdown(res)
                if res['source_documents']:
                    st.markdown("**Sources:**")
                    for doc in res['source_documents']:
                        source_url = doc.metadata['source']
                        if source_url not in grouped_metadata:
                            grouped_metadata[source_url] = {"pages": [], "rows": []}
                        if 'page' in doc.metadata:
                            grouped_metadata[source_url]["pages"].append(doc.metadata['page'])
                        if 'row' in doc.metadata:
                            grouped_metadata[source_url]["rows"].append(doc.metadata['row'])
                    answer = res['result']
                    message_text = answer  # Commencez avec le texte de la réponse

                    # Construire le texte des métadonnées
                    metadata_texts = []
                    for url, data in grouped_metadata.items():
                        metadata_texts.append(f"- **Source du document :** {url}")
                        pages_text = ", ".join(map(str, data["pages"]))
                        rows_text = ", ".join(map(str, data["rows"]))
                        metadata_texts.append(f"  - **Pages :** {pages_text}")
                        metadata_texts.append(f"  - **Lignes :** {rows_text}")

                    metadata_string = "\n".join(metadata_texts)
                    # message_text += "\n### Sources :\n" + metadata_string  # Ajouter les métadonnées au texte de la réponse
                    # message_placeholder.markdown(message_text)  # Afficher le texte concaténé
                    return answer, grouped_metadata
            except Exception as e:
                print("Erreur lors de l'exécution de la requête :", e)
                message_placeholder.markdown("Une erreur est survenue lors de la recherche de votre réponse.")
                st.error("Une erreur est survenue lors de la recherche de votre réponse.")
                return "Une erreur est survenue lors de la recherche de votre réponse."

# dictionary to store the previous messages, create a 'memory' for the LLM
def initialize_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

# display the messages
def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# example questions when user first load up the app. Will disappear after user send the first query
def show_examples():
    examples = st.empty()
    with examples.container():
        with st.chat_message('assistant'):
            st.markdown('Example questions:')
            st.markdown(' - Comment configurer les emails dans Oxygene (SAV) ?')
            st.markdown(" - Quelles sont les procédures à suivre lorsque le service Mobilités n'arrive pas à récupérer les tournées ?")
            st.markdown(" - Quelles sont les procédures à suivre lorsque le service Mobilités n'arrive pas à récupérer les nouveaux articles ?")
            st.markdown(" - Comment configurer une nouvelle nomenclature dans le SAV ?")
            st.markdown('Comment puis-je vous aider?')
    return examples

def main():
    setup_page()
    initialize_session()
    display_messages()
    examples = show_examples()
    model, embeddings_model_name, persist_directory, target_source_chunks = get_environment_variables()
    retriever = create_knowledge_base(embeddings_model_name, persist_directory, target_source_chunks)
    
    query = st.chat_input(placeholder='Posez une question...')  # starting with empty query

    if query:   # if user input a query and hit 'Enter'
        examples.empty()
        st.session_state.messages.append({  # add the query into session state/ dictionary
            'role': 'user', 
            'content': query
        })

        with st.chat_message('user'):
            st.markdown(query)

        answer, grouped_metadata = handle_query(query, model, retriever)

        st.session_state.messages.append({  # add the answer into session state/ dictionary
             'role': 'assistant',
             'content': answer  
        })
        display_sources(grouped_metadata)

if __name__ == "__main__":
    main()