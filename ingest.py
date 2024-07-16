import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import base64

from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.docstore.document import Document


# Charger les variables d'environnement
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L12-v2')
chunk_size = 500
chunk_overlap = 50

# Chargeurs de documents personnalisés
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper pour revenir à text/plain lorsque le défaut ne fonctionne pas"""

    def load(self) -> List[Document]:
        """Wrapper ajoutant une solution de repli pour elm sans html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Essayer le texte brut
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Ajouter file_path au message d'exception
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc
class CustomCSVLoader(CSVLoader):
    def load(self) -> List[Document]:
        documents = super().load()  # Utilisez la méthode de chargement existante
        for index, doc in enumerate(documents):
            doc.metadata['row'] = index  # Ajoutez le numéro de ligne aux métadonnées
        return documents
# Mapper les extensions de fichiers aux chargeurs de documents et leurs arguments
LOADER_MAPPING = {
    ".csv": (CustomCSVLoader, {"csv_args":{"delimiter":";"}}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {"extract_images":True }),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Ajouter plus de mappages pour d'autres extensions de fichiers et chargeurs si nécessaire
}


# Charge un seul document à partir du chemin de fichier spécifié
def load_single_document(file_path: str) -> List[Document]: # Retourne une liste d'objets 'Document'
    ext = "." + file_path.rsplit(".", 1)[-1]
    print("file_path ==>", file_path)
    print("ext ==>", ext)
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        print("loader_class ==>", loader_class)
        print("loader_args ==>", loader_args)
        loader = loader_class(file_path=file_path, **loader_args)
        print("loader ==>", loader)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

# S'il y a plus d'un document, charge tous les documents d'un répertoire source, en ignorant éventuellement des fichiers spécifiques
def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Charge tous les documents du répertoire des documents source, en ignorant les fichiers spécifiés
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    # Traitement parallèle
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Chargement de nouveaux documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):    # Appeler load_single_document pour chaque chemin de fichier en parallèle
                results.extend(docs)
                pbar.update()

    return results

# Charge les documents depuis source_directory en utilisant load_document
def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Charger les documents et les diviser en morceaux
    """
    print(f"Chargement des documents depuis {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("Aucun nouveau document à charger")
        exit(0)
    print(f"Chargé {len(documents)} nouveaux documents depuis {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # diviser les documents en morceaux de texte
    texts = text_splitter.split_documents(documents)
    print(f"Divisé en {len(texts)} morceaux de texte (max. {chunk_size} tokens chacun)")
    return texts    # retourner une liste de morceaux de texte divisés

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Vérifie si le vectorstore existe
    """
    # vérifie la présence des fichiers et dossiers nécessaires pour un vectorstore valide
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # Au moins 3 documents sont nécessaires dans un vectorstore fonctionnel
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Créer des embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        print(f"Ajout au vectorstore existant à {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Création des embeddings. Cela peut prendre quelques minutes...")
        db.add_documents(texts)
    else:
        print("Création d'un nouveau vectorstore")
        texts = process_documents()
        print(f"Création des embeddings. Cela peut prendre quelques minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    
    db = None
    print(f"Ingestion terminée ! Vous pouvez maintenant exécuter app.py pour commencer à interroger vos documents.")


if __name__ == "__main__":
    main()