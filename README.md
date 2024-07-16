# Business Assistant using Mistral 7B (with Ollama, langchain, chromaDB, streamlit)
![1__6rQ18vn8wNAqdcEeTxBug](https://github.com/tien-tran0906/mistral_personal_mba/assets/117805369/805ce0bd-9586-4306-a73c-f0226c06b6d5)
<!-- 1. Clone the project ```git clone https://github.com/tien-tran0906/mistral_personal_mba.git``` -->
1. Install python: https://www.python.org/downloads/release/python-3115/ version Python 3.11.5
2. Enable virtual environment:
    ```python -m venv venv```
    ```for windows: venv\Scripts\Activate.ps1```
    ```for mac/linux: source venv/bin/activate```
3. Install the pip packages ```pip install -r requirements.txt```
4. If you want to use your own document, delete the .PDF file in 'source_documents' and drop in your own
5. Run ```python ingest.py``` pour ingérer les documents dans la base de données
6. Run ```python directory_public.py``` pour créer le répertoire des documents publics
7. Run ```streamlit run app.py``` pour démarrer l'application OxyBot
