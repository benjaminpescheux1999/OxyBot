# Assistant OxyBot (with Ollama, langchain, chromaDB, streamlit)
![1__6rQ18vn8wNAqdcEeTxBug](https://github.com/tien-tran0906/mistral_personal_mba/assets/117805369/805ce0bd-9586-4306-a73c-f0226c06b6d5)
<!-- 1. Clone the project ```git clone https://github.com/tien-tran0906/mistral_personal_mba.git``` -->
1. Installer python: https://www.python.org/downloads/release/python-3115/ version Python 3.11.5
2. Activer l'environnement virtuel:
    ```python -m venv venv```
    ```for windows: venv\Scripts\Activate.ps1```
    ```for mac/linux: source venv/bin/activate```
3. Installer les packages pip ```pip install -r requirements.txt```
4. Si vous voulez utiliser vos propres documents, supprimez le fichier présent dans 'source_documents' et glissez-y vos propres documents
5. Run ```python directory_public.py``` pour créer le répertoire des documents publics
6. Run ```python ingest.py``` pour ingérer les documents dans la base de données
7. Run ```streamlit run app.py``` pour démarrer l'application OxyBot
