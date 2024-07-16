from flask import Flask, send_from_directory, abort
import os

app = Flask(__name__)

@app.route('/files/<path:filename>')
def download_file(filename):
    # Assurez-vous que le chemin est sécurisé
    safe_path = os.path.abspath(os.path.join('source_documents', filename))
    if not safe_path.startswith(os.path.abspath('source_documents')):
        abort(404)  # Sécurité pour éviter le Directory Traversal

    # Vérifiez si le fichier existe
    if not os.path.exists(safe_path):
        abort(404)  # Si le fichier n'existe pas, retournez une erreur 404

    return send_from_directory('source_documents', filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
