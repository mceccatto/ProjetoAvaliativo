from flask import Flask, render_template, request, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import base64
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Configuração inicial do seaborn
sns.set_theme(style="whitegrid")

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta_aqui'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    # Verifica se a extensão do arquivo é permitida
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_graphs(df):
    # Gera gráficos para análise visual dos dados
    graphs = []

    # Configurações globais
    colors = ['#FF9999', '#66B3FF']

    # Configuração global do matplotlib
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.autolayout': True
    })

    try:
        # Distribuição de Classes (Pie Chart)
        fig, ax = plt.subplots()
        class_dist = df['label'].value_counts()
        ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Distribuição de Classes (Spam vs Ham)', pad=20)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig)

        # Comprimento das mensagens por classe (Box Plot)
        fig, ax = plt.subplots()
        df['message_length'] = df['message'].str.len()
        sns.boxplot(ax=ax, x='label', y='message_length', data=df, palette=colors)
        ax.set_title('Comprimento das Mensagens por Classe', pad=20)
        ax.set_xlabel('Classe')
        ax.set_ylabel('Comprimento da Mensagem')
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig)

        # Distribuição de comprimento das mensagens (KDE Plot)
        fig, ax = plt.subplots()
        for label, color in zip(df['label'].unique(), colors):
            sns.kdeplot(ax=ax, data=df[df['label'] == label]['message_length'],
                        label=label, color=color)
        ax.set_title('Distribuição do Comprimento das Mensagens', pad=20)
        ax.set_xlabel('Comprimento da Mensagem')
        ax.set_ylabel('Densidade')
        ax.legend()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig)

        # Top palavras mais comuns em Spam
        vectorizer = CountVectorizer(stop_words='english', max_features=10)
        spam_messages = df[df['label'] == 'spam']['message']
        if not spam_messages.empty:
            spam_vectors = vectorizer.fit_transform(spam_messages)
            words = vectorizer.get_feature_names_out()
            counts = spam_vectors.sum(axis=0).A1

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(words, counts, color=colors[0])
            ax.set_title('Top 10 Palavras mais Comuns em Spam', pad=20)
            plt.xticks(rotation=45, ha='right')
            ax.set_xlabel('Palavras')
            ax.set_ylabel('Frequência')
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            graphs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            plt.close(fig)

        # Média de palavras por classe
        df['word_count'] = df['message'].str.split().str.len()
        fig, ax = plt.subplots()
        avg_words = df.groupby('label')['word_count'].mean()
        avg_words.plot(kind='bar', color=colors, ax=ax)
        ax.set_title('Número Médio de Palavras por Classe', pad=20)
        ax.set_xlabel('Classe')
        ax.set_ylabel('Média de Palavras')
        plt.xticks(rotation=0)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close(fig)

    except Exception as e:
        print(f"Erro ao gerar gráficos: {str(e)}")
        return []

    return graphs


def get_classifier(classifier_name, params):
    # Retorna o classificador apropriado com os parâmetros especificados
    try:
        if classifier_name == 'naive_bayes':
            return MultinomialNB(
                alpha=float(params.get('alpha', 1.0))
            )

        elif classifier_name == 'svm':
            return SVC(
                kernel=params.get('kernel', 'linear'),
                C=float(params.get('C', 1.0)),
                probability=True,
                random_state=42
            )

        elif classifier_name == 'random_forest':
            max_depth = params.get('max_depth')
            max_depth = int(max_depth) if max_depth and max_depth != 'None' else None
            return RandomForestClassifier(
                n_estimators=int(params.get('n_estimators', 100)),
                max_depth=max_depth,
                random_state=42
            )

        elif classifier_name == 'logistic_regression':
            return LogisticRegression(
                max_iter=int(params.get('max_iter', 100)),
                C=float(params.get('lr_C', 1.0)),
                random_state=42
            )

        return MultinomialNB()

    except Exception as e:
        print(f"Erro ao configurar classificador: {str(e)}")
        return MultinomialNB()


def train_and_evaluate_model(df, classifier_name, params):
    # Treina e avalia o modelo com os parâmetros especificados
    try:
        # Preparar os dados
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['message'])
        y = df['label']

        # Dividir os dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Obter e treinar o classificador
        classifier = get_classifier(classifier_name, params)
        classifier.fit(X_train, y_train)

        # Fazer predições
        y_pred = classifier.predict(X_test)

        # Gerar relatório de classificação
        report = classification_report(y_test, y_pred)

        # Gerar matriz de confusão
        cm = confusion_matrix(y_test, y_pred)

        # Salvar o modelo e o vectorizer
        model_filename = f'model_{classifier_name}.joblib'
        vectorizer_filename = f'vectorizer_{classifier_name}.joblib'
        joblib.dump(classifier, os.path.join(app.config['UPLOAD_FOLDER'], model_filename))
        joblib.dump(vectorizer, os.path.join(app.config['UPLOAD_FOLDER'], vectorizer_filename))

        return report, cm

    except Exception as e:
        print(f"Erro no treinamento do modelo: {str(e)}")
        return None, None


def generate_model_performance_graph(cm):
    # Gera um gráfico de matriz de confusão com backend não-interativo
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'], ax=ax)
        ax.set_title('Matriz de Confusão')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predito')

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        confusion_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        return confusion_matrix_img

    except Exception as e:
        print(f"Erro ao gerar matriz de confusão: {str(e)}")
        return None


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = None
    columns = None
    filename = None
    graphs = None
    model_report = None
    confusion_matrix_img = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nenhum arquivo selecionado', 'error')
            return render_template('upload.html')

        file = request.files['file']
        if file.filename == '':
            flash('Nenhum arquivo selecionado', 'error')
            return render_template('upload.html')

        if file and allowed_file(file.filename):
            try:
                # Leitura do arquivo
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                df = pd.read_csv(filepath)

                # Verificação das colunas necessárias
                required_columns = {'message', 'label'}
                if not required_columns.issubset(df.columns):
                    raise ValueError("O arquivo deve conter as colunas 'message' e 'label'")

                # Preparação dos dados
                data = df.head(10).values.tolist()
                columns = df.columns.tolist()

                # Geração dos gráficos
                graphs = generate_graphs(df)
                if not graphs:
                    flash('Erro ao gerar gráficos de análise', 'error')

                # Obter parâmetros do classificador
                classifier_name = request.form.get('classifier', 'naive_bayes')
                params = {
                    'alpha': request.form.get('alpha'),
                    'kernel': request.form.get('kernel'),
                    'C': request.form.get('C'),
                    'n_estimators': request.form.get('n_estimators'),
                    'max_depth': request.form.get('max_depth'),
                    'max_iter': request.form.get('max_iter'),
                    'lr_C': request.form.get('lr_C')
                }

                # Treinar e avaliar o modelo
                model_report, cm = train_and_evaluate_model(df, classifier_name, params)
                if model_report and cm is not None:
                    confusion_matrix_img = generate_model_performance_graph(cm)
                    flash(f'Arquivo {filename} processado com sucesso!', 'success')
                else:
                    flash('Erro ao treinar o modelo', 'error')

            except Exception as e:
                flash(f'Erro ao processar o arquivo: {str(e)}', 'error')
                data = None
                columns = None
                graphs = None
                model_report = None
                confusion_matrix_img = None
        else:
            flash('Formato de arquivo não permitido. Use apenas CSV.', 'error')

    return render_template('upload.html',
                           data=data,
                           columns=columns,
                           filename=filename,
                           graphs=graphs,
                           model_report=model_report,
                           confusion_matrix_img=confusion_matrix_img)


if __name__ == '__main__':
    app.run(debug=True)