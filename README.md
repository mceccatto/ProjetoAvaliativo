# Spam Analyzer

Este é um aplicativo web para análise e classificação de mensagens spam usando diferentes algoritmos de machine learning. O sistema permite upload de datasets, visualização de dados, geração de gráficos de análise e treinamento de modelos de classificação.

## 🚀 Funcionalidades

- Upload de arquivos CSV contendo mensagens para análise
- Visualização dos dados carregados
- Análise visual através de diferentes tipos de gráficos
- Múltiplos algoritmos de classificação:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - Regressão Logística
- Configuração de parâmetros para cada algoritmo
- Relatório de desempenho do modelo
- Visualização da matriz de confusão

## 📋 Pré-requisitos

- Python 3.7+
- pip (gerenciador de pacotes Python)

## 🔧 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/spam-analyzer.git
cd spam-analyzer
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📦 Dependências Principais

```txt
flask
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
werkzeug
```

## 🛠️ Estrutura do Projeto

```
spam-analyzer/
│
├── app.py              # Aplicação principal Flask
├── templates/          # Templates HTML
│   └── upload.html     # Página de upload e visualização
├── uploads/           # Diretório para arquivos enviados
├── requirements.txt   # Dependências do projeto
└── README.md         # Documentação
```

## 💻 Como Usar

1. Inicie o servidor Flask:
```bash
python app.py
```

2. Acesse o aplicativo em seu navegador:
```
http://localhost:5000
```

3. Prepare seu arquivo de dados:
   - O arquivo deve estar no formato CSV
   - Deve conter as colunas obrigatórias:
     - `message`: texto da mensagem
     - `label`: classificação (spam/ham)

4. Upload e Análise:
   - Faça upload do arquivo CSV
   - Selecione o algoritmo de classificação
   - Configure os parâmetros desejados
   - Clique em "Processar Dataset"

## ⚙️ Parâmetros dos Algoritmos

### Naive Bayes
- `alpha`: Parâmetro de suavização (default: 1.0)

### SVM
- `kernel`: Tipo de kernel (linear, rbf, polynomial)
- `C`: Parâmetro de regularização

### Random Forest
- `n_estimators`: Número de árvores
- `max_depth`: Profundidade máxima das árvores

### Regressão Logística
- `max_iter`: Número máximo de iterações
- `C`: Força da regularização L2

## 📊 Visualizações Geradas

O sistema gera automaticamente os seguintes gráficos:

1. Distribuição de Classes (Spam vs Ham)
2. Comprimento das Mensagens por Classe
3. Distribuição do Comprimento das Mensagens
4. Top 10 Palavras mais Comuns em Spam
5. Número Médio de Palavras por Classe

## 📈 Métricas de Avaliação

O sistema fornece as seguintes métricas de desempenho:

- Relatório de classificação completo
  - Precisão
  - Recall
  - F1-Score
  - Suporte
- Matriz de confusão visual

## 🔒 Segurança

- Validação de extensão de arquivo
- Nomes de arquivo seguros
- Chave secreta para sessão Flask

## ⚠️ Observações Importantes

- O diretório `uploads/` deve ter permissões de escrita
- Recomenda-se o uso de ambiente virtual
- Em produção, configure adequadamente a `SECRET_KEY` do Flask
- O tamanho máximo do arquivo pode precisar de ajuste dependendo do servidor

## ✒️ Autores

Gustavo Henrique
Luan Barleze
Marcelo Ceccatto
Matheus de Carvalho Pereira Pinto
Nathan Pechebovicz