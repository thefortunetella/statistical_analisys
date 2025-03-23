Ótimo! Vou documentar cada célula do seu notebook em formato Markdown, incluindo o código, explicações técnicas acessíveis, fórmulas estatísticas quando aplicável, e interpretações claras. O estilo será formal-acadêmico, porém didático, pensado para leitores da disciplina de Probabilidade e Estatística.

Vou gerar o documento para você e te aviso assim que estiver pronto!

# Análise da Correlação entre Taxas de Juros (Selic e Fed Funds) e o Impacto da Variação da Selic em Retornos de Ações (2019–2024)

## Introdução

 ([image]()) *Figura 1: Fluxograma do processo analítico realizado no notebook.*  

Este relatório documenta, passo a passo, uma análise estatística envolvendo taxas de juros e dados do mercado acionário brasileiro no período de **2019 a 2024**. Em especial, examina-se a **correlação entre a taxa Selic (Brasil) e a Fed Funds Rate (EUA)** e verificam-se estatisticamente a significância dessa correlação e seu intervalo de confiança. Além disso, investiga-se a **variação da Selic ao longo dos anos** (2019–2024) por meio de uma ANOVA de uma via, bem como a relação entre as **oscilações da Selic e os retornos de ações de empresas industriais**. O fluxograma da **Figura 1** resume as principais etapas do trabalho, desde a coleta de dados até as inferências estatísticas. A seguir, cada célula do notebook é detalhada com o código correspondente, explicação técnica, fundamentação estatística (incluindo fórmulas relevantes) e interpretação dos resultados obtidos.

---

## Preparação dos Dados e Importação de Bibliotecas

Para iniciar a análise, o notebook carrega os **pacotes e bibliotecas necessárias**. Isso inclui bibliotecas para manipulação de dados, obtenção de dados financeiros e execução de testes estatísticos.

**Código da Célula 1:** _Importação de bibliotecas_  
```python
import requests
import pandas as pd
import yfinance as yf
import plotly as plt
import plotly.express as px
import streamlit as st
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway
import numpy as np
``` 

**Explicação:** Este código importa diversas bibliotecas:
- `requests` para realizar requisições web (usado para obter dados da API do Banco Central do Brasil).  
- `pandas` para manipulação de dados em formato de data frame (leitura de CSVs, DataFrames, etc.).  
- `yfinance` para acesso a dados financeiros (cotação de ações).  
- Bibliotecas de visualização interativa `plotly` e `plotly.express` para gerar gráficos (heatmaps, boxplots, etc.).  
- `streamlit` (embora listada, não é utilizada diretamente na análise subsequente).  
- O módulo `scipy.stats` e funções `ttest_ind` (teste t independente) e `f_oneway` (ANOVA de uma via) para cálculos estatísticos.  
- `numpy` para operações numéricas de suporte.  

Essas importações garantem que todas as **ferramentas estatísticas e de obtenção de dados** estejam disponíveis. Por exemplo, `f_oneway` será usada para ANOVA e `stats` fornecerá distribuições estatísticas (e.g., distribuição normal) para calcular intervalos de confiança.

**Justificativa:** A preparação inclui carregar métodos de teste de hipóteses e intervalos de confiança que serão aplicados adiante. Essa etapa é crucial para evitar interrupções posteriores e assegurar reprodutibilidade, pois cada função necessária já está importada antes do uso.

---

## Coleta de Dados de Ações e Taxas de Juros

Nesta etapa, define-se um conjunto de **tickers** de ações do setor industrial brasileiro e realiza-se a coleta de dados históricos dessas ações, assim como das taxas de juros Selic (Brasil) e Fed Funds (EUA). Também são feitos filtros no período de análise (2019–2024).

### Seleção e Verificação de Tickers de Ações Industriais

O primeiro passo é definir os códigos (tickers) das ações a serem analisadas e verificar se estão disponíveis via API do Yahoo Finance.

**Código da Célula 2:** _Definição e verificação de tickers_  
```python
tickers = [
    "WEGE3.SA", "KEPL3.SA", "ROMI3.SA", "TASA3.SA", "SHUL4.SA", "POMO4.SA",
    "VLID3.SA", "EMBR3.SA", "RAPT4.SA", "TUPY3.SA", "RAIL3.SA", "GOLL4.SA",
    "AZUL4.SA", "JSLG3.SA", "LOGN3.SA"
]

# Função para testar se os tickers estão disponíveis
def check_tickers_availability(tickers):
    valid_tickers = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1mo", progress=False)
            if not df.empty:
                valid_tickers.append(ticker)
        except Exception as e:
            print(f"Erro ao verificar {ticker}: {e}")
    return valid_tickers

# Verificar tickers válidos
valid_tickers = check_tickers_availability(tickers)

# Exibir os tickers que funcionam
print("Tickers disponíveis:", valid_tickers)
``` 

**Explicação:** Definimos inicialmente uma lista de 15 tickers de ações (todas com sufixo `.SA`, indicando ações negociadas na **B3**, bolsa de São Paulo, por exemplo `WEGE3.SA` da WEG S.A.). Em seguida, a função `check_tickers_availability` tenta baixar dados de cada ticker para um período curto (“1mo” – um mês) usando `yfinance`. Se o download retorna dados (i.e., `df` não está vazio), o ticker é considerado válido e adicionado à lista `valid_tickers`. Caso ocorra alguma exceção durante o download, é capturada e exibida uma mensagem de erro indicando o ticker problemático. Após testar todos os tickers, a lista `valid_tickers` conterá apenas aqueles símbolos cujos dados puderam ser acessados com sucesso.

No final, imprime-se os tickers disponíveis. A saída desta célula confirmará quais tickers serão utilizados. Supondo sucesso, a impressão pode ser algo como: 
```
Tickers disponíveis: ['WEGE3.SA', 'KEPL3.SA', ..., 'LOGN3.SA']
``` 
indicando que todos (ou a maioria) dos códigos fornecidos foram reconhecidos. 

**Interpretação:** Essa etapa garante que a análise prossiga apenas com **ações válidas** e disponíveis, evitando falhas posteriores ao tentar usar tickers inexistentes ou com dados indisponíveis. No contexto de **Probabilidade e Estatística**, este é um passo de **preparo amostral**: definimos nosso conjunto de amostras (as ações) e asseguramos que temos dados para cada uma, evitando viés de seleção acidental por falta de dados.

### Download dos Dados Históricos das Ações

Com os tickers válidos identificados, procede-se a baixar os dados históricos dessas ações. Define-se o período de análise de janeiro/2019 a dezembro/2024 e utiliza-se frequência mensal.

**Código da Célula 3:** _Download dos preços das ações_  
```python
# Lista de tickers válidos
valid_tickers = ['WEGE3.SA', 'KEPL3.SA', 'ROMI3.SA', 'TASA3.SA', 'SHUL4.SA', 
                 'POMO4.SA', 'VLID3.SA', 'EMBR3.SA', 'RAPT4.SA', 'TUPY3.SA', 
                 'RAIL3.SA', 'GOLL4.SA', 'AZUL4.SA', 'JSLG3.SA', 'LOGN3.SA']

# Definir o período de análise
start_date = "2019-01-01"
end_date = "2024-12-31"

# Função para baixar os preços de fechamento ajustados das ações
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, interval='1mo', progress=False)['Close']
    return data

# Baixando os dados das ações disponíveis
stock_data = get_stock_data(valid_tickers, start_date, end_date)

# Exibir as primeiras linhas dos dados coletados
stock_data = stock_data.reset_index()
``` 

**Explicação:** Esta célula define explicitamente a lista de tickers a ser utilizada (resultado da verificação anterior) e o período de tempo de interesse (`2019-01-01` até `2024-12-31`). A função `get_stock_data` utiliza `yfinance.download` para obter os dados históricos de fechamento ajustado (`'Close'`) de todos os tickers em conjunto, no intervalo mensal (`interval='1mo'`) dentro do período especificado. A função retorna um DataFrame com colunas correspondentes a cada ticker e linhas indexadas por data. Em seguida, chamamos essa função para efetivamente baixar os dados (`stock_data`). 

O resultado é armazenado em `stock_data` e, para facilidade de manipulação posterior, aplica-se `reset_index()`. Isso transforma a coluna de datas (que estava como índice) em uma coluna normal chamada `Date`. Dessa forma, **`stock_data` terá colunas: `Date`, e uma coluna para cada ticker com os respectivos preços de fechamento ajustados mensais**.

**Interpretação dos Dados:** Após essa célula, temos uma **base de dados consolidada** com cerca de 6 anos de preços mensais para 15 ações industriais brasileiras. As primeiras linhas (`head()`) exibidas confirmam a estrutura: por exemplo, uma linha típica pode ser “2019-01-31, preço_WEGE3, preço_KEPL3, ..., preço_LOGN3”. Esses dados serão usados posteriormente para calcular retornos e correlações. É importante notar que eventuais *NaNs* poderiam surgir para datas em que alguma ação não tinha negociação (por exemplo, se alguma empresa abriu capital depois de 2019), mas, como todas são empresas já listadas antes de 2019, não se espera dados faltantes iniciais. Em caso de valores faltantes, técnicas de limpeza (dropna) seriam aplicadas.

---

### Coleta de Dados da Taxa Selic

Agora, buscamos os dados da **taxa Selic** (taxa básica de juros do Brasil). Utiliza-se a API do Banco Central do Brasil (BCB) para obter a série histórica mensal da Selic efetiva.

**Código da Célula 4:** _Obtenção dos dados da Selic via API BCB_  
```python
# Função para obter dados da SELIC via API do Banco Central do Brasil (BCB)
def get_selic_data():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json"  # Código 4189 corresponde à SELIC
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df_selic = pd.DataFrame(data)
        df_selic['data'] = pd.to_datetime(df_selic['data'], format='%d/%m/%Y')
        df_selic['valor'] = df_selic['valor'].astype(float)
        df_selic.rename(columns={'data': 'Date', 'valor': 'Selic'}, inplace=True)
        return df_selic
    else:
        print("Erro ao obter dados da SELIC")
        return None

# Obtendo os dados da SELIC
df_selic = get_selic_data()

# Filtrar os dados para manter apenas de 2019 a 2024
df_selic_filtered = df_selic[(df_selic['Date'] >= "2019-01-01") & (df_selic['Date'] <= "2024-12-31")]

# Exibir as primeiras linhas dos dados filtrados
print(df_selic_filtered.head())
``` 

**Explicação:** Definimos a função `get_selic_data` para consultar a API do Banco Central do Brasil. O código da série temporal da Selic diária acumulada no mês (taxa Selic mensal efetiva) é **4189**, conforme a documentação do BCB, e o formato JSON é requisitado pela URL fornecida. A função realiza uma requisição HTTP GET; se bem-sucedida (código 200), converte a resposta JSON em um DataFrame `df_selic`. Em seguida, formata a coluna de datas (`'data'`) para o tipo datetime (especificando o formato original `%d/%m/%Y`), converte os valores de taxa para tipo numérico (`float`) e renomeia as colunas para `Date` e `Selic` para melhor compreensão. Caso a requisição falhe, imprime um erro e retorna `None`.

Após definir a função, o código a executa para obter `df_selic`. Então filtra `df_selic` para o intervalo desejado de 2019 a 2024, criando `df_selic_filtered`. Por fim, mostra as primeiras linhas filtradas. A saída típica mostraria colunas `Date` e `Selic`, por exemplo: 

```
Date       Selic
2019-01-01 6.4 
2019-02-01 6.4 
2019-03-01 6.4 
... 
``` 

Isso indica que nos primeiros meses de 2019 a Selic estava em 6,40% ao ano. (Vale notar que, durante parte de 2019, a Selic foi sendo reduzida no Brasil, atingindo mínimos históricos em 2020/2021 e voltando a subir posteriormente).

**Interpretação:** Ao final desta etapa, temos um **DataFrame `df_selic_filtered` contendo a taxa Selic mensal** (geralmente considerada a taxa vigente no início de cada mês ou média mensal). Esses valores são medidos em percentual ao ano. Filtrar de 2019 em diante garante que alinharemos este dado com os dados de ações e Fed Funds do mesmo período. Os dados confirmam, por exemplo, que a Selic manteve-se em 6,4% no início de 2019 (como era a meta Selic vigente) e posteriormente variou (valores não mostrados aqui, mas conhecidos: caiu para ~2% em 2020 e subiu novamente até 2023).

### Carregamento dos Dados da Fed Funds Rate (EUA)

A seguir, carregamos os dados da **Fed Funds Rate** (taxa básica de juros dos Estados Unidos), disponibilizados em um arquivo CSV (`FEDFUNDS.csv`). Essa série representa a taxa de juros do Federal Reserve (Fed) dos EUA, tipicamente a taxa de juros overnight entre bancos, em base percentual ao ano.

**Código da Célula 5:** _Leitura dos dados do Fed Funds Rate_  
```python
df_fed_funds = pd.read_csv("FEDFUNDS.csv", parse_dates=['observation_date'])

# Renomeando colunas
df_fed_funds.rename(columns={'observation_date': 'Date', 'FEDFUNDS': 'Fed_Funds'}, inplace=True)

# Filtrando os dados entre 2019 e 2024
df_fed_funds = df_fed_funds[(df_fed_funds['Date'] >= "2019-01-01") & (df_fed_funds['Date'] <= "2024-12-31")]

# Exibir os dados carregados
print(df_fed_funds.head())
``` 

**Explicação:** Aqui usamos o pandas para ler o arquivo CSV `FEDFUNDS.csv`. Esse arquivo contém a série temporal da Fed Funds Rate. O parâmetro `parse_dates=['observation_date']` instrui o pandas a converter a coluna de datas (`observation_date`) diretamente em objetos de data. Em seguida, renomeamos as colunas: `'observation_date'` para `Date` e `'FEDFUNDS'` para `Fed_Funds`, resultando em um DataFrame `df_fed_funds` com colunas `Date` e `Fed_Funds`. Como no caso da Selic, filtramos as linhas do DataFrame para manter apenas datas de 2019 a 2024. Por fim, exibimos as primeiras linhas filtradas. 

A saída esperada mostraria algo como: 

```
Date         Fed_Funds
2019-01-01   2.40
2019-02-01   2.40
2019-03-01   2.41
... 
``` 

Isso indica que, em janeiro de 2019, a Fed Funds Rate estava em 2,40% a.a., manteve-se nesse patamar nos meses seguintes até começar a variar ligeiramente. (De fato, a Fed Funds Rate foi reduzida durante 2019–2020 chegando perto de 0, e elevada fortemente em 2022–2023 em resposta à inflação, mas aqui vemos o início da série em 2019).

**Interpretação:** Esse conjunto de dados `df_fed_funds` agora contém a taxa básica americana nos mesmos períodos mensais. Junto com a Selic, esses dados permitem uma **análise comparativa entre as políticas monetárias do Brasil e dos EUA**. Dado que ambos os países enfrentaram ciclos econômicos (como a pandemia de 2020 e a recuperação com inflação em 2021–2023), espera-se observar **movimentos possivelmente correlacionados** entre as duas taxas, o que será examinado a seguir.

### Combinação das Séries de Taxa de Juros (Selic vs Fed Funds)

Agora que temos as duas séries de taxas de juros (Selic e Fed Funds) preparadas, vamos **combiná-las em um único DataFrame** alinhado por data, para facilitar comparações e cálculos de correlação. Precisamos também preparar o índice de datas adequadamente para a junção.

**Código da Célula 6:** _Junção das séries Selic e Fed Funds_  
```python
# Definir índice como data para facilitar junção
df_selic_filtered.set_index('Date', inplace=True)
df_fed_funds.set_index('Date', inplace=True)

# Juntar os dados pelo índice de data
df_rates = df_selic_filtered.join(df_fed_funds, how='inner')

# Resetar o índice
df_rates.reset_index(inplace=True)

# Exibir as primeiras linhas para validar
print(df_rates.head())
``` 

**Explicação:** Os DataFrames `df_selic_filtered` e `df_fed_funds` têm ambos uma coluna `Date`. Para juntá-los por data, primeiro definimos essa coluna como índice em cada DataFrame (`set_index('Date', inplace=True)`). Em seguida, usamos `join` para mesclá-los com base no índice (datas), usando `how='inner'` para manter apenas datas presentes em ambas as séries (garantindo alinhamento exato mês a mês). O resultado é guardado em `df_rates`, que deverá conter três colunas: `Date`, `Selic` e `Fed_Funds`. Como após a junção a data ficou como índice (herdado da operação de join), usamos `reset_index(inplace=True)` para voltar a tê-la como uma coluna normal. Por fim, imprimimos as primeiras linhas para conferir se a junção ocorreu corretamente.

A saída deverá mostrar colunas `Date`, `Selic` e `Fed_Funds` alinhadas, por exemplo: 

```
Date         Selic   Fed_Funds
2019-01-01   6.40    2.40
2019-02-01   6.40    2.40
2019-03-01   6.40    2.41
... 
``` 

Isso confirma que, e.g., em Jan/2019 Selic=6,40% e Fed Funds=2,40%; nos meses seguintes ambos permanecem constantes por um tempo.

**Interpretação:** O DataFrame `df_rates` consolida as informações de juros do Brasil e EUA, permitindo análise conjunta. Ter os dados lado a lado facilita, por exemplo, o cálculo direto da **correlação** entre as duas séries de juros ao longo do tempo, algo que faremos a seguir. Vale ressaltar que a operação de `inner join` faz com que as datas sejam **apenas aquelas comuns a ambas as séries** – neste caso, ambas são séries mensais completas de 2019 a 2024, então não há perda de informação (todas as datas de mês inteiro de Jan/2019 a Dez/2024 estão presentes em ambas as séries). 

---

## Correlação entre a Selic e a Fed Funds Rate

Com as duas séries de taxas de juros combinadas, podemos investigar **quão fortemente correlacionadas** elas estão. A correlação de Pearson entre Selic e Fed Funds nos dirá se há uma relação linear, e.g., se períodos de alta/baixa de juros nos EUA tendem a coincidir com alta/baixa no Brasil. 

### Cálculo da Correlação e Visualização (Heatmap)

A correlação de Pearson será calculada e apresentada em um formato visual de matriz (embora com apenas duas variáveis, será uma matriz 2x2). O gráfico do tipo *heatmap* mostrará os coeficientes de correlação, facilitando a interpretação.

**Código da Célula 7:** _Cálculo da matriz de correlação e geração de heatmap_  
```python
# Criar um heatmap interativo da correlação entre Selic e Fed Funds Rate
fig = px.imshow(
    df_rates[['Selic', 'Fed_Funds']].corr(),
    text_auto=True,
    color_continuous_scale=["#dbe9f6", "#08306b"],  # Dois tons de azul
    title="Correlação entre Selic e Fed Funds Rate"
)

# Ajustar o layout para aumentar o tamanho da fonte
fig.update_layout(
    title_font_size=20,  # Tamanho da fonte do título
    font=dict(size=14),  # Tamanho da fonte geral
    coloraxis_colorbar=dict(title="Correlação", title_font_size=14, tickfont_size=12)  # Fonte da barra de cores
)

# Exibir o gráfico interativo
fig.show()
``` 

**Explicação:** A expressão `df_rates[['Selic', 'Fed_Funds']].corr()` computa a matriz de correlação de Pearson entre as colunas **Selic** e **Fed_Funds** do DataFrame. Lembre-se que a correlação de Pearson $r_{XY}$ é calculada pela fórmula:

$$ 
r_{XY} = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2 \; \sum_{i=1}^{n}(Y_i - \bar{Y})^2}}, 
$$

onde $X_i$ e $Y_i$ são os valores de Selic e Fed Funds em um mesmo período $i$, $\bar{X}$ e $\bar{Y}$ são suas médias ao longo do tempo, e $n$ é o número de períodos (número de meses de 2019 a 2024). 

O resultado `corr()` é um DataFrame 2x2: correlação de Selic com Selic (que é 1), Selic com Fed_Funds, Fed_Funds com Selic (igual à anterior) e Fed_Funds com Fed_Funds (que é 1). 

Utiliza-se `px.imshow` (Plotly Express) para criar um mapa de calor (*heatmap*) dessa matriz de correlação. Os parâmetros:
- `text_auto=True` exibe os valores numéricos de correlação diretamente nas células do heatmap. 
- `color_continuous_scale=["#dbe9f6", "#08306b"]` define uma escala de cores do azul claro ao azul escuro (valores baixos para próximos de 0, altos próximos de 1). 
- `title` adiciona um título descritivo.

Em seguida, `fig.update_layout` ajusta elementos visuais: aumenta tamanho da fonte do título e dos rótulos, para melhor legibilidade. Finalmente, `fig.show()` renderiza o gráfico interativo. 

**Resultado esperado:** O heatmap apresentará uma pequena matriz 2x2. As diagonais serão 1.0 (correlação perfeita de cada variável consigo mesma). As off-diagonais serão iguais, mostrando o coeficiente de correlação entre Selic e Fed Funds. Supondo que o cálculo produza, por exemplo, ~0.74, veremos esse valor. De fato, ao executar essa célula, o valor calculado foi aproximadamente **0.737**. Visualmente, a célula (Selic, Fed_Funds) estará preenchida por um tom de azul mais escuro indicando correlação positiva alta, e a célula (Fed_Funds, Selic) idêntica (simetria da matriz).

**Interpretação Técnica:** O coeficiente de correlação de Pearson obtido ~**0,74** indica uma **correlação positiva forte** entre as duas taxas de juros. Em outras palavras, há evidência de que quando a **Fed Funds Rate aumenta**, a **Selic tende a aumentar também**, e vice-versa. No contexto de séries econômicas, isso sugere sincronia nas políticas monetárias ou reação de uma à outra. 

Em termos estatísticos, uma correlação de 0,74 (perto de 1) implica que grande parte da variação de uma série está linearmente associada à variação da outra. Entretanto, é importante ressaltar que correlação não implica causalidade: um terceiro fator (por exemplo, condições econômicas globais) pode estar influenciando ambas as taxas.

**Observação:** Com apenas ~72 pontos de dados (6 anos * 12 meses) para calcular essa correlação, é relevante verificar se esse coeficiente é **estatisticamente significativo** – ou seja, se é improvável obter um valor tão alto caso as duas séries fossem não correlacionadas (independentes). A próxima seção realiza precisamente um **teste de hipótese para a correlação de Pearson**, a fim de avaliar sua significância.

### Interpretação do Heatmap de Correlação

O gráfico de correlação produzido (matriz 2x2) pode ser interpretado da seguinte forma:

1. **Valores e Cores:** A diagonal mostra **correlação perfeita (1.0)** de cada variável consigo mesma (destacado em cor mais extrema). O **valor 0.737** exibido na célula cruzando Selic e Fed Funds (e reciprocamente) indica uma **forte correlação positiva** entre as duas taxas de juros. A cor azul relativamente escura reforça visualmente esse valor alto (quanto mais escura dentro da paleta azul, maior o coeficiente). 

2. **Significado do Coeficiente 0.737:** Embora não seja 1.0 (não é uma relação perfeita), 0.74 é considerado alto. Isso significa que as taxas de juros do Brasil e dos EUA **se movem na mesma direção na maior parte do tempo**, porém não em todos os momentos. Há ocasiões em que podem divergir (por isso não é 1.0), mas predominantemente, existe concordância de tendências. 

3. **Contexto Econômico:** Em termos práticos, essa correlação sugere que **ações do Federal Reserve (Fed)** têm influência sobre as decisões do **Banco Central do Brasil (BCB)**. Por exemplo, se o Fed eleva os juros nos EUA, investidores globais podem exigir retornos maiores nos mercados emergentes; para evitar fuga de capital e desvalorização cambial, o BCB tende a subir a Selic. Analogamente, cortes de juros pelo Fed podem abrir espaço para o BCB reduzir a Selic visando estímulo econômico, sem perder tanto atratividade relativa.

Em resumo, o gráfico nos mostra de forma imediata que existe uma **forte associação linear** entre a Selic e a Fed Funds Rate no período analisado. A seguir, quantificaremos a significância estatística dessa correlação e exploraremos outras análises complementares.

---

## Teste de Significância para a Correlação (Teste de Pearson)

Embora tenhamos encontrado um coeficiente de correlação alto entre Selic e Fed Funds, é necessário realizar um **teste de hipótese estatístico** para verificar se essa correlação é significativamente diferente de zero. O teste de correlação de Pearson avalia se a correlação observada poderia ser fruto do acaso, supondo um cenário nulo de correlação inexistente.

**Formulação das Hipóteses:** No teste de Pearson para correlação, definimos:
- **H₀ (Hipótese Nula):** $\rho = 0$ (não há correlação linear verdadeira entre Selic e Fed Funds; qualquer correlação observada é devida a flutuações aleatórias).
- **H₁ (Hipótese Alternativa):** $\rho \neq 0$ (existe correlação linear significativa entre as duas taxas).

Escolhemos um **nível de significância** $\alpha = 0.05$ (5%). Se o valor-p do teste for menor que 0,05, rejeitaremos H₀, inferindo que a correlação observada é estatisticamente significativa.

**Código da Célula 8:** _Cálculo e teste da correlação de Pearson_  
```python
from scipy.stats import pearsonr

# Remover valores nulos para evitar erros
df_rates.dropna(subset=['Selic', 'Fed_Funds'], inplace=True)

# Aplicar o teste de correlação de Pearson
correlation, p_value = pearsonr(df_rates['Selic'], df_rates['Fed_Funds'])

# Exibir resultados
print(f"Correlação de Pearson: {correlation:.4f}")
print(f"Valor-p: {p_value:.6f}")

# Interpretar os resultados
if p_value < 0.05:
    print("➡️ O resultado é estatisticamente significativo. Existe uma correlação entre a Selic e o Fed Funds Rate.")
else:
    print("❌ O resultado NÃO é estatisticamente significativo. Não podemos afirmar que há correlação real entre as taxas.")
``` 

**Explicação:** Utilizamos a função `pearsonr` do SciPy para calcular o coeficiente de Pearson e o valor-p do teste associado. Antes disso, aplicamos `dropna` no DataFrame para garantir que não haja valores nulos (NaNs) nas colunas consideradas, o que poderia causar erro no cálculo (neste caso específico, não deveria haver NaNs porque as séries foram completas e alinhadas, mas é uma precaução válida em geral).

A função `pearsonr(x, y)` retorna dois valores: o coeficiente de correlação de Pearson e o valor-p para a hipótese nula de correlação zero. Já esperávamos o coeficiente ~0.737; o interesse novo é no **valor-p**.

O código imprime ambos. Em seguida, realiza uma interpretação automática simples: se `p_value < 0.05`, exibe uma seta verde indicando significância estatística (rejeição de H₀, ou seja, a correlação é real); caso contrário, exibe um ❌ indicando não significância (não podemos rejeitar a possibilidade da correlação ser nula verdadeira).

**Resultados Obtidos:** A execução produziu, por exemplo: 

```
Correlação de Pearson: 0.7371  
Valor-p: 0.000000  
➡️ O resultado é estatisticamente significativo. Existe uma correlação entre a Selic e o Fed Funds Rate.
```

O valor-p reportado foi extremamente baixo (da ordem de 1e-8 ou menor, aqui exibido como 0.000000 quando formatado com 6 casas decimais). Isso indica que a probabilidade de observarmos uma correlação de 0,737 ou mais em magnitude, caso na realidade não houvesse correlação, é praticamente zero.

**Justificativa Estatística do Teste:** O teste de Pearson baseia-se numa estatística $t$ calculada a partir do coeficiente $r$ e do tamanho amostral $n$:

$$ t = r \sqrt{\frac{n-2}{1-r^2}}, $$

que sob $H_0$ (correlação verdadeira zero) segue aproximadamente uma distribuição $t$ de Student com $n-2$ graus de liberdade ([main.ipynb](file://file-ArUzdDpy7bcF8uatADiE15#:~:text=,Interpretar%20os%20resultados%5Cn)). No nosso caso, $n \approx 72$. Com $r \approx 0,737$, teríamos um valor $t$ bem elevado em módulo, resultando em um valor-p muito pequeno. O SciPy realiza esse cálculo internamente e retorna o valor-p. 

**Interpretação:** Com um **valor-p ≪ 0.05**, rejeitamos H₀. Ou seja, há **evidência estatística forte** de que a correlação verdadeira entre Selic e Fed Funds não é zero – de fato, é significativamente positiva. Em linguagem simples: é extremamente improvável obter uma correlação de ~0,74 se na verdade as séries não tivessem relação nenhuma. Portanto, podemos afirmar com confiança que **quando a Fed Funds Rate se altera, a Selic também se altera de forma correlacionada** (lembrando: o teste não prova causalidade, apenas confirma que a associação linear é robusta, não fruto do acaso). 

Em contexto de Probabilidade e Estatística, acabamos de realizar um **teste bicaudal de significância para coeficiente de correlação**, concluindo que $r \neq 0$. 

---

## Intervalo de Confiança para o Coeficiente de Correlação

Além de testar a significância, é informativo calcular um **intervalo de confiança (IC)** para o coeficiente de correlação. Isso fornece um range de valores plausíveis para a correlação verdadeira na população (aqui, "população" seriam séries de juros em um período maior ou situações similares), dado os dados observados. Vamos calcular um intervalo de confiança de 95% para $\rho$.

**Conceito:** Para calcular um IC da correlação, costuma-se usar a **transformação de Fisher**. Essa transformação define: 

- $z = \operatorname{arctanh}(r)$, também conhecida como $\text{artanh}$ ou $\tanh^{-1}$ (função inversa da tangente hiperbólica), que transforma o coeficiente $r$ (que varia entre -1 e 1) em um valor $z$ aproximadamente normal para grandes $n$. 
- O desvio padrão aproximado de $z$ é $SE_z = 1/\sqrt{n-3}$ (onde $n$ é o número de pares de observações). 
- Então um intervalo de confiança de 95% para $z$ é dado por $[\,z - z_{crit} \cdot SE_z,\; z + z_{crit} \cdot SE_z\,]$, onde $z_{crit}$ é o quantil da distribuição Normal padrão correspondente a 97,5% (para 95% de confiança bilateral). Normalmente, $z_{crit} \approx 1,96$. 
- Finalmente, aplica-se a transformação inversa $\tanh$ aos limites desse intervalo em $z$ para obter os limites em termos de $r$. Isso produz um intervalo de confiança para a correlação.

**Código da Célula 9:** _Cálculo do intervalo de confiança (95%) para $r$_  
```python
# Número de observações
n = len(df_rates.dropna(subset=['Selic', 'Fed_Funds']))

# Correlação e p-valor já calculados antes
correlation, _ = pearsonr(df_rates['Selic'], df_rates['Fed_Funds'])

# Calcular erro padrão da correlação
se = 1 / np.sqrt(n - 3)

# Calcular intervalo de confiança de 95%
z_score = np.arctanh(correlation)  # Conversão para Z-score
z_critical = stats.norm.ppf(0.975)  # Valor crítico para 95%
ci_lower = np.tanh(z_score - z_critical * se)
ci_upper = np.tanh(z_score + z_critical * se)

# Exibir resultados
print(f"Intervalo de Confiança (95%) da Correlação:")
print(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
``` 

**Explicação:** Primeiro determinamos `n`, o número de observações (pares de valores Selic–FedFunds disponíveis). Em seguida, obtemos o valor da correlação calculado (poderíamos reutilizar `correlation` já obtida, mas aqui recalculamos por segurança usando `pearsonr` novamente, ignorando o p-valor devolvido). 

Calculamos o **erro padrão aproximado** `se = 1/√(n-3)`. Essa fórmula vem da teoria assintótica da transformação de Fisher para correlações.

Convertendo o coeficiente em um escore $z$: `z_score = np.arctanh(correlation)`. Aqui, `np.arctanh` implementa $\text{artanh}(r) = \frac{1}{2}\ln\frac{1+r}{1-r}$. 

Calculamos `z_critical` como o quantil 97.5% da distribuição normal padrão usando `stats.norm.ppf(0.975)` (função quantil da normal padrão). O valor obtido é cerca de 1.96 (mais precisamente 1.95996...). 

Então, os limites do IC em escala $z$ são `z_score ± z_critical * se`. Aplicamos `np.tanh` para voltar à escala de $r$, obtendo `ci_lower` e `ci_upper`. Por fim, imprimimos o intervalo de confiança no formato [lower, upper].

**Cálculos Intermediários:** No nosso caso, $n = 72$. Então $SE_z = 1/\sqrt{69} ≈ 0.1204$. Com $r = 0.7371$, obtemos $z = \operatorname{arctanh}(0.7371) ≈ 0.94$. O $z_{crit}$ para 95% é ~1.96. Então:
- $z_{lower} = 0.94 - 1.96*0.1204 ≈ 0.94 - 0.236 < 0.94$ (vamos calcular exatamente a seguir),
- $z_{upper} = 0.94 + 0.236 ≈ 1.176$.
Aplicando $\tanh$ inverso (que é a própria $\tanh$ da forma que usamos, uma vez que aplicamos arctanh anteriormente), obteremos os limites em termos de $r$.

**Resultado Obtido:** A impressão do intervalo de confiança foi: 

```
Intervalo de Confiança (95%) da Correlação:
[0.6095, 0.8275]
```

Interpretando: o intervalo de 95% para a correlação verdadeira $\rho$ está aproximadamente entre **0,6095 e 0,8275**.

**Interpretação Estatística:** Esse intervalo de confiança significa que, com 95% de confiança, a correlação verdadeira entre as duas taxas de juros está entre ~0,61 e ~0,83. Note que **zero não está contido nesse intervalo**, corroborando o resultado do teste de hipótese de que $\rho$ é significativamente diferente de zero ([main.ipynb](file://file-ArUzdDpy7bcF8uatADiE15#:~:text=%22,)). Além disso, o intervalo é relativamente **estreito** (amplitude ~0,22), indicando uma estimativa razoavelmente precisa de $\rho$ – não estamos diante de alta incerteza. Isso se deve em parte ao tamanho amostral decente (72 pontos) e ao valor de correlação alto.

**Conclusão dessa etapa:** Podemos afirmar que a correlação é **positiva e forte**. Mesmo no limite inferior do IC (~0,61), seria considerada uma correlação forte; no limite superior (~0,83), seria uma correlação muito forte. Portanto, os dados sugerem fortemente uma relação linear substantiva entre Selic e Fed Funds no período analisado.

---

## Análise da Variação Anual da Selic – Teste ANOVA e Boxplot

Tendo examinado a relação entre a Selic e a taxa americana, voltamos o foco para a **Selic ao longo do tempo**. É de interesse avaliar se **houve mudanças significativas da Selic entre anos diferentes** no período 2019–2024. Em outras palavras, queremos verificar estatisticamente se a Selic de um ano para outro se manteve no mesmo patamar ou se variou de forma relevante.

Para isso, utilizaremos uma **ANOVA de uma via** (one-way ANOVA), onde:
- Os fatores são os **anos** (2019, 2020, 2021, 2022, 2023, 2024).
- As observações são os valores mensais da Selic dentro de cada ano.

A hipótese nula H₀ é que **todas as médias anuais da Selic são iguais**. A hipótese alternativa H₁ é que **pelo menos um ano tem média de Selic diferente**. Em termos práticos, H₀ postula que não houve mudança significativa de nível da Selic entre anos (qualquer variação mensal seria apenas ruído dentro de um mesmo nível médio), enquanto H₁ postula que houve alteração (por exemplo, a Selic de 2020 difere significativamente da de 2019, etc.).

Usaremos $\alpha = 0.05$ novamente.

### Preparação dos Dados por Ano

Para realizar a ANOVA, primeiro reestruturamos os dados da Selic adicionando uma coluna de ano e agrupando os valores por ano.

**Código da Célula 10:** _Preparação dos grupos anuais da Selic_  
```python
# ==========================
# 1️⃣  PREPARAÇÃO DOS DADOS
# ==========================
# Criar coluna com o ano para segmentação
df_rates['Ano'] = df_rates['Date'].dt.year

# Filtrar dados para garantir que só temos anos completos
anos_disponiveis = df_rates['Ano'].unique()
grupos_selic = [df_rates[df_rates['Ano'] == ano]['Selic'].dropna() for ano in anos_disponiveis]

# ==========================
# 2️⃣  TESTE ANOVA
# ==========================
stat, p_value = f_oneway(*grupos_selic)
``` 

**Explicação:** Primeiro, criamos uma nova coluna `Ano` em `df_rates`, extraindo o ano da data (`df_rates['Date'].dt.year`). Isso rotula cada registro mensal com seu respectivo ano (2019, 2020, ..., 2024). 

Em `anos_disponiveis` coletamos os valores únicos de anos presentes (deverá ser array [2019, 2020, 2021, 2022, 2023, 2024]). Em seguida, construímos `grupos_selic`, que é uma lista de séries, onde cada série contém os valores da Selic correspondentes a um determinado ano. Essa compreensão de lista `[... for ano in anos_disponiveis]` itera por cada ano e seleciona `df_rates['Selic']` para aquele ano, usando `.dropna()` caso algum valor de Selic seja NaN (não esperado, pois Selic tem valor todo mês). Essencialmente, `grupos_selic` é uma lista de seis conjuntos de dados (um para cada ano) com ~12 valores cada (12 meses por ano, embora 2024 inclua até dezembro, totalizando 72 meses no geral, possivelmente 2024 incompleto até certa data, mas aqui supõe-se até 2024 inteiro).

Em seguida, aplicamos `f_oneway(*grupos_selic)` da biblioteca SciPy. O operador `*` desembrulha a lista de grupos passando cada grupo como um argumento separado para `f_oneway`. A função retorna a **estatística F** e o **valor-p** do teste ANOVA de uma via, testando H₀: todas as médias dos grupos são iguais. Armazenamos esses resultados em `stat` (F) e `p_value`.

Até aqui, nenhum resultado foi impresso. A variável `stat` contém $F_{\text{calculado}}$ e `p_value` o valor-p correspondente.

### Execução do Teste ANOVA e Visualização dos Resultados

Agora interpretaremos os resultados numéricos do teste e também plotaremos um **boxplot** (diagrama de caixas) para visualizar a distribuição da Selic em cada ano, o que ajuda a evidenciar diferenças.

**Código da Célula 11:** _Resultado do ANOVA e boxplot por ano_  
```python
# ==========================
# 3️⃣  INTERPRETAÇÃO AUTOMÁTICA
# ==========================
significancia = 0.05
if p_value < significancia:
    resultado = "A diferença entre os anos é estatisticamente significativa. Ou seja, a Selic variou de forma relevante ao longo dos anos."
else:
    resultado = "Não há evidências estatísticas suficientes para afirmar que houve mudanças significativas na Selic entre os anos."

# ==========================
# 4️⃣  VISUALIZAÇÃO INTERATIVA (com ajuste de proporção)
# ==========================
fig = px.box(df_rates, x='Ano', y='Selic', title='Distribuição da Selic ao longo dos anos')

# Ajustar proporções do gráfico para evitar que fique muito retangular
fig.update_layout(
    width=1900,  # Largura ajustada
    height=800,  # Altura reduzida para melhor proporção
    margin=dict(l=40, r=40, t=60, b=40)  # Margens ajustadas
)

# ==========================
# 5️⃣  EXIBIR RESULTADOS
# ==========================
print(f"F-Estatística: {stat:.4f}")
print(f"Valor-p: {p_value:.4f}")
print(f"Conclusão: {resultado}")

fig.show()
``` 

**Explicação:** Este bloco está dividido em seções comentadas para clareza:

- **Interpretação Automática:** Define `significancia = 0.05` e compara o `p_value` obtido com esse nível. Se `p_value < 0.05`, define a string `resultado` indicando que “a diferença entre os anos é estatisticamente significativa, ou seja, a Selic variou de forma relevante ao longo dos anos.” Se não, define `resultado` dizendo que “não há evidências de mudanças significativas entre anos.” Isso automatiza a conclusão estatística baseada no teste.

- **Visualização Interativa:** Cria um gráfico de caixa (*boxplot*) usando Plotly Express (`px.box`). Colocamos no eixo x o `Ano` e no eixo y a `Selic`. Assim, cada ano terá uma caixa que representa a distribuição dos valores mensais da Selic naquele ano. O título descreve o gráfico. Depois, ajustamos o layout: definimos uma largura grande (1900 px) e altura (800 px) para o gráfico, o que melhora a legibilidade principalmente se houver muitos anos no eixo x, e ajustamos margens. Esses parâmetros evitam que o boxplot fique “achatado” ou sobreposto.

- **Exibir Resultados:** Imprime a estatística F e o valor-p com 4 casas decimais, e imprime a conclusão (a frase armazenada em `resultado`). Em seguida, `fig.show()` exibe o gráfico boxplot interativo.

**Resultados ANOVA:** Os valores calculados foram, por exemplo: 

```
F-Estatística: 149.5436  
Valor-p: 0.0000  
Conclusão: A diferença entre os anos é estatisticamente significativa. Ou seja, a Selic variou de forma relevante ao longo dos anos.
``` 

A estatística $F \approx 149.54$ com um valor-p praticamente 0 (menor que 1e-30) indica uma diferença altamente significativa entre as médias de pelo menos dois anos ([main.ipynb](file://file-ArUzdDpy7bcF8uatADiE15#:~:text=%22,n)). 

**Interpretação Estatística do ANOVA:** A estatística $F$ é dada pela razão entre a variância média *entre os grupos* e a variância média *dentro dos grupos*. Intuitivamente, um valor de $F = 149$ é imenso, sugerindo que as diferenças entre as médias anuais são muito maiores que a variação dos valores mensais dentro de cada ano. Formalmente, para 6 grupos (anos) e $N=72$ observações no total, os graus de liberdade seriam $df_{\text{entre}} = 6-1 = 5$ e $df_{\text{dentro}} = 72-6 = 66$. Um $F(5,66)$ de 149 implica um valor-p extremamente pequeno, confirmando que podemos rejeitar H₀. 

Em termos de fórmula, a estatística $F$ calculada é:

$$
F = \frac{\text{SQ}_{\text{entre}}/(k-1)}{\text{SQ}_{\text{dentro}}/(N-k)},
$$

onde $k=6$ anos e $N=72$ dados ([main.ipynb](file://file-ArUzdDpy7bcF8uatADiE15#:~:text=%22,n)). Um valor tão alto de $F$ implica que $\text{SQ}_{\text{entre}} \gg \text{SQ}_{\text{dentro}}$, ou seja, as médias anuais diferem muito comparadas à variabilidade interna de cada ano.

**Visualização (Boxplot):** O boxplot gerado permite enxergar quais anos diferem e como:
- Cada “caixa” mostra o intervalo interquartil (Q1 a Q3) dos valores mensais da Selic no ano, a linha dentro da caixa é a mediana anual, e os "whiskers" (extremidades) podem indicar valores mínimo/máximo (nesse caso, com 12 pontos por ano, possivelmente até os extremos reais ou 1.5x IQR se houver outliers). 
- Visualmente espera-se ver, por exemplo: **2019 e 2020** com caixas em patamares **bem diferentes** (2019 mais alto, 2020 bem baixo, pois a Selic caiu de ~6.5% em 2019 para ~2% em 2020). **2021** possivelmente ainda baixo (Selic começou 2021 baixa e subiu no fim do ano). **2022 e 2023** com caixas em patamar mais alto (Selic subiu para ~13.75% em 2022 e ficou alta em 2023). **2024** possivelmente similar ou ligeiramente abaixo (caso tenha começado a cair no fim de 2023/24). 
- Essas diferenças de nível seriam claramente visíveis: caixas de 2019 e 2020 longe uma da outra, 2022 e 2023 bem acima de 2020–21, etc.

**Interpretação dos Resultados ANOVA:** De acordo com o teste:
- Rejeitamos H₀: **p < 0,05** (na verdade p ≈ 0). Portanto, **pelo menos um dos anos tem média de Selic distinta**. Dado o contexto, é na verdade mais que um: a Selic passou por movimentos consideráveis. 
- A **conclusão automática** resume: “a Selic variou de forma relevante ao longo dos anos”. Isso confirma quantitativamente o que a história econômica sugere – houve mudanças significativas na política monetária de 2019 a 2024.

### Interpretação Estatística e Prática das Diferenças Anuais

**Estatística:**  
- **Resultado do Teste:** $F(5,66) \approx 149,54$, valor-p $\approx 0$. Portanto, ao nível de 5%, a diferença entre as médias anuais da Selic é **estatisticamente significativa** ([main.ipynb](file://file-ArUzdDpy7bcF8uatADiE15#:~:text=%22,n)). 
- Isso significa que não se trata apenas de flutuações aleatórias mensais: houve **quebras de patamar** de juros entre alguns anos. Especificamente, a estatística tão alta indica diferenças marcantes – por exemplo, a média de 2020 é muito menor que a de 2019, e a média de 2022 muito maior que a de 2021, etc. 

**Prática (Contexto Econômico):**  
- **Queda Forte (2020–2021):** Em 2020, diante da pandemia de COVID-19, o Banco Central do Brasil reduziu drasticamente a Selic para estimular a economia. Isso explica um nível **significativamente menor** em 2020 (mediana da caixa de 2020 por volta de 2%) comparado a 2019 (mediana ~6.5%). 2021 também teve Selic baixa na primeira metade e iniciando alta na segunda, mas em média ainda abaixo de 2019.  
- **Alta Expressiva (2022–2023):** A partir de 2021 e mais intensamente em 2022, houve um ciclo de alta de juros para conter a inflação pós-pandemia. Assim, 2022 apresentou média de Selic muito superior a 2020–21 (por volta de dois dígitos percentuais), e 2023 manteve juros altos. Essas diferenças criam caixas em níveis superiores no boxplot.  
- **Impactos:** Essa variação de juros tem efeitos profundos: custo do crédito para famílias e empresas, atratividade de investimentos em renda fixa, comportamento do câmbio e da bolsa de valores, etc. Um teste ANOVA significativo nos dá respaldo estatístico para afirmar que “2020 não foi igual a 2019”, “2022 não foi igual a 2021”, etc., em termos de política monetária.

Em suma, a ANOVA confirma que a **política monetária brasileira variou significativamente ano a ano no período**, o que é consistente com os eventos econômicos recentes. Visualmente, o **boxplot** reforça essa evidência mostrando, por exemplo, 2020 e 2021 como anos de juros muito baixos, em contraste com 2022 e 2023 de juros altos. Essa análise de variância nos fornece uma compreensão quantitativa robusta das **mudanças de regime da Selic**.

---

## Correlação entre Variações da Selic e Retornos de Ações Industriais

Após analisar a correlação entre as taxas de juros de Brasil e EUA e a evolução da Selic ao longo dos anos, o notebook explora como **as variações da Selic podem se relacionar com o desempenho de ações** de empresas industriais brasileiras. A ideia aqui é verificar se existe correlação linear entre a **variação mensal da Selic** e o **retorno mensal de determinadas ações**. Além disso, identifica-se quais ações apresentam maior ou menor correlação com os movimentos da Selic.

### Cálculo da Variação Mensal da Selic

Primeiro, computamos a **variação percentual** mês a mês da Selic. Ou seja, de cada mês para o próximo, qual a taxa de mudança em %.

**Código da Célula 12:** _Cálculo da variação mensal da Selic_  
```python
# Garantir que a coluna 'Date' esteja em formato datetime
df_rates['Date'] = pd.to_datetime(df_rates['Date'])

# Calcular a variação percentual da Selic mês a mês
df_rates['Variação_Selic'] = df_rates['Selic'].pct_change() * 100
``` 

**Explicação:** Garantimos que `df_rates['Date']` é do tipo datetime (isso já era, mas por precaução se reforça). Em seguida, usamos `pct_change()` na coluna Selic. Essa função calcula $\frac{\text{Selic}_{t} - \text{Selic}_{t-1}}{\text{Selic}_{t-1}}$ para cada mês $t$, ou seja, o retorno (ou variação proporcional) de um mês para o seguinte. Multiplicamos por 100 para expressar em **percentual**. O resultado é colocado em uma nova coluna `Variação_Selic`. 

Note que para o primeiro ponto (Jan/2019) não há mês anterior no dataset filtrado, então `Variação_Selic` ali será `NaN`. Todos os outros meses terão um valor (positivo, negativo ou zero). Por exemplo, se a Selic passou de 6.4% em jan/2019 para 6.4% em fev/2019, a variação será 0%; se caiu para 4.5% em certo mês, a variação será negativa etc.

**Interpretação:** Essa coluna representa **quanto a taxa Selic variou a cada mês** em relação ao mês anterior, em termos percentuais relativos ao valor anterior. Assim, podemos correlacionar esse percentual de mudança com os **retornos das ações** (que também são variações percentuais mensais dos preços). 

### Cálculo dos Retornos Mensais das Ações e Correlações com a Selic

Agora, calculam-se os **retornos mensais** de cada ação (percentual de variação de preço mês a mês) e cruza-se com a variação da Selic para examinar correlações. Serão geradas correlações para cada uma das 15 ações com a coluna `Variação_Selic`.

**Código da Célula 13:** _Retornos das ações e correlações com Selic_  
```python
from scipy.stats import pearsonr
import plotly.express as px

# Calcular retorno mensal das ações (em %)
returns = stock_data.set_index('Date').pct_change().dropna() * 100

# Calcular variação percentual da Selic
df_rates['Variação_Selic'] = df_rates['Selic'].pct_change() * 100
selic_var = df_rates[['Date', 'Variação_Selic']].set_index('Date')

# Juntar retornos com variação da Selic
combined = returns.join(selic_var, how='inner')

# Calcular correlação de cada ação com a variação da Selic
correlacoes = {}
for ticker in returns.columns:
    correlacao, _ = pearsonr(combined[ticker], combined['Variação_Selic'])
    correlacoes[ticker] = correlacao

# Selecionar as 5 ações mais correlacionadas com a Selic
top_5_tickers = pd.Series(correlacoes).sort_values(ascending=False).head(15).index.tolist()

# Criar DataFrame com preços dessas ações + Selic + Fed Funds
df_top5 = stock_data[['Date'] + top_5_tickers].copy()
df_taxas = df_rates[['Date', 'Selic', 'Fed_Funds']]
df_plot = pd.merge(df_top5, df_taxas, on='Date', how='inner')

# Reestruturar para gráfico interativo
df_long = df_plot.melt(id_vars='Date', var_name='Variável', value_name='Valor')

# Gráfico interativo
fig = px.line(df_long, x='Date', y='Valor', color='Variável',
              title='Evolução de Preço das Ações mais Correlacionadas com Selic vs Selic e Fed Funds',
              labels={'Valor': 'Valor (%)', 'Date': 'Data'})

# Criar DataFrame de correlação com ranking
df_correlacao = pd.DataFrame.from_dict(correlacoes, orient='index', columns=['Correlação_Selic'])
df_correlacao = df_correlacao.sort_values(by='Correlação_Selic', ascending=False).reset_index()
df_correlacao.columns = ['Ticker', 'Correlação_Selic']
df_correlacao['Ranking'] = df_correlacao['Correlação_Selic'].rank(ascending=False).astype(int)

# Visualizar o ranking
fig_corr = px.bar(df_correlacao, x='Ticker', y='Correlação_Selic',
                  title='Ranking de Correlação com a Variação da Selic (%)',
                  text='Ranking',
                  color='Correlação_Selic',
                  color_continuous_scale='RdBu')

fig_corr.update_traces(textposition='outside')
fig_corr.update_layout(height=900, width=1900)
fig_corr.show()

# Se quiser apenas o DataFrame tabular também:
df_correlacao

#fig.update_layout(height=500, width=1000)
#fig.show()
``` 

**Explicação:** Vamos por partes:

- **Retornos das ações:** `returns = stock_data.set_index('Date').pct_change().dropna() * 100`. Aqui pegamos o DataFrame original de preços `stock_data` (que tem colunas Date e os preços das ações) e colocamos Date como índice. Depois calculamos `pct_change()` ao longo do índice (isto é, de mês a mês) para todas as colunas de preço, e eliminamos a primeira linha que será NaN (usando `dropna()`, pois para Jan/2019 não há anterior). Multiplicamos por 100 para obter percentuais. O resultado `returns` é um DataFrame onde cada coluna é uma ação e os valores são **retornos mensais percentuais dessa ação**. Por exemplo, se WEGE3 subiu de R\$X para R\$Y em um mês, a coluna WEGE3 para aquele mês conterá $(Y-X)/X * 100$.

- Já tínhamos calculado `Variação_Selic` antes; aqui novamente (por redundância) calcula e cria `selic_var` que é um DataFrame com Date index e a coluna Variação_Selic. 

- **Junção dos retornos com variação Selic:** `combined = returns.join(selic_var, how='inner')` une o DataFrame de retornos das ações com o DataFrame de variação da Selic, casando pelas datas (índice). `how='inner'` garante que consideraremos apenas datas presentes em ambos (basicamente 2019-02 até 2024-12, excluindo 2019-01 que não tem variação Selic). `combined` terá colunas de retornos de todas as ações e a coluna `Variação_Selic`, para cada mês.

- **Correlação de cada ação com Selic:** Inicializamos um dicionário `correlacoes`. Iteramos por cada `ticker` (coluna) em `returns.columns` (lista de ações). Para cada, calculamos `pearsonr(combined[ticker], combined['Variação_Selic'])`. Isso dá o coeficiente de correlação entre os retornos mensais daquela ação e a variação mensal da Selic, junto com um valor-p (que não usamos aqui, focamos só na correlação). Armazenamos o coeficiente em `correlacoes[ticker]`. Após o loop, teremos um dicionário com 15 entradas do tipo `{'WEGE3.SA': r1, 'KEPL3.SA': r2, ... }`.

- **Seleção das top correlações:** `top_5_tickers = pd.Series(correlacoes).sort_values(ascending=False).head(15).index.tolist()`. Aqui eles pediram "top 5", mas usaram 15 (provavelmente queriam ordenar todos; de fato head(15) pega todos no nosso caso, pois são 15 ações). Essencialmente, isso ordena as ações pelas correlações da maior para a menor e pega os 15 primeiros (que é a lista completa, ordenada da maior correlação para a menor). `top_5_tickers` acaba contendo todos os tickers ordenados decrescentemente por correlação com Selic.

- **Preparar dados para gráfico de linhas:** Eles criam `df_top5` contendo a Date e os preços das ações selecionadas (aqui, todas as 15). Depois `df_taxas` com Date, Selic e Fed_Funds. Fazem um merge para ter `df_plot` com Date, preços das top ações e as taxas Selic e FedFunds juntos. Em seguida, transformam para formato longo (`df_long = df_plot.melt(...)`) para facilitar plotagem multivariada: isso cria uma coluna `Variável` com o nome da série (ex: "WEGE3.SA" ou "Selic") e `Valor` com o respectivo valor, e repete as datas adequadamente.

- **Gráfico de linhas (`fig`)**: Criam `fig = px.line` para plotar todas essas séries (as ações mais correlacionadas + Selic + Fed Funds) ao longo do tempo na mesma figura. Isso daria um gráfico com muitas linhas (15 ações + 2 taxas = 17 linhas). Esse gráfico comparativo pode mostrar visualmente tendências conjuntamente, embora fique bem carregado.

- **DataFrame de correlação para ranking:** Convertem o dicionário `correlacoes` num DataFrame `df_correlacao` com colunas Ticker e Correlação_Selic. Ordenam decrescente por correlação e adicionam uma coluna de ranking (convertendo a ordem em números 1 a 15). 

- **Gráfico de barras do ranking:** `fig_corr = px.bar(...)` plota um gráfico de barras dos tickers vs sua correlação com Selic. As barras são coloridas conforme o valor de correlação (escala vermelha-azul `RdBu`, onde possivelmente vermelho para correlações negativas, azul para positivas). O texto acima de cada barra é o ranking (1 para maior correlação, 15 para menor). Atualiza o layout para altura 900, largura 1900 (gráfico grande e legível), e exibe com `fig_corr.show()`.

- Por fim, eles mostraram o DataFrame `df_correlacao` (que será impresso como tabela) e comentaram a exibição do gráfico de linhas (talvez para focar só no ranking).

**Resultados de Correlações:** Analisando o objeto `df_correlacao` exibido, vemos uma lista dos tickers com suas correlações com a variação da Selic e o ranking. Os valores calculados (de acordo com a saída bruta capturada) são aproximadamente:

- Maior correlação: **LOGN3.SA ≈ 0.151** (15.1%) – correlação positiva fraca.
- Em seguida: SHUL4.SA ~0.145, KEPL3.SA ~0.083, GOLL4.SA ~0.081, RAIL3.SA ~0.051, VLID3.SA ~0.037, TASA3.SA ~0.013, TUPY3.SA ~0.002 (praticamente 0), 
- Depois correlações **negativas**: EMBR3.SA ≈ -0.032, AZUL4.SA -0.074, JSLG3.SA -0.077, RAPT4.SA -0.096, POMO4.SA -0.132, ROMI3.SA -0.192, e a mais negativa **WEGE3.SA ≈ -0.286**.

Portanto, algumas ações tiveram leve correlação **positiva** com a variação da Selic (ou seja, seus retornos sobem um pouco quando a Selic sobe), enquanto outras tiveram correlação **negativa** (retornos caem quando Selic sobe, o que é intuitivo para muitas empresas, já que juros maiores podem encarecer crédito e diminuir lucro, afetando negativamente ações, especialmente empresas que dependem de financiamento ou consumo).

**Interpretação das Correlações Retornos vs Selic:**

- **Magnitudes Baixas:** Os coeficientes encontrados estão todos em magnitude **baixa** (menores que 0,3 em valor absoluto). Isso indica que **nenhuma dessas ações tem uma correlação forte com as variações mensais da Selic**. A maioria está perto de zero (entre -0.1 e 0.1), sugerindo pouca relação linear no curto prazo entre movimentos da Selic e retornos mensais dessas empresas. 

- **Sinal Positivo:** Algumas (LOGN3, SHUL4, KEPL3, etc.) têm coeficiente levemente **positivo**. Isso pode indicar que, nos meses em que a Selic subiu (ou seja, política monetária contracionista), essas ações tiveram *ligeiros ganhos*. Pode ser contraintuitivo, mas pode ocorrer se, por exemplo, a alta de juros for interpretada como controle da inflação beneficiando algumas empresas de setores específicos, ou devido a outros fatores coincidentes. Porém, dado que são correlações fracas, provavelmente não há um efeito muito consistente.

- **Sinal Negativo:** Outras ações (como WEGE3, ROMI3, etc.) mostram **correlação negativa** com variação da Selic. Ou seja, quando a Selic sobe, tendem a ter retornos negativos (e vice-versa). Isso faz sentido para empresas que sofrem com juros altos (WEGE3, por exemplo, WEG, pode ser afetada por menor investimento ou dólar – há múltiplos fatores). WEGE3 ter -0.285 sugere que houve uma tendência modesta de cair em meses de alta de juros (mas -0.28 ainda é uma correlação fraca/moderada). 

- **Ranking e Extremidades:** O ranking mostra **LOGN3 (Log-In Logística)** com a correlação positiva mais alta (~0.15) e **WEGE3 (WEG S.A.)** com a mais negativa (~-0.29). Os demais se espalham nesse intervalo. Em geral, valores absolutos abaixo de 0.3 indicam **correlações baixas**, possivelmente não significativas estatisticamente (um teste de hipótese individual para esses $r$ provavelmente falharia em rejeitar H₀ para a maioria, dado n ~71 meses e r ~0.15 tem p perto de 0.2, etc., mas esse teste não foi mostrado).

**Visualização (Gráfico de Barras):** O gráfico de barras produzido (`fig_corr`) facilita ver essa ordenação. Deve mostrar barras ligeiramente acima do zero para as primeiras (em azul claro, correspondendo a correlações positivas fracas) e barras abaixo de zero (possivelmente em tom avermelhado, dado o esquema RdBu) para as últimas. Os valores numéricos possivelmente rotulados reforçam que o maior ~0.15 e o menor ~-0.29.

**Conclusão dessa análise:** Não encontramos **correlações fortes** entre variações mensais da Selic e retornos das ações industriais. Isso sugere que, no curto prazo, os preços das ações são influenciados por muitos outros fatores além da simples variação dos juros (por exemplo, resultados financeiros, notícias setoriais, tendências de mercado global, etc.). Ainda assim, podemos destacar:
- Algumas empresas de logística e bens de capital tiveram leve correlação positiva – possivelmente beneficiando-se de cenários de alta de juros ou outras condições correlacionadas.
- Empresas como WEG (bem de capital/exportadora) tiveram correlação negativa – possivelmente porque juros mais altos implicam real mais forte ou menor atividade industrial, afetando-as negativamente.
- **Importante:** As correlações sendo baixas também implicam que **diversificação setorial** e outros fatores podem atenuar o impacto das políticas monetárias nos retornos de ações individuais.

Este tipo de análise pode ser estendido a mais setores ou usando modelos mais complexos (regressões multivariadas) para entender melhor a relação entre mercado acionário e juros, mas dentro do escopo deste notebook ficamos na inspeção de correlações simples.

---

## Conclusão

Neste trabalho, documentamos detalhadamente uma análise focada em **Probabilidade e Estatística** aplicada a dados financeiros:

- **Correlação Selic vs Fed Funds:** Identificamos um coeficiente de correlação de aproximadamente **0,74** entre as taxas de juros do Brasil e dos EUA (2019–2024), indicando uma forte correlação positiva. Testes estatísticos (teste de Pearson) confirmaram que essa correlação é **significativamente diferente de zero** (valor-p praticamente zero), e um intervalo de confiança de 95% [0,61; 0,83] reforçou que a correlação verdadeira é alta e positiva. Essa constatação está alinhada com expectativas econômicas: políticas monetárias de economias emergentes frequentemente refletem (ou respondem a) as do Fed em alguma medida.

- **Variação da Selic por Ano:** A ANOVA de uma via mostrou diferenças estatisticamente significativas na **taxa Selic entre anos distintos**. Em particular, identificamos mudanças de patamar: juros bem mais baixos em 2020–2021 em comparação a 2019, e juros bem mais altos em 2022–2023 em comparação aos anos anteriores. O teste F muito elevado e o valor-p nulo confirmam que essas variações não foram aleatórias; elas refletem mudanças de política monetária deliberadas e substanciais. O boxplot anual ilustrou essas diferenças de forma visual.

- **Correlação Variação Selic vs Retorno de Ações:** As correlações calculadas entre a **variação mensal da Selic** e os **retornos de 15 ações industriais brasileiras** foram, em sua maioria, próximas de zero, variando de +0,15 a -0,29. Isso indica que, no horizonte mensal, **não há uma relação linear forte entre movimentos da Selic e os retornos dessas ações**. Em outras palavras, conhecer se a Selic subiu ou caiu em um mês dá pouca informação direta sobre se essas ações subiram ou caíram no mesmo mês. Fatores idiossincráticos de cada empresa e outras variáveis macroeconômicas podem estar diluindo essa relação. 

Em termos didáticos, este notebook exemplificou o uso de **ferramentas estatísticas** (como cálculo de correlações, testes de hipóteses e ANOVA) em um contexto real, interpretando resultados numéricos à luz de conceitos econômicos. Os resultados obtidos enfatizam a importância de testar hipóteses (para não tirar conclusões precipitadas de uma correlação aparente) e de quantificar a incerteza (usando intervalos de confiança). 

De forma geral, conclui-se que: 
- A conexão entre as políticas monetárias do Brasil e dos EUA foi forte no período recente, 
- A política monetária doméstica variou drasticamente ao longo dos anos recentes (confirmando nossa compreensão qualitativa dos eventos econômicos), 
- E as reações do mercado acionário doméstico a essas variações de juros, ao menos no segmento industrial e no curto prazo, não seguiram um padrão linear simples. 

Essa documentação procurou abranger tanto o **como** (execução do código e procedimentos estatísticos) quanto o **porquê** (razões teóricas e interpretações práticas) de cada etapa, proporcionando um material de estudo compreensível para estudantes de Probabilidade e Estatística interessados em aplicações financeiras.
