# EmpregaData — Projeto Integrador IV

Aplicação de análise e classificação da demanda do mercado de trabalho brasileiro usando dados do CAGED.

## Melhorias desta versão
- previsão do **mês seguinte**, em vez de classificar o próprio mês;
- divisão temporal: treinamento até dez/2025 e teste em jan-jun/2026;
- uso da taxa líquida de emprego (saldo/estoque), tornando setores e UFs mais comparáveis;
- dois modelos Random Forest: um para setores e outro para UFs;
- recomendação combinada: 60% sinal do setor + 40% sinal do estado;
- dashboard com visão geral, análise de área, ranking de oportunidades e metodologia do modelo;
- gráficos históricos e matrizes de confusão;
- interface Streamlit.

## Como executar

No terminal, dentro da pasta do projeto:

```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run src/app.py
```

## Interpretação
O sistema estima a **tendência de demanda para o próximo mês** a partir do comportamento histórico. A demanda é classificada em Baixa, Média ou Alta. O resultado é uma ferramenta acadêmica de apoio à exploração do mercado de trabalho e não uma garantia de contratação.

## Validação
Para a validação com pessoas reais, peça aos participantes para testar a aplicação e responder um questionário sobre facilidade, clareza, utilidade e confiança percebida.
