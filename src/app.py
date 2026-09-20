
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; MODELS=ROOT/"models"

pack=joblib.load(MODELS/"modelos_demanda.joblib")
sec_model=pack["sector_model"]; uf_model=pack["uf_model"]
sec_features=pack["sector_features"]; uf_features=pack["uf_features"]
last_date=pd.Timestamp(pack["last_date"])

sector_hist=pd.read_csv(DATA/"historico_setores.csv",parse_dates=["date"])
uf_hist=pd.read_csv(DATA/"historico_ufs.csv",parse_dates=["date"])
t1=pd.read_csv(DATA/"analise_setores.csv")
t2=pd.read_csv(DATA/"analise_ufs.csv")
sec_fc=pd.read_csv(DATA/"previsoes_setores_jul2026.csv")
uf_fc=pd.read_csv(DATA/"previsoes_ufs_jul2026.csv")
metrics=json.loads((DATA/"metricas_modelos.json").read_text(encoding="utf-8"))

REGION_MAP=pack["region_map"]
st.set_page_config(page_title="EmpregaData",page_icon="📊",layout="wide")

st.title("📊 EmpregaData")
st.subheader("Inteligência de dados aplicada ao mercado de trabalho")
st.caption("Projeto Integrador IV • CAGED • dados até junho de 2026")

with st.sidebar:
    st.header("Menu")
    page=st.radio("Escolha uma página",["🏠 Visão geral","🔎 Analisar área","🚀 Encontrar oportunidades","🤖 Sobre o modelo"])
    st.divider()
    st.caption("Fonte: arquivo CAGED fornecido pelo projeto.")

def pct(x): return f"{x:.1%}"
def fmt_int(x): return f"{int(round(x)):,}".replace(",",".")

if page=="🏠 Visão geral":
    st.markdown("## Como está o mercado de trabalho?")
    c1,c2,c3,c4=st.columns(4)
    # Total values from Table 1 June 2026
    # Os indicadores nacionais vêm da linha Brasil de analise_ufs.csv.
    # Não somar analise_setores.csv, pois ela contém categorias agregadas
    # e subsetores, o que causaria dupla contagem.
    brasil = t2[t2["uf"] == "Brasil"].iloc[0]
    total_adm = float(brasil["adm_jun"])
    total_des = float(brasil["desl_jun"])
    total_saldo = float(brasil["saldo_jun"])
    c1.metric("Admissões — jun/2026",fmt_int(total_adm))
    c2.metric("Desligamentos — jun/2026",fmt_int(total_des))
    c3.metric("Saldo — jun/2026",fmt_int(total_saldo))
    total_saldo_ano = float(brasil["saldo_ano"])
    c4.metric("Saldo acumulado — 2026", fmt_int(total_saldo_ano))

    st.markdown("### 📈 Saldo por setor")
    top=t1.sort_values("saldo_jun",ascending=False).head(10)
    fig=px.bar(top,x="saldo_jun",y="setor",orientation="h",text_auto=True,
               labels={"saldo_jun":"Saldo","setor":"Setor"})
    fig.update_layout(yaxis={"categoryorder":"total ascending"},height=500)
    st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        topuf=t2.sort_values("saldo_jun",ascending=False).head(10)
        fig=px.bar(topuf,x="saldo_jun",y="uf",orientation="h",text_auto=True,
                   labels={"saldo_jun":"Saldo","uf":"Estado"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        region=t2.copy(); region["regiao"]=region.uf.map(REGION_MAP)
        region=region.groupby("regiao",as_index=False).saldo_jun.sum().sort_values("saldo_jun",ascending=False)
        fig=px.bar(region,x="regiao",y="saldo_jun",text_auto=True,labels={"saldo_jun":"Saldo","regiao":"Região"})
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("### 📊 Evolução do saldo — Brasil")
    # Build national history directly from UF history, summing states
    br=uf_hist.groupby("date",as_index=False).agg(saldo=("Saldos","sum"),estoque=("Estoque","sum"))
    br["taxa_liquida"]=br.saldo/br.estoque
    fig=px.line(br,x="date",y="saldo",markers=True,labels={"date":"Mês","saldo":"Saldo"})
    fig.update_layout(height=420)
    st.plotly_chart(fig,use_container_width=True)

elif page=="🔎 Analisar área":
    st.markdown("## 🔎 Analisar uma área")
    regioes=sorted(set(REGION_MAP.values()))
    regiao=st.selectbox("Região",regioes)
    estados=sorted([u for u in t2.uf.unique() if REGION_MAP.get(u)==regiao])
    uf_sel=st.selectbox("Estado",estados)
    setores=sorted(sector_hist.setor.unique())
    setor_sel=st.selectbox("Área / setor econômico",setores)

    sf=sec_fc[sec_fc.setor==setor_sel].iloc[0]
    ufrow=uf_fc[uf_fc.uf==uf_sel].iloc[0]
    probs={c:0.6*float(sf[f"prob_{c}"])+0.4*float(ufrow[f"prob_{c}"]) for c in ["Baixa","Média","Alta"]}
    pred=max(probs,key=probs.get)
    saldo=float(t1.loc[t1.setor==setor_sel,"saldo_jun"].iloc[0])

    c1,c2,c3=st.columns(3)
    c1.metric("Saldo do setor — jun/2026",fmt_int(saldo))
    icon={"Alta":"🟢","Média":"🟡","Baixa":"🔴"}[pred]
    c2.metric("Demanda estimada para jul/2026",f"{icon} {pred}")
    c3.metric("Probabilidade da classe",pct(probs[pred]))

    st.markdown("### Probabilidades combinadas")
    p_df=pd.DataFrame({"Nível":["Baixa","Média","Alta"],"Probabilidade":[probs["Baixa"],probs["Média"],probs["Alta"]]})
    fig=px.bar(p_df,x="Nível",y="Probabilidade",text_auto=".1%",range_y=[0,1],
               labels={"Probabilidade":"Probabilidade estimada"})
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig,use_container_width=True)

    st.info("A estimativa combina o sinal histórico do setor (60%) e o sinal histórico do estado (40%). Ela indica uma tendência de demanda, não uma garantia de contratação.")

    st.markdown("### 📈 Histórico do setor")
    sh=sector_hist[sector_hist.setor==setor_sel].sort_values("date")
    fig=px.line(sh,x="date",y="taxa_liquida",markers=True,
                labels={"date":"Mês","taxa_liquida":"Taxa líquida (saldo/estoque)"})
    fig.update_layout(yaxis_tickformat=".2%")
    st.plotly_chart(fig,use_container_width=True)

elif page=="🚀 Encontrar oportunidades":
    st.markdown("## 🚀 Encontrar áreas com maior potencial")
    regiao=st.selectbox("Região",sorted(set(REGION_MAP.values())))
    estados=sorted([u for u in t2.uf.unique() if REGION_MAP.get(u)==regiao])
    uf_sel=st.selectbox("Estado",estados)

    ufr=uf_fc[uf_fc.uf==uf_sel].iloc[0]
    ranking=sec_fc.copy()
    ranking["prob_combinada"]=0.6*ranking["prob_Alta"]+0.4*float(ufr["prob_Alta"])
    ranking["demanda"]="Baixa"
    ranking.loc[ranking.prob_combinada>=0.34,"demanda"]="Média"
    ranking.loc[ranking.prob_combinada>=0.50,"demanda"]="Alta"
    ranking=ranking.sort_values("prob_combinada",ascending=False).head(10)

    st.write(f"### Áreas com maior potencial em **{uf_sel}**")
    show=ranking[["setor","demanda","prob_combinada"]].copy()
    show["prob_combinada"]=show["prob_combinada"].map(lambda x:f"{x:.1%}")
    show.columns=["Área / setor","Demanda estimada","Probabilidade de alta demanda"]
    st.dataframe(show,use_container_width=True,hide_index=True)

    fig=px.bar(ranking.sort_values("prob_combinada"),x="prob_combinada",y="setor",orientation="h",
               text=ranking.sort_values("prob_combinada").prob_combinada.map(lambda x:f"{x:.1%}"),
               labels={"prob_combinada":"Probabilidade estimada de alta demanda","setor":"Setor"})
    fig.update_layout(xaxis_tickformat=".0%",height=500)
    st.plotly_chart(fig,use_container_width=True)

    st.info("O ranking combina 60% da probabilidade de alta demanda do setor com 40% da probabilidade de alta demanda do estado. O objetivo é apoiar a exploração do mercado, não garantir oportunidades individuais.")

elif page=="🤖 Sobre o modelo":
    st.markdown("## 🤖 Como o Machine Learning funciona?")
    st.write("O projeto usa dois modelos Random Forest: um aprende padrões históricos dos setores econômicos e outro aprende padrões históricos das Unidades da Federação.")

    c1,c2=st.columns(2)
    c1.metric("Acurácia — modelo de setores",pct(metrics["setor"]["accuracy"]))
    c2.metric("Acurácia — modelo de UFs",pct(metrics["uf"]["accuracy"]))

    st.markdown("### Como evitamos vazamento de informação?")
    st.write("O modelo usa apenas dados disponíveis até o mês de referência para tentar classificar o mês seguinte. O treinamento vai até dezembro de 2025 e o teste usa janeiro a junho de 2026.")

    st.markdown("### Variável prevista")
    st.write("Usamos a **taxa líquida de emprego = saldo de empregos / estoque**, pois ela é mais comparável entre setores e estados de tamanhos diferentes.")
    st.write("As classes Baixa, Média e Alta são definidas pelos tercis da taxa líquida no conjunto de treinamento.")

    st.markdown("### Matriz de confusão — setores")
    cm=np.array(metrics["setor"]["confusion_matrix"])
    fig=px.imshow(cm,text_auto=True,x=["Baixa","Média","Alta"],y=["Baixa","Média","Alta"],
                  labels=dict(x="Predito",y="Real",color="Quantidade"))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Matriz de confusão — UFs")
    cm=np.array(metrics["uf"]["confusion_matrix"])
    fig=px.imshow(cm,text_auto=True,x=["Baixa","Média","Alta"],y=["Baixa","Média","Alta"],
                  labels=dict(x="Predito",y="Real",color="Quantidade"))
    st.plotly_chart(fig,use_container_width=True)

    st.warning("A aplicação tem finalidade acadêmica. A classificação representa uma estimativa baseada no comportamento histórico dos dados do CAGED e não constitui promessa de emprego ou previsão garantida de vagas.")
