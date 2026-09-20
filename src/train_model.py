
from pathlib import Path
import pandas as pd
import numpy as np
import re, json, joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; MODELS=ROOT/"models"
XLSX=DATA/"3-tabelas_Junho de 2026.xlsx"

MONTH_RE=re.compile(r"^(Janeiro|Fevereiro|Março|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro)/\d{4}$")
MONTHS_PT={"Janeiro":1,"Fevereiro":2,"Março":3,"Abril":4,"Maio":5,"Junho":6,"Julho":7,"Agosto":8,"Setembro":9,"Outubro":10,"Novembro":11,"Dezembro":12}

def parse_monthly(sheet):
    raw=pd.read_excel(XLSX,sheet_name=sheet,header=None)
    months=raw.iloc[4].ffill(); metrics=raw.iloc[5]
    cols=[]
    for c in range(2,raw.shape[1]):
        m=str(months.iloc[c]).strip(); metric=str(metrics.iloc[c]).strip()
        if MONTH_RE.match(m) and metric in {"Estoque","Admissões","Desligamentos","Saldos"}:
            cols.append((c,m,metric))
    rows=raw.iloc[6:,[1]+[c for c,_,_ in cols]].copy()
    rows.columns=["categoria"]+[f"{m}__{metric}" for _,m,metric in cols]
    rows=rows[rows.categoria.notna() & (rows.categoria.astype(str).str.strip()!="Total")]
    long=rows.melt(id_vars="categoria",var_name="mes_metric",value_name="valor")
    long[["mes","metric"]]=long.mes_metric.str.split("__",n=1,expand=True)
    long.valor=pd.to_numeric(long.valor,errors="coerce")
    wide=long.pivot_table(index=["categoria","mes"],columns="metric",values="valor",aggfunc="first").reset_index()
    wide["date"]=wide.mes.map(lambda s:pd.Timestamp(int(s.split("/")[1]),MONTHS_PT[s.split("/")[0]],1))
    wide=wide.sort_values(["categoria","date"]).reset_index(drop=True)
    wide["estoque_ref"]=wide.groupby("categoria")["Estoque"].ffill().bfill()
    wide["taxa_liquida"]=wide.Saldos/wide.estoque_ref
    return wide

def make_supervised(hist,key):
    rows=[]
    for entity,g in hist.groupby(key):
        g=g.sort_values("date").reset_index(drop=True)
        for i in range(6,len(g)-1):
            h=g.iloc[i-5:i+1]; target=g.iloc[i+1]
            vals=h.taxa_liquida.to_numpy(float)
            if len(vals)!=6 or not np.isfinite(vals).all() or not np.isfinite(target.taxa_liquida):
                continue
            rows.append({
                key:entity,"date_base":g.iloc[i].date,"target_date":target.date,"target_taxa":float(target.taxa_liquida),
                "taxa_atual":float(vals[-1]),"taxa_media_3m":float(vals[-3:].mean()),
                "taxa_media_6m":float(vals.mean()),"taxa_tendencia_6m":float(np.polyfit(np.arange(6),vals,1)[0]),
                "taxa_volatilidade_6m":float(vals.std()),"saldo_atual":float(h.iloc[-1].Saldos),
                "estoque_atual":float(h.iloc[-1].estoque_ref),"admissoes_atual":float(h.iloc[-1].Admissões),
                "desligamentos_atual":float(h.iloc[-1].Desligamentos),"saldo_medio_6m":float(h.Saldos.mean()),
                **{f"taxa_lag_{6-j}m":float(vals[j]) for j in range(6)}
            })
    return pd.DataFrame(rows)

def fit(sup,key):
    train=sup[sup.target_date<=pd.Timestamp("2025-12-01")].copy()
    test=sup[(sup.target_date>=pd.Timestamp("2026-01-01"))&(sup.target_date<=pd.Timestamp("2026-06-01"))].copy()
    q1,q2=train.target_taxa.quantile([1/3,2/3]).tolist()
    label=lambda s:np.select([s<=q1,s<=q2],["Baixa","Média"],default="Alta")
    train["demanda"]=label(train.target_taxa); test["demanda"]=label(test.target_taxa)
    exclude={key,"date_base","target_date","target_taxa","demanda"}
    features=[c for c in train.columns if c not in exclude]
    model=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=3,class_weight="balanced",random_state=42)
    model.fit(train[features],train.demanda)
    pred=model.predict(test[features])
    metrics={"accuracy":float(accuracy_score(test.demanda,pred)),
             "report":classification_report(test.demanda,pred,output_dict=True,zero_division=0),
             "confusion_matrix":confusion_matrix(test.demanda,pred,labels=["Baixa","Média","Alta"]).tolist(),
             "labels":["Baixa","Média","Alta"],"train_rows":len(train),"test_rows":len(test),
             "test_period":"jan/2026 a jun/2026","thresholds":[float(q1),float(q2)]}
    return model,features,metrics

sector=parse_monthly("Tabela 6").rename(columns={"categoria":"setor"})
uf=parse_monthly("Tabela 7").rename(columns={"categoria":"uf"})
sec_sup=make_supervised(sector,"setor"); uf_sup=make_supervised(uf,"uf")
sec_model,sec_features,sec_metrics=fit(sec_sup,"setor")
uf_model,uf_features,uf_metrics=fit(uf_sup,"uf")

def latest_features(hist,key,entity):
    g=hist[hist[key]==entity].sort_values("date")
    h=g.iloc[-6:]; vals=h.taxa_liquida.to_numpy(float)
    if len(h)<6 or not np.isfinite(vals).all(): return None
    d={"taxa_atual":vals[-1],"taxa_media_3m":vals[-3:].mean(),"taxa_media_6m":vals.mean(),
       "taxa_tendencia_6m":np.polyfit(np.arange(6),vals,1)[0],"taxa_volatilidade_6m":vals.std(),
       "saldo_atual":float(h.iloc[-1].Saldos),"estoque_atual":float(h.iloc[-1].estoque_ref),
       "admissoes_atual":float(h.iloc[-1].Admissões),"desligamentos_atual":float(h.iloc[-1].Desligamentos),
       "saldo_medio_6m":float(h.Saldos.mean())}
    for j,v in enumerate(vals): d[f"taxa_lag_{6-j}m"]=float(v)
    return pd.DataFrame([d])

def forecast_all(hist,key,model,features):
    out=[]
    for e in hist[key].unique():
        f=latest_features(hist,key,e)
        if f is None: continue
        p=model.predict_proba(f[features])[0]
        probs={c:float(p[list(model.classes_).index(c)]) for c in ["Baixa","Média","Alta"]}
        out.append({key:e,"demanda_prevista":max(probs,key=probs.get),**{f"prob_{c}":v for c,v in probs.items()}})
    return pd.DataFrame(out)

sec_fc=forecast_all(sector,"setor",sec_model,sec_features)
uf_fc=forecast_all(uf,"uf",uf_model,uf_features)

REGION_MAP={
"Rondônia":"Norte","Acre":"Norte","Amazonas":"Norte","Roraima":"Norte","Pará":"Norte","Amapá":"Norte","Tocantins":"Norte",
"Maranhão":"Nordeste","Piauí":"Nordeste","Ceará":"Nordeste","Rio Grande do Norte":"Nordeste","Paraíba":"Nordeste","Pernambuco":"Nordeste",
"Alagoas":"Nordeste","Sergipe":"Nordeste","Bahia":"Nordeste","Minas Gerais":"Sudeste","Espírito Santo":"Sudeste","Rio de Janeiro":"Sudeste",
"São Paulo":"Sudeste","Paraná":"Sul","Santa Catarina":"Sul","Rio Grande do Sul":"Sul","Mato Grosso do Sul":"Centro-Oeste","Mato Grosso":"Centro-Oeste",
"Goiás":"Centro-Oeste","Distrito Federal":"Centro-Oeste"}

sector.to_csv(DATA/"historico_setores.csv",index=False,encoding="utf-8-sig")
uf.to_csv(DATA/"historico_ufs.csv",index=False,encoding="utf-8-sig")
sec_sup.to_csv(DATA/"supervisionado_setores.csv",index=False,encoding="utf-8-sig")
uf_sup.to_csv(DATA/"supervisionado_ufs.csv",index=False,encoding="utf-8-sig")
sec_fc.to_csv(DATA/"previsoes_setores_jul2026.csv",index=False,encoding="utf-8-sig")
uf_fc.to_csv(DATA/"previsoes_ufs_jul2026.csv",index=False,encoding="utf-8-sig")

pack={"sector_model":sec_model,"uf_model":uf_model,"sector_features":sec_features,"uf_features":uf_features,
      "region_map":REGION_MAP,"last_date":"2026-06-01"}
MODELS.mkdir(exist_ok=True)
joblib.dump(pack,MODELS/"modelos_demanda.joblib")
metrics={"metodologia":{"target":"taxa líquida de emprego do mês seguinte (saldo/estoque)",
"split":"treino até dezembro/2025; teste de janeiro a junho/2026",
"features":"somente dados disponíveis até o mês-base","classes":"tercis da variável-alvo calculados no treino",
"combinacao_interface":"60% setor + 40% UF"},"setor":sec_metrics,"uf":uf_metrics}
(DATA/"metricas_modelos.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Modelo de setores — acurácia: {sec_metrics['accuracy']:.1%}")
print(f"Modelo de UFs — acurácia: {uf_metrics['accuracy']:.1%}")
print("Previsão preparada para julho/2026.")
