"""PRIORITEST-STAR — application de visualisation du Top-k (Streamlit).

Prérequis : python train_final.py --project ant   puis   python analyse_star.py --project ant
Lancement : streamlit run app.py
"""
import json
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE, GRAY = '#257de0', '#f47230', '#9a9a9a'
st.set_page_config(page_title='PRIORITEST-STAR', layout='wide')


@st.cache_data
def load(project):
    d = ROOT / 'results' / 'analysis'
    df = pd.read_csv(d / 'ranked_classes.csv')
    info = json.loads((d / 'analysis.json').read_text(encoding='utf-8'))
    return df, info


st.title('PRIORITEST-STAR — priorisation des classes à tester')
try:
    df, info = load('ant')
except FileNotFoundError:
    st.error("Résultats introuvables. Lancez d'abord : python train_final.py --project ant puis python analyse_star.py --project ant")
    st.stop()
st.caption(f"Projet {info['project'].capitalize()} {info['test']} (version de test) · {info['n']} classes · "
           f"{info['positives']} défectueuses · modèle STAR-R ({' + '.join(info['blend'])}) · "
           "détection des défauts de la version observée, pas prédiction de la version suivante.")

# ---------------------------------------------------------------- Contrôles (une ligne)
c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
mode = c1.radio("Ordre d'inspection", ['Densité de risque (risque / LOC)', 'Risque brut'], horizontal=False)
budget_type = c2.radio('Type de budget', ['% des lignes de code', 'Nombre de classes (Top-k)'], horizontal=False)
if budget_type.startswith('%'):
    pct = c3.slider('Budget (% des LOC totales)', 1, 100, 20)
else:
    k_top = c3.slider('k (classes inspectées)', 1, len(df), 30)
show_labels = st.toggle('Afficher les vrais labels (évaluation)', value=True)

key = 'rang_densite' if mode.startswith('Densité') else 'rang_risque'
d = df.sort_values(key).reset_index(drop=True)
d['loc_cumul_%'] = 100 * d['loc'].cumsum() / d['loc'].sum()
sel = d[d['loc_cumul_%'] <= pct] if budget_type.startswith('%') else d.head(k_top)

# ---------------------------------------------------------------- Indicateurs
found = int(sel['defectueuse'].sum()); tot = int(df['defectueuse'].sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric('Classes à inspecter', f'{len(sel)} / {len(df)}')
m2.metric('Lignes de code', f"{int(sel['loc'].sum()):,} ({100 * sel['loc'].sum() / df['loc'].sum():.0f} %)".replace(',', ' '))
if show_labels:
    m3.metric('Défauts trouvés (rappel)', f'{found} / {tot} ({100 * found / max(1, tot):.0f} %)')
    m4.metric('Précision', f'{100 * found / max(1, len(sel)):.0f} %')

# ---------------------------------------------------------------- Tableau Top-k + détail
left, right = st.columns([1.6, 1])
with left:
    st.subheader('Classes priorisées')
    cols = ['classe', 'score', 'risque_par_kLOC', 'loc', 'wmc', 'cbo', 'rfc', 'alpha_temporel', 'accord_cos']
    if show_labels:
        cols.insert(1, 'defectueuse')
    view = sel[cols].copy(); view.index = np.arange(1, len(view) + 1); view.index.name = 'rang'
    view['classe'] = view['classe'].str.replace('org.apache.tools.ant.', '', regex=False)
    st.dataframe(view, height=430, width='stretch',
                 column_config={'score': st.column_config.ProgressColumn('score', min_value=0., max_value=1., format='%.3f'),
                                'defectueuse': st.column_config.CheckboxColumn('défaut')})
    st.download_button('Télécharger la sélection (CSV)', sel.to_csv(index=False).encode('utf-8'), 'selection_topk.csv')
with right:
    st.subheader('Détail d\'une classe')
    name = st.selectbox('Classe', d['classe'], format_func=lambda s: s.replace('org.apache.tools.ant.', ''))
    r = d[d['classe'] == name].iloc[0]
    st.markdown(f"**Rang** : {int(r[key])} / {len(d)} · **score** {r['score']:.3f} · **{r['risque_par_kLOC']:.2f}** par kLOC")
    st.markdown(f"Poids de fusion : temporel α = **{r['alpha_temporel']:.2f}**, structurel β = **{1 - r['alpha_temporel']:.2f}** · "
                f"accord cos(zT, zS) = **{r['accord_cos']:.2f}**")
    if show_labels:
        st.markdown('Label réel : ' + ('🟠 **défectueuse**' if r['defectueuse'] else '🔵 saine'))
    st.dataframe(pd.DataFrame({'métrique': ['loc', 'wmc', 'cbo', 'rfc', 'ca', 'ce', 'lcom', 'max_cc'],
                               'valeur': [r[m] for m in ['loc', 'wmc', 'cbo', 'rfc', 'ca', 'ce', 'lcom', 'max_cc']],
                               'médiane du projet': [df[m].median() for m in ['loc', 'wmc', 'cbo', 'rfc', 'ca', 'ce', 'lcom', 'max_cc']]}),
                 hide_index=True, width='stretch')

# ---------------------------------------------------------------- Courbe effort / rappel
if show_labels:
    st.subheader('Courbe effort–rappel')
    curves = []
    for lab, k in (('Densité de risque', 'rang_densite'), ('Risque brut', 'rang_risque')):
        s = df.sort_values(k)
        curves.append(pd.DataFrame({'LOC inspectées (%)': np.r_[0, 100 * s['loc'].cumsum() / s['loc'].sum()],
                                    'Défauts trouvés (%)': np.r_[0, 100 * s['defectueuse'].cumsum() / tot], 'ordre': lab}))
    curves.append(pd.DataFrame({'LOC inspectées (%)': [0, 100], 'Défauts trouvés (%)': [0, 100], 'ordre': 'Aléatoire'}))
    c = pd.concat(curves)
    nearest = alt.selection_point(nearest=True, on='pointerover', fields=['LOC inspectées (%)'], empty=False)
    base = alt.Chart(c).encode(x=alt.X('LOC inspectées (%):Q', scale=alt.Scale(domain=[0, 100])),
                               y=alt.Y('Défauts trouvés (%):Q', scale=alt.Scale(domain=[0, 100])),
                               color=alt.Color('ordre:N', scale=alt.Scale(domain=['Densité de risque', 'Risque brut', 'Aléatoire'],
                                                                         range=[ORANGE, BLUE, GRAY]), legend=alt.Legend(orient='bottom', title=None)))
    lines = base.mark_line(strokeWidth=2)
    pts = base.mark_point(size=60, filled=True).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
                                                        tooltip=['ordre', alt.Tooltip('LOC inspectées (%):Q', format='.1f'),
                                                                 alt.Tooltip('Défauts trouvés (%):Q', format='.1f')]).add_params(nearest)
    x_now = pct if budget_type.startswith('%') else float(100 * sel['loc'].sum() / df['loc'].sum())
    rule = alt.Chart(pd.DataFrame({'x': [x_now]})).mark_rule(color='#999', strokeDash=[4, 4]).encode(x=alt.X('x:Q', title='LOC inspectées (%)'))
    st.altair_chart((lines + pts + rule).properties(height=320), width='stretch')
    st.caption("Le risque étant corrélé à la taille, le tri par risque brut inspecte d'abord les grosses classes et épuise le budget ; "
               "le tri par densité trouve environ 2,5 fois plus de défauts à 20 % des LOC.")
