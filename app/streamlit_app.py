"""C10 - Application de saisie de tickets connectée à l'API TicketSort.

Lancer : streamlit run app/streamlit_app.py
Le résultat de l'IA change l'expérience : catégorie pré-remplie, file de traitement suggérée,
et avertissement explicite quand la confiance est faible (le ticket part en revue humaine).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.client import ApiError, classify  # noqa: E402

# File de traitement associée à chaque catégorie du modèle (dataset IT Service Ticket Classification, Kaggle).
FILES = {
    "access": "Équipe Identités & Accès",
    "administrative rights": "Équipe Identités & Accès (droits admin)",
    "hardware": "Équipe Proximité (matériel)",
    "hr support": "Support RH",
    "internal project": "Équipe Projets internes",
    "miscellaneous": "Support niveau 1 (tri manuel)",
    "purchase": "Service Achats IT",
    "storage": "Équipe Stockage & Sauvegarde",
}
FILE_PAR_DEFAUT = "Support niveau 1 (tri manuel)"

st.set_page_config(page_title="TicketSort — Nouveau ticket", page_icon="🎫", layout="centered")
st.title("Déclarer un incident")
st.caption("La catégorie est proposée automatiquement par un modèle d'IA. Vous pouvez la corriger.")

with st.form("ticket"):
    # Accessibilité : libellé explicite, aide textuelle, pas d'information portée par la couleur seule.
    texte = st.text_area("Description du problème", height=150,
                         help="Décrivez ce qui ne fonctionne pas, depuis quand, et sur quel équipement.")
    submitted = st.form_submit_button("Analyser le ticket")

if submitted:
    try:
        with st.spinner("Classification en cours…"):
            res = classify(texte)
    except ApiError as exc:
        st.error(f"⚠️ {exc} Vous pouvez choisir la catégorie manuellement ci-dessous.")
        res = None

    # Catégories connues du modèle (renvoyées par l'API) ; en mode dégradé, liste de référence locale.
    categories = sorted(res.probabilites) if res else sorted(FILES)
    # En mode dégradé, aucune catégorie n'est présélectionnée : l'utilisateur doit choisir lui-même.
    default = categories.index(res.categorie) if res and res.categorie in categories else None
    if res:
        if res.a_verifier:
            st.warning(f"⚠️ Confiance faible ({res.confiance:.0%}) : vérifiez la catégorie. Le ticket sera relu par un humain.")
        else:
            st.success(f"✅ Catégorie proposée : **{res.categorie}** (confiance {res.confiance:.0%})")
        st.bar_chart(res.probabilites)
        st.caption(f"Modèle version {res.version_modele}")

    choix = st.selectbox("Catégorie (modifiable)", categories, index=default,
                         placeholder="Choisissez une catégorie")
    if choix:
        st.info(f"File de traitement : **{FILES.get(choix, FILE_PAR_DEFAUT)}**")
    else:
        st.caption("Sélectionnez une catégorie pour connaître la file de traitement.")