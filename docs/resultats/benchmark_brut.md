| candidat | n_evalues | f1_macro | latence_ms | cout_eur_1k | kg_co2 | donnees | entrainement |
|---|---|---|---|---|---|---|---|
| baseline_nb | 9568 | 0.6711 | 0.026 | 0.0 | 2.783779239880933e-07 | restent en local | requis |
| tfidf_logreg | 9568 | 0.8661 | 0.057 | 0.0 | 2.784153549813086e-05 | restent en local | requis |
| tfidf_logreg_meme_echantillon | 80 | 0.9254 | 0.057 | 0.0 | — | restent en local | requis |
| claude_zero_shot (claude-haiku-5-5) | 80 | 0.4746 | 1053.8 | 0.016 | — | envoyées à un fournisseur externe (Anthropic) | aucun |
