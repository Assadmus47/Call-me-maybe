# 🗓️ TODO — Call me maybe (5 jours)

## Jour 1 — Setup + CLI + Parsing 🟢
*Objectif : le programme tourne, lit les fichiers, comprend les arguments*

- [✅] **Étape 0** — Setup : structure des dossiers + git + `uv init`
- [✅] **Étape 1** — CLI : lire les 3 arguments (`--functions_definition`, `--input`, `--output`)
- [✅] **Étape 2** — Parser `functions_definition.json` → modèles Pydantic
- [ ] **Étape 3** — Parser `function_calling_tests.json` → modèles Pydantic
- [ ] **Étape 4** — Écrire le fichier de sortie (même avec des données bidons)

> ✅ Fin de journée : `uv run python -m src` tourne sans crash, lit les inputs, écrit un output.

---

## Jour 2 — Explorer le SDK 🟡
*Objectif : comprendre comment le LLM fonctionne concrètement*

- [ ] **Étape 5** — Tester `encode()`, `get_logits_from_input_ids()`, `decode()`
- [ ] **Étape 6** — Charger le fichier vocab, comprendre le format token ↔ id
- [ ] **Étape 6b** — Construire le mapping id → token string

> ✅ Fin de journée : tu sais ce qu'est un token, un logit, et comment naviguer dans le vocab.

---

## Jour 3 — Constrained decoding (le cœur) 🔴
*Objectif : forcer le modèle à générer du JSON valide token par token*

- [ ] **Étape 7** — Comprendre la boucle de génération (schéma à la main)
- [ ] **Étape 7b** — Coder un cas ultra simple : générer `{"a": <nombre>}` avec contrainte
- [ ] **Étape 8** — Généraliser : gérer string, number, boolean, plusieurs paramètres

> ✅ Fin de journée : le decoder produit du JSON 100% valide pour n'importe quel schéma.

---

## Jour 4 — Sélection de fonction + Pipeline complet 🔴
*Objectif : le LLM choisit la bonne fonction, tout s'assemble*

- [ ] **Étape 9** — Faire choisir la fonction au LLM (avec contrainte sur les noms valides)
- [ ] **Étape 10** — Pipeline complet : prompt → fonction → arguments → fichier de sortie
- [ ] **Étape 11** — Gestion d'erreurs : fichier manquant, JSON cassé, edge cases

> ✅ Fin de journée : pipeline end-to-end qui tourne sur tous les prompts de test.

---

## Jour 5 — Finition + qualité 42 ⚙️
*Objectif : code propre, conforme aux exigences du sujet*

- [ ] **Étape 12** — `flake8` → 0 erreur
- [ ] **Étape 12b** — `mypy` avec les flags du sujet → 0 erreur
- [ ] **Étape 12c** — Type hints + docstrings PEP 257 partout
- [ ] **Étape 13** — Makefile (`install`, `run`, `debug`, `clean`, `lint`)
- [ ] **Étape 14** — README complet (en anglais, sections obligatoires du sujet)
- [ ] **Étape 15** — Vérification finale : `uv sync` + test à froid + pas d'`output/` dans le git

> ✅ Fin de journée : projet prêt à soumettre 🎉

---

## 🎁 Bonus (si t'as du temps en avance)
- [ ] Tokenizer maison (encode/decode sans utiliser directement les méthodes du SDK)
- [ ] Tests unitaires avec pytest
- [ ] Support nested arguments