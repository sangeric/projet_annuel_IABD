# Presentation du PA

## Recommendation pour réviser les algos

- Lire du Pseudo code
- Le comprendre
- Fermer le support
- Faire le code 

## Obectifs du PA

- Améliorer la compréhension (Généraliser ?)
- Réaliser une App (Site web OU App Windows OU App Linux)
- Faire un Rapport pour le PA
- Faire attention à l'interaction avec le Jury pour leurs montrer que l'on comprend ce que l'on a fait
- Le but de l'app est : Résoudre un problème de régression ou de classification avec un model de machine learning
- Utiliser des librairies qui suivent la convention "C ABI" (dll en Windows, so en Linux, dylib en Mac)
- Peux faire les libs en C, C++, Rust ou Zig
- Pour les libs, ils y aura des cas de tests
- UNIQUEMENT A TITRE DE COMPARAISON, on doit utiliser les libs préexistante comme tensorflow, pytorch, keras
- Créer un dataset pas prééxistant
- Il faut faire de l'expérimentation
- Pour les libs, Linear - MLP (mulit layer perceptron) - RBF - (SUM)
- Apprendre à faire une implémentation CUDA pour l'apprentissage
- Les libs et les algo sont 100x plus important que l'app elle même
- Les libs n'ont pas le droit d'avoir de libs externes mais tout le reste si

+---------+                                                           +--------------------------+
|         |                                                           |                          |
| DataSet | ----------------+------------------------+                | Interaction avec le Jury | ---+
|         |                 |                        |                |                          |    |
+---------+                 v                        v                +--------------------------+    |
                                                                                                      |
+--------------+       +----------+         +-----------------+       +-----+      +------------+     |
|              |       |          |         |                 |       |     |      |            |     |
| Cas de Tests | ----> |   Lib    |    +--> | Experimentation | ----> | App | ---> | Comprendre |  <--+
|              |       | Zig Rust | ---+    |                 |       |     |      |            |
+--------------+       |  C  C++  |         +-----------------+       +-----+      +------------+
                       |          |
                       +----------+                                   +---------+        ^
                                                                      |         |        |
                                                                      | Rapport | -------+
                                                                      |         |
                                                                      +---------+

## Dâtes provisoires

- 13/01/2026 : Groupe soit formé + Proposition de projet
- 06/04/2026 : Modèle linéaire et algo fonctionnels sur des cas de tests
- 20/07/2026 : Soutenance
