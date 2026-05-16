# Fine-Tuning PPO : Peaufiner un modèle sans le casser

## La grande idée

Une fois que nous avons un modèle de récompense (reward model) qui note les réponses, nous voulons que notre modèle de langage produise des réponses avec des scores plus élevés. PPO (Proximal Policy Optimization) fait exactement cela — mais il ajoute une ceinture de sécurité pour que le modèle ne chasse pas le score au point d'oublier comment écrire un texte normal.

Considérez cela comme une étape de polissage. Le modèle parle déjà couramment ; nous le poussons simplement à s'exprimer d'une manière que le modèle de récompense valorise, tout en gardant sa « voix » reconnaissable.

## Une analogie concrète

Imaginez un chef qui cuisine déjà bien mais qui apprend maintenant à plaire à un critique culinaire spécifique.

Après chaque plat, le critique donne une note. Le chef est soumis à deux pressions :

1. **Obtenir un score plus élevé.** Cuisiner de la manière que le critique apprécie.
2. **Ne pas devenir méconnaissable.** Si le chef abandonne complètement son propre style — en versant du sel par tasses entières juste pour chasser un score — la nourriture devient bizarre. Les clients cessent de venir.

PPO capture ces deux pressions :

- La partie **récompense** pousse le modèle vers des réponses que le juge apprécie.
- La partie **pénalité KL** ramène le modèle vers la façon dont il s'exprimait avant le début de l'entraînement. KL est simplement une façon de mesurer « à quel point le nouveau comportement est différent de l'ancien ».

Ensemble, ils disent : *améliore-toi, mais reste toi-même*.

## Comment fonctionne l'apprentissage (intuition uniquement)

Chaque cycle d'entraînement ressemble à ceci :

1. Prendre quelques amorces (prompts). Laisser le modèle actuel produire des réponses.
2. Noter les réponses avec le modèle de récompense.
3. Comparer avec le **modèle de référence** — une copie figée du modèle avant l'entraînement. Si les nouvelles réponses sont radicalement différentes, soustraire une pénalité KL de la récompense.
4. Pousser le modèle vers les réponses qui ont obtenu de bons scores.

Le terme « Proximal » dans PPO signifie *ne pas faire de grands bonds*. Chaque mise à jour est une petite étape prudente. Les grands bonds dans l'entraînement de politique provoquent des plantages, c'est pourquoi les méthodes antérieures comme le gradient de politique classique étaient si instables.

## Ce que montre l'expérience

Nous commençons avec une politique neuve et un modèle de récompense entraîné. PPO s'exécute pendant 150 itérations, échantillonnant des lots de réponses et mettant à jour la politique.

![Entraînement PPO](outputs/ppo_fine_tuning.png)

- **À gauche** — le score moyen du modèle de récompense grimpe régulièrement. La politique apprend à produire des réponses que le juge apprécie.
- **Au milieu** — la divergence KL par rapport au modèle de référence augmente. La politique s'éloigne de son point de départ. C'est attendu, mais si cela augmentait sans contrôle, le modèle dériverait vers le non-sens.
- **À droite** — la récompense mise en forme (récompense brute moins la pénalité KL) suit de près la récompense brute au début, puis prend du retard à mesure que la KL grimpe. La pénalité fait son travail : faire « payer » le modèle pour avoir trop dérivé.

Dans un système RLHF réel, vous ajustez le coefficient KL jusqu'à ce que le score continue d'augmenter tout en maintenant la cohérence du modèle. Une pénalité trop faible et le modèle « pirate » la récompense en émettant des phrases répétitives bizarres. Une pénalité trop forte et le modèle ne s'améliore jamais.

## Place dans le pipeline RLHF

Il s'agit de la deuxième étape de la recette classique du RLHF :

1. Entraîner un modèle de récompense à partir des préférences.
2. **Affiner (fine-tuning) le modèle de langage avec PPO en utilisant ce modèle de récompense.**
3. (Facultatif) Sauter l'étape 2 avec DPO si vous préférez un chemin plus simple.

PPO est l'outil de base que des entreprises comme OpenAI et Anthropic ont utilisé pour la première vague de modèles alignés, y compris InstructGPT et le ChatGPT original.

## Pourquoi c'est important en dehors du laboratoire

Le schéma « s'améliorer, mais ne pas dériver » se retrouve partout :

- Un pianiste qui travaille un morceau difficile ne change pas toute sa technique pour réussir un passage — cela gâcherait le reste du récital.
- Une entreprise qui modifie un site web pour augmenter les inscriptions doit tout de même garder la marque reconnaissable pour les utilisateurs existants.
- Une usine qui ajuste un bouton dans un processus maintient les autres réglages proches des valeurs connues comme bonnes.

PPO n'est qu'une version rigoureuse de cette idée universelle, écrite en langage mathématique.

## Résumé en une phrase

**Le fine-tuning PPO pousse un modèle vers une récompense plus élevée tandis qu'une pénalité KL le maintient proche de son comportement d'origine — s'améliorer, tout en restant soi-même.**
