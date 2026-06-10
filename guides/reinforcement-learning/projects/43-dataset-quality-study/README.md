# Dataset-Quality Study

## Key Insight

The same [offline RL](/shared/glossary/#offline-rl) algorithm can look brilliant or useless depending only on *who collected the data*, and this study makes that dependence visible by running one fixed method across three [D4RL](/shared/glossary/#d4rl) datasets of the same task — `random` (a flailing [behavior policy](/shared/glossary/#behavior-policy)), `medium` (a half-trained one), and `expert` (a polished one) — and plotting final [return](/shared/glossary/#return) against data quality. The lesson is that offline RL cannot conjure skill the data never shows: on expert data even plain [behavior cloning](/shared/glossary/#bc) is hard to beat, while on random data the algorithm's ability to *stitch together* good fragments from many mediocre trajectories is what separates real offline RL from imitation. Knowing where your dataset sits on this curve tells you whether a sophisticated method is even worth the trouble.
