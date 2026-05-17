# Ajuste Fino com PPO: Polindo um Modelo sem Quebrá-lo

## A Grande Ideia

Uma vez que temos um modelo de recompensa que pontua as respostas, queremos que nosso modelo de linguagem produza respostas com pontuações mais altas. O PPO (Proximal Policy Optimization) faz exatamente isso — mas adiciona um cinto de segurança para que o modelo não persiga a pontuação e esqueça como escrever um texto normal.

Pense nisso como uma etapa de polimento. O modelo já fala fluentemente; nós apenas damos um empurrãozinho para que ele fale de uma forma que o modelo de recompensa recompense, mantendo sua voz reconhecível.

## Uma Analogia da Vida Real

Imagine um chef que já cozinha bem, mas agora está aprendendo a agradar a um crítico gastronômico específico.

Após cada prato, o crítico dá uma nota. O chef tem duas pressões:

1. **Obter uma nota mais alta.** Cozinhar do jeito que o crítico gosta.
2. **Não se tornar irreconhecível.** Se o chef abandonar completamente seu próprio estilo — jogando sal às xícaras apenas para perseguir uma nota — a comida fica estranha. Os clientes param de vir.

O PPO captura ambas as pressões:

- A parte da **recompensa** empurra o modelo em direção às respostas que o juiz gosta.
- A parte da **penalidade KL** puxa o modelo de volta para a forma como ele falava antes do início do treinamento. KL é apenas uma forma de medir "o quão diferente é o novo comportamento em relação ao antigo".

Juntos, eles dizem: *melhore, mas continue sendo você mesmo*.

## Como o Aprendizado Funciona (Apenas Intuição)

Cada rodada de treinamento se parece com:

1. Pegar alguns comandos (prompts). Deixar o modelo atual produzir respostas.
2. Pontuar as respostas com o modelo de recompensa.
3. Comparar com o **modelo de referência** — uma cópia congelada do modelo de antes do treinamento. Se as novas respostas forem muito diferentes, subtrair uma penalidade KL da recompensa.
4. Empurrar o modelo em direção às respostas que obtiveram boa pontuação.

O "Proximal" no PPO significa *não dê saltos grandes*. Cada atualização é um passo pequeno e cuidadoso. Grandes saltos no treinamento de políticas causam falhas catastróficas, e é por isso que métodos anteriores, como o gradiente de política básico (vanilla), eram tão instáveis.

## O que o Experimento Mostra

Começamos com uma nova política e um modelo de recompensa treinado. O PPO executa por 150 iterações, amostrando lotes de respostas e atualizando a política.

![Treinamento PPO](outputs/ppo_fine_tuning.png)

- **Esquerda** — a pontuação média do modelo de recompensa sobe constantemente. A política está aprendendo a produzir respostas que o juiz gosta.
- **Meio** — a divergência KL em relação ao modelo de referência cresce. A política está se afastando de onde começou. Isso é esperado, mas se crescesse sem controle, o modelo derivaria para o absurdo.
- **Direita** — a recompensa moldada (recompensa bruta menos a penalidade KL) acompanha de perto a recompensa bruta no início, depois fica para trás à medida que a KL sobe. A penalidade está fazendo seu trabalho: fazendo o modelo "pagar" por se afastar demais.

Em um sistema RLHF real, você ajusta o coeficiente KL até que a pontuação ainda suba, mas o modelo permaneça coerente. Uma penalidade muito pequena e o modelo "hackeia" a recompensa emitindo frases repetitivas estranhas. Muito grande e o modelo nunca melhora.

## Onde isso se Enquadra no Pipeline de RLHF

Este é o passo dois da receita clássica de RLHF:

1. Treinar um modelo de recompensa a partir de preferências.
2. **Ajustar o modelo de linguagem com PPO usando esse modelo de recompensa.**
3. (Opcional) Pular o passo 2 com DPO se você quiser um caminho mais simples.

O PPO é a ferramenta de trabalho que empresas como OpenAI e Anthropic usaram para a primeira onda de modelos alinhados, incluindo o InstructGPT e o ChatGPT original.

## Por que isso Importa Fora do Laboratório

O padrão "melhorar, mas não derivar" aparece em todos os lugares:

- Um pianista que pratica uma peça difícil não muda toda a sua técnica para acertar uma passagem — isso estragaria o resto do recital.
- Uma empresa que ajusta um site para aumentar as inscrições ainda precisa manter a marca reconhecível para os usuários existentes.
- Uma fábrica que ajusta um botão em um processo mantém os outros próximos às configurações conhecidas como boas.

O PPO é apenas uma versão cuidadosa desta ideia universal, escrita em matemática.

## Resumo em uma Frase

**O ajuste fino com PPO empurra um modelo em direção a uma recompensa mais alta, enquanto uma penalidade KL o mantém próximo de seu comportamento original — melhore, mas continue sendo você mesmo.**
