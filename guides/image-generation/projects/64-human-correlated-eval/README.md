# Human-Correlated Eval

## Key Insight

Automatic graders are cheap but only trustworthy if they agree with people, so this project pits an [LLM-as-judge](/shared/glossary/#llm-as-judge) against real human ratings on the same 100 outputs and measures how often they agree. Collecting 3 human and 3 model ratings per image (rather than one of each) averages out individual noise, giving a more stable estimate of the true agreement. The lesson is methodological: never trust an automatic metric until you have shown it correlates with human judgment — the metric is only a *proxy*, and a proxy that disagrees with people is worthless.
