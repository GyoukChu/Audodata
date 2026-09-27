Prompts for the CS research-paper pipeline.
- main_agent.md      : VERBATIM main-agent prompt from Meta's RAM README (projects/autodata), with only the acceptance
                       threshold lines templated ({{WEAK_CRITERIA_SHORT}} etc.) so the config's preset drives them.
- task_prompt.md     : the task (user) message for the main agent; "The paper text is in the task prompt" (paper Fig. 7).
- challenger.md      : expanded from paper Fig. 8 (every instruction kept; output format made explicit).
- quality_verifier.md: expanded from paper Fig. 9 (every check kept; output lines as listed in the figure).
- judge.md           : rubric judge (not given in the paper; implements the stated semantics: binary per criterion,
                       strict, negative criteria scored when the behaviour occurs, no reference answer).
- solver_user.md     : what the weak/strong solvers see: context + question only.
