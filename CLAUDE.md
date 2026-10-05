# CLAUDE.md

## Analysis workflow: I write the conclusions

I'm building my own analytical judgment, so interpreting results is my job, not Claude's.

When Claude runs an analysis (EDA, plots, statistics, model results):

1. **Claude builds and runs it.** Code, plots and tables are fine. Explain what each plot or table shows and how to read it (axes, units, what the metric measures), but not what it means for the data.
2. **Claude doesn't draw conclusions.** No insights cells, no "this suggests...", no recommendations for features or modelling. Leave an empty `*insights*:` markdown cell for me to fill in.
3. **I write the conclusions** in the notebook.
4. **Then we validate together.** When I ask, Claude checks each of my claims against the data. For each claim, say whether it's right, wrong or imprecise and why, and run a query when a claim needs checking. Point out anything important I missed as a question ("what happens to X when Y?"), not as the answer.

If I ask directly for Claude's interpretation, it can give it, but only after I've written mine.
