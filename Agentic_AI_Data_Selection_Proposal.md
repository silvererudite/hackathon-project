# Agentic AI for Scientific Discovery Under Data Selection Uncertainty

## Proposal Summary

This proposal develops an agentic AI system that does more than train a predictive model. The system must decide which observations are sufficiently reliable to support a scientific conclusion, quantify uncertainty in that decision, and test whether the conclusion replicates under alternative defensible data selections.

The primary benchmark will use the open SCI OPS dataset from Zorowitz, Solis, Niv, and Bennett. The dataset is especially suitable because the associated study demonstrates that careless or inattentive responding can create spurious associations between cognitive task behavior and psychiatric symptom measures. The paper reports two independent online samples with a combined sample size of 779 participants. It also shows that excluding participants flagged by survey attention checks can eliminate apparent symptom to behavior associations, while exclusion based only on task performance is less effective [1]. The public repository provides trial data, survey responses, participant metadata, quality metrics, analysis scripts, and results notebooks [2].

The central research question is:

> Can an LLM based scientific agent construct a behavioral classifier while explicitly reasoning about uncertainty over which participants and trials should be included, and can it distinguish a robust scientific relationship from one induced by low quality data?

## 1. Motivation

In many machine learning benchmarks, the training dataset is treated as fixed and correct. Scientific data rarely satisfy this assumption. Researchers must make choices about participant exclusion, missing observations, response quality, outliers, preprocessing, and train test partitioning. Different plausible choices can produce different effect sizes or even reverse a scientific conclusion.

This proposal defines **data selection uncertainty** as uncertainty about whether each observation should contribute to model fitting and scientific inference. Instead of hiding this uncertainty inside one preprocessing script, the agent must expose, model, and evaluate it.

The project is an agentic AI for scientific discovery benchmark because the agent must complete a closed research loop:

1. Inspect the data schema and documentation.

2. Form hypotheses about data quality and behavioral prediction.

3. Construct leakage safe features.

4. Train and calibrate candidate models.

5. Identify observations for which inclusion is uncertain.

6. Compare alternative selection policies.

7. Evaluate whether downstream scientific claims are stable.

8. Report evidence, uncertainty, limitations, and replication results.

The goal is therefore not merely high classification accuracy. The goal is reliable discovery under uncertain data inclusion.

## 2. Dataset

### 2.1 Primary dataset: SCI OPS

The primary source is the SCI OPS repository associated with *Inattentive responding can induce spurious associations between task behaviour and symptom measures* [1, 2]. It is catalogued by OpenCogData, a database of public human cognitive task datasets that requires trial or item level data [3].

Inspection of the public repository gives the following benchmark structure:

| Component | Approximate participants | Trial rows | Main behavioral structure |
| --- | ---: | ---: | --- |
| Original sample | 386 | 36,540 | Reversal learning or bandit style task |
| Replication sample | 393 | 80,400 | Two step decision task |
| Total | 779 | 116,940 | Two independent cognitive task settings |

The repository includes several useful data layers:

| Layer | Example variables | Proposed role |
| --- | --- | --- |
| Trial behavior | choice, accuracy, outcome, reaction time, state, transition | Behavioral classification and feature construction |
| Quality metrics | infrequency checks, personal reliability, Mahalanobis distance, survey time, task time, response variability, win stay lose shift | Data quality target and selector features |
| Participant metadata | platform, age, gender, language, browser interactions | Domain shift checks and subgroup analysis |
| Symptom scores | anxiety, depression, mania, anhedonia, worry and related scales | Downstream scientific inference |

The two samples are a major advantage. The original sample can be used for agent development and internal validation, while the replication sample provides a genuine external test of whether the learned data selection policy and scientific findings transfer across tasks.

### 2.2 Why this dataset fits the proposal

SCI OPS has four properties that are uncommon in standard classification datasets:

1. **The data selection problem is scientifically consequential.** Inclusion of inattentive participants can generate false positive associations.

2. **A useful but imperfect quality label exists.** Failure on one or more infrequency items can define a proxy careless or inattentive label.

3. **There are multiple views of quality.** Task behavior, survey behavior, timing, consistency, and multivariate outlier measures do not always agree.

4. **External replication is possible.** A policy learned on one sample can be evaluated on a distinct sample and cognitive task.

This makes the dataset more appropriate than a benchmark in which uncertainty is created only by randomly hiding labels.

## 3. Research Questions

The project will test four questions.

### RQ1: Can behavioral and metadata features identify unreliable participants?

The agent will estimate the probability that participant \(i\) is attentive:

\[
q_i = P(z_i = 1 \mid x_i),
\]

where \(z_i=1\) means that the participant is suitable for inclusion and \(x_i\) contains permissible task, timing, consistency, and metadata features.

The infrequency score will be used only to construct an evaluation target. It will not be provided as an input feature.

### RQ2: How does uncertain inclusion affect behavioral classification?

A second model will predict trial level behavior such as current or next choice:

\[
P(y_{i,t} \mid h_{i,t}, s_{i,t}),
\]

where \(h_{i,t}\) represents the participant's history of prior choices and outcomes, and \(s_{i,t}\) represents the currently available task state.

The agent will compare models trained on all participants, hard filtered participants, probabilistically weighted participants, and multiple sampled inclusion sets.

### RQ3: Are symptom to behavior relationships stable across defensible selection policies?

For every policy, the agent will estimate prespecified associations between symptom scores and behavioral measures. The key output is not one coefficient. It is a distribution over coefficients induced by uncertainty in data selection.

A scientific claim will be considered robust only if its direction, magnitude, and uncertainty remain sufficiently stable across reasonable policies and replicate in the second sample.

### RQ4: Can an agent recognize when task data are insufficient for quality control?

The original study found that task performance alone was often insufficient to remove spurious associations [1]. A capable scientific agent should detect this limitation. When quality predictions are poorly calibrated or ambiguous, it should recommend collecting an additional attention check or reliability measurement instead of claiming that the selection problem has been solved.

## 4. Proposed Agent Architecture

The system will contain an LLM research controller and a restricted set of statistical tools.

| Module | Responsibility |
| --- | --- |
| Schema analyst | Reads documentation, identifies units of analysis, targets, feature timing, and missingness |
| Hypothesis planner | Defines candidate quality mechanisms and downstream scientific hypotheses |
| Feature engineer | Creates participant summaries and lagged trial features without temporal leakage |
| Model trainer | Fits logistic regression, tree ensembles, and calibrated probabilistic classifiers |
| Selection reasoner | Compares hard filtering, probabilistic weighting, active querying, and sensitivity analyses |
| Replication critic | Tests transfer from the original sample to the replication sample |
| Scientific reporter | Produces claims with uncertainty, evidence, alternative explanations, and failure conditions |

The LLM will not directly calculate results from raw values. It will write or call reproducible analysis code, inspect diagnostics, and choose the next analysis action from a bounded action space. Every action and rationale will be logged.

## 5. Data Selection Methods

### 5.1 Baseline policies

The following policies will establish a comparison set:

1. Include every participant.

2. Exclude any participant with at least one failed infrequency item.

3. Exclude fixed quantiles based on task reaction time or accuracy.

4. Exclude multivariate outliers based on Mahalanobis distance.

5. Use a learned binary quality classifier with a fixed threshold.

### 5.2 Probabilistic inclusion

Hard filtering treats every participant as either completely valid or completely invalid. The proposed alternative uses \(q_i\) as a probabilistic inclusion weight:

\[
\mathcal{L}(\theta)=\sum_i \sum_t q_i\,\ell\bigl(y_{i,t}, f_\theta(x_{i,t})\bigr).
\]

This preserves information from ambiguous participants while reducing their influence. Bootstrap or posterior draws of \(q_i\) will produce multiple plausible datasets, allowing downstream uncertainty to include data selection uncertainty.

### 5.3 Active quality querying

To simulate a limited quality assurance budget, most infrequency labels will initially be hidden. At each round, the agent will select participants for whom an additional quality label would be most informative.

Acquisition strategies will include:

1. Random querying.

2. Predictive entropy.

3. Ensemble disagreement.

4. Uncertainty combined with participant diversity.

5. Expected change in the downstream scientific coefficient.

The fifth policy is especially important. A participant may be uninformative for improving quality classification accuracy but highly influential for determining whether a symptom to behavior effect exists. This connects active learning to scientific discovery instead of optimizing only a generic predictive score. Model specific subsampling using influence functions provides one relevant methodological foundation [4].

## 6. Experimental Design

### Phase 1: Reproduce the data quality phenomenon

The first phase will reproduce selected original results using the released scripts and data. This establishes that the benchmark pipeline can recover the reported change in symptom to behavior relationships after quality filtering.

### Phase 2: Train the quality selector

Participant level features will be constructed from task behavior, timing, response consistency, and metadata. Candidate models will include:

1. Regularized logistic regression.

2. Random forest.

3. Gradient boosted trees.

4. A calibrated ensemble of the above.

Cross validation will be performed at the participant level. Calibration will be treated as a first class objective because the inclusion probabilities are used as scientific weights.

### Phase 3: Train the behavioral classifier

Trial level models will predict a prespecified behavior such as choice or accuracy. Features may include previous choices, previous outcomes, trial index, block, state, transition history, and participant level summaries.

Models will be trained under each data selection policy. Splits will always be grouped by participant.

### Phase 4: Quantify scientific conclusion stability

For every selection policy and random seed, the pipeline will estimate the same symptom to behavior relationships. The analysis will record:

1. Effect direction.

2. Effect magnitude.

3. Confidence or credible interval.

4. Whether the interval includes zero.

5. Between policy variability.

6. Replication in the second sample.

### Phase 5: Evaluate the agent

The complete agent will be compared with fixed human specified pipelines. Evaluation will ask whether the agent selects appropriate analyses, detects leakage, requests quality information efficiently, and avoids making claims that fail to replicate.

## 7. Evaluation Metrics

| Objective | Metrics |
| --- | --- |
| Quality classification | AUROC, AUPRC, balanced accuracy, Brier score, expected calibration error |
| Behavioral classification | Macro F1, balanced accuracy, negative log likelihood, Brier score |
| Label efficiency | Performance as a function of quality labels queried, area under the learning curve |
| Selection stability | Jaccard similarity, inclusion probability variance, subgroup inclusion rates |
| Scientific stability | Effect sign agreement, coefficient variance, interval overlap, false positive rate |
| Replication | Direction agreement, standardized effect difference, predictive transfer |
| Agent quality | Invalid action rate, leakage violations, reproducibility, justified stopping decisions |

The primary success criterion will be a joint criterion rather than classification accuracy alone:

> The agent must reduce spurious scientific effects, maintain competitive behavioral prediction, provide calibrated inclusion probabilities, and transfer its conclusions to the replication sample.

## 8. Leakage and Validity Controls

Several safeguards are essential.

1. **Participant grouped splitting:** Trials from the same participant must never appear in both training and test sets.

2. **Temporal feature validity:** A model predicting the current choice may use only information available before that choice. Current outcome and post response measures are prohibited.

3. **Quality target isolation:** Infrequency items and variables computed directly from them cannot be quality classifier inputs.

4. **Symptom isolation:** Symptom scores should not be used to decide participant quality in the primary analysis because this could directly manufacture the downstream relationship.

5. **Policy preregistration:** The set of candidate selection policies and scientific effects should be specified before evaluating replication results.

6. **No clinical interpretation:** The model identifies response quality within this dataset. It is not a diagnostic model for psychiatric disorders.

## 9. Expected Contributions

The project is expected to contribute:

1. A reproducible benchmark for agentic scientific reasoning with trial level cognitive data.

2. A formal treatment of participant inclusion as a probabilistic rather than deterministic decision.

3. An evaluation framework connecting active data selection to downstream scientific effect stability.

4. Evidence about whether LLM agents can recognize when additional data quality measurements are necessary.

5. A practical distinction between predictive success and reliable scientific discovery.

## 10. Risks and Limitations

The infrequency label is a proxy for careless or inattentive responding, not an error free ground truth. Some attentive participants may fail a check, while some inattentive participants may pass. The proposal therefore treats the label as noisy and evaluates sensitivity to alternative quality definitions.

The original and replication samples use different cognitive tasks. This makes transfer more difficult, but it is scientifically valuable because a successful policy must rely on general quality signals rather than task specific shortcuts.

The LLM may prefer complex models or produce post hoc narratives. Requiring prespecified hypotheses, executable code, held out replication, and complete action logs will reduce this risk.

Finally, probabilistic weighting does not automatically remove selection bias. Results must still be presented as sensitivity analyses rather than as proof that one true inclusion set has been recovered.

## 11. Minimal Viable Project

A focused initial study can be completed in four steps:

1. Reproduce the unfiltered and infrequency filtered symptom to behavior associations.

2. Train a calibrated subject level classifier for `infreq > 0` using only permissible task and timing features.

3. Compare all data, hard exclusion, and probabilistic weighting for one behavioral classification target.

4. Test whether conclusions and calibration transfer from the original sample to the replication sample.

This version is small enough to implement quickly while retaining the central scientific contribution. Active querying and influence based selection can be added after the basic benchmark is validated.

## 12. Expected Outcome

The most informative outcome may not be that the agent perfectly identifies bad data. A stronger scientific result would be that the agent learns which conclusions are robust, which depend on arbitrary selection choices, and when the available task data cannot resolve uncertainty without additional quality measurements.

That behavior would distinguish an agent for scientific discovery from an automated model fitting assistant.

## References

1. Zorowitz, S., Solis, J., Niv, Y., & Bennett, D. (2023). *Inattentive responding can induce spurious associations between task behaviour and symptom measures*. Nature Human Behaviour, 7, 1667 to 1681. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37414886/) | [Open access article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170515/)

2. Niv Lab. *SCI OPS: Code and data for Inattentive responding can induce spurious associations between task behavior and symptom measures*. [GitHub repository](https://github.com/nivlab/sciops)

3. NIMH Data Science and Sharing Team. *OpenCogData*. [Project repository](https://github.com/nimh-dsst/OpenCogData) | [Dataset catalogue](https://nimh-dsst.github.io/OpenCogData/)

4. Raj, A., Musco, C., Mackey, L., & Fusi, N. (2020). *Model specific data subsampling with influence functions*. [arXiv:2010.10218](https://arxiv.org/abs/2010.10218)
