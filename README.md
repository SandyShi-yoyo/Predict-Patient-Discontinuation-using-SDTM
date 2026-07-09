================================================================================
Predicting Early Discontinuation in a Clinical Trial (CDISC Pilot 01 SDTM)
================================================================================

This project predicts which randomized subjects will discontinue a study
early, using only information available shortly after treatment start. It is
built on the public CDISC SDTM Pilot 01 dataset (DM, DS, AE, EX, LB, VS, MH,
QS domains).

Feeding all data (dm, ds, ae, ex, lb ,vs, mh, qs) in .xpt/.csv format and the prompt to 
Claude Sonnet 5 should allow the user to fully execute the plan and print key summary 
and model evaluation results. 

Repository contents
--------------------------------------------------------------------------
  
  prompt.txt                       The prompt provided to the Claude Sonnet 5 to generate the below codes 
  python code test/config.py               Shared RANDOM_SEED (123) and determinism helpers
  python code test/load_data.py             Step 1: load and clean the SDTM domains
  python code test/feature_engineering.py   Step 2: build baseline + time-varying features
  python code test/modeling.py               Step 3: train and evaluate the models
  requirements.txt          Pinned package versions

Environment (exact versions used to produce the results below)
--------------------------------------------------------------------------
  Python        3.12.3
  pandas        3.0.2
  numpy         2.4.4
  scikit-learn  1.8.0
  scipy         1.17.1
  pyreadstat    1.3.5

RANDOM_SEED = 123 is defined once in config.py and reused for every seeded
operation across all three scripts (numpy, random, train/holdout split,
cross-validation folds, IsolationForest, SVC/GridSearchCV, and the saga
solver used by LogisticRegressionCV), so the pipeline is exactly
reproducible end-to-end given this environment.

How to run the python code
--------------------------------------------------------------------------
  1. Place the SDTM domain files (.xpt or .csv) in a folder named "sdtm/"
     next to these scripts.
  2. pip install pandas==3.0.2 numpy==2.4.4 scikit-learn==1.8.0
     scipy==1.17.1 pyreadstat==1.3.5
  3. python modeling.py
     (this internally calls load_data.py and feature_engineering.py)

================================================================================
METHODOLOGY
================================================================================

1. Cohort definition
--------------------------------------------------------------------------
Screen-failure subjects (ARM == "Screen Failure") are removed. Randomized
subjects are those with a non-missing ARMCD. It is assumed that the 
randomization information are available.  A subject is labeled
discontinued (discont = 1) if their DS record's DSDECOD or DSTERM matches
one of: ADVERSE EVENT, DEATH, LACK OF EFFICACY, LOST TO FOLLOW-UP, PHYSICIAN
DECISION, PROTOCOL VIOLATION, STUDY TERMINATED BY SPONSOR, WITHDRAWAL BY
SUBJECT. All other randomized subjects are labeled discont = 0.

Resulting cohort: 254 randomized subjects (144 discontinued, 110 completed).

2. Baseline vs. time-varying information
--------------------------------------------------------------------------
Both are used, and are combined into a single feature table per subject:

  Baseline (fixed at/near study entry):
    - Demographics (age, sex, race, country, site)
    - Baseline labs (LBBLFL = 'Y', or earliest LBDTC with LBDY <= 1 if no
      baseline flag is present)
    - Baseline vitals (VSBLFL = 'Y', by test and time point)
    - Medical history (count of conditions, count flagged "severe")
    - Baseline questionnaire/scale scores, QS (QSBLFL = 'Y', or earliest
      QSDTC with QSDY <= 1)

  Time-varying (accumulated after treatment start, up to a per-subject
  cutoff date -- see below):
    - Adverse events: count, number serious, maximum severity
    - Exposure: total dose administered, number of dosing days
    - Lab trends: mean/max/min for each lab test
    - Vital-sign trends: mean/max/min for each vital-sign test

  Anomaly-detection features (derived from the combined table): an
  Isolation Forest anomaly score/flag (iso_score, iso_flag) computed over
  all numeric baseline + time-varying features, plus a narrower anomaly
  score/flag computed on the AE features alone (isoae_score, isoae_flag).

3. Time-varying cutoff window
--------------------------------------------------------------------------
Yes -- a single cutoff window (in weeks after RFSTDTC, each subject's
reference start date) is applied uniformly to every randomized subject, so
that the same "look-back horizon" is used for everyone and no subject's
features can peek past the point where the earliest true discontinuation in
the data actually occurred. Concretely:

  cutoff_weeks = MIN( (RFENDTC - RFSTDTC) in weeks ),
                 taken ONLY over subjects with discont == 1, EXCLUDING any
                 subject whose DS dsterm == 'PROTOCOL ENTRY CRITERIA NOT MET'
                 (these are early exclusions/screen-fail-adjacent subjects,
                 not genuine on-treatment discontinuations, so they are
                 excluded from the cutoff calculation to avoid an
                 artificially short window).
  cutoff_weeks = max(cutoff_weeks, 0)   [floored at 0]

  cutoff_date(subject) = RFSTDTC(subject) + cutoff_weeks

For this run, cutoff_weeks = 0.57 weeks (~4 days). All AE, EX, LB, and VS
records used for time-varying features are restricted to
record_date <= cutoff_date for that subject. This is a deliberately
conservative (short) window: it reflects the fastest observed
discontinuation in the data, so the model is evaluated on genuinely early,
practically actionable signal rather than on data that would only be
available after some subjects had already left the study.

4. Outcome type: binary vs. time-to-event
--------------------------------------------------------------------------
Discontinuation is treated as a BINARY outcome (discont: 0/1), not a
time-to-event outcome. No censoring, hazard, or survival model
(Cox/Kaplan-Meier/etc.) is used; the question the models answer is
"will this subject discontinue at some point," not "when." This is a
simplification: it does not account for differential follow-up time or
right-censoring, which a time-to-event framing (e.g., a Cox proportional
hazards model on the same cutoff-window features) would handle more
rigorously and could be a natural extension of this project.

5. Models
--------------------------------------------------------------------------
Two models are trained on identical preprocessed features and compared:

  a) Elastic-net logistic regression (sklearn LogisticRegressionCV,
     solver="saga", 5-fold stratified CV over regularization strength C and
     l1_ratio in {0.1, 0.5, 0.9}, selected by CV AUC).
  b) RBF-kernel SVM (sklearn SVC, 3-fold stratified CV grid search over
     C in {0.5, 1, 2} and gamma in {"scale", "auto"}, selected by CV AUC).

Preprocessing (fit on the development set only, applied unchanged to
holdout): median imputation of numeric predictors, one-hot encoding of
nominal predictors, removal of zero-variance numeric columns, and
standardization (both models are scale-sensitive).

Data are split 70% development / 30% holdout using a stratified split on
the outcome (177 development / 77 holdout; holdout: 44 discontinued / 33
completed).

6. Explainability
--------------------------------------------------------------------------
Yes, both models are explainable, using two different but complementary
attribution methods appropriate to each model type:

  - Logistic regression: attribution is direct. Each predictor has a
    standardized regression coefficient; exponentiating it gives an odds
    ratio, which is directly interpretable as "how much the odds of
    discontinuation change per one-standard-deviation change in this
    predictor" (or presence of a category, for one-hot-encoded predictors).
    This is a fully transparent, coefficient-based attribution -- no
    approximation is needed.

  - SVM (RBF kernel): the RBF kernel is nonlinear, so there is no single
    global coefficient per feature. Instead, feature importance is
    estimated with holdout-set permutation importance: each feature is
    independently shuffled and the resulting drop in holdout AUC is
    measured (20 repeats per feature, seeded for reproducibility). Larger
    AUC drop = more important feature. This is a model-agnostic,
    post-hoc attribution method, not an intrinsic property of the SVM.

================================================================================
RESULTS
================================================================================

Holdout-set evaluation (n = 77; 44 discontinued / 33 completed):

  model      AUC     PR-AUC   Accuracy  Sensitivity  Specificity
  --------   ------   ------   --------   -----------   -----------
  logistic   0.655    0.715    0.649      0.864         0.364
  svm        0.576    0.635    0.584      0.977         0.061

Confusion matrix - logistic regression (rows = truth, cols = predicted;
0 = completed, 1 = discontinued):
              pred:0   pred:1
    truth:0     12       21
    truth:1      6       38

Confusion matrix - SVM (RBF):
              pred:0   pred:1
    truth:0      2       31
    truth:1      1       43

Best hyperparameters:
  Logistic regression: C = 0.0127, l1_ratio = 0.1 (i.e., regularization is
    dominated by the L2/ridge component, with light L1/lasso sparsity).
  SVM: C = 0.5, gamma = "auto".

Top predictors -- logistic regression (standardized coefficient / odds
ratio):
  1. dose_total                                    OR = 1.135
  2. armcd_Pbo (placebo arm)                        OR = 0.881
  3. qs_daitm18 (questionnaire item)                 OR = 1.067
  4. qs_npitm08s (questionnaire item)                 OR = 1.066
  5. qs_mmitm02 (questionnaire item)                  OR = 0.952
  6. qs_mhitm06 (questionnaire item)                  OR = 1.048
  7. lb_min_ery_mean_corpuscular_hgb_concentration    OR = 0.958
  8. qs_daitm31 (questionnaire item)                  OR = 1.041
  9. qs_daitm32 (questionnaire item)                  OR = 1.041
  10. qs_daitm15 (questionnaire item)                 OR = 0.964

Top predictors -- SVM (permutation importance, holdout AUC drop):
  1. dose_total            0.0147
  2. armcd_Pbo              0.0147
  3. armcd_Xan_Hi            0.0059
  4. qs_daitm03              0.0057
  5. armcd_Xan_Lo             0.0054
  6. qs_daitm38               0.0050
  7. qs_daitm33               0.0049
  8. ae_sev_max                0.0048
  9. qs_daitm37                0.0045
  10. siteid_716                0.0044

Anomaly detection (Isolation Forest, over the full combined feature table):
  37 of 254 subjects (14.6%) flagged as anomalous.
  Features most distinguishing flagged vs. non-flagged subjects
  (standardized mean difference): Hemoglobin A1C (baseline and all lab
  trend variants, |effect size| ~= 1.37), followed by several QS items
  (qs_NPTOT, qs_NPITM06F, qs_NPITM03S, qs_ACTOT, and others, |effect size|
  ~= 1.0-1.3).

================================================================================
CONCLUSIONS
================================================================================

- Logistic regression clearly outperforms the SVM on this task: higher AUC
  (0.655 vs. 0.576), higher PR-AUC, higher accuracy, and a much more
  balanced confusion matrix. The SVM is heavily biased toward predicting
  "discontinued" (74 of 77 holdout predictions), which inflates its
  sensitivity (0.977) at the cost of near-zero specificity (0.061) -- it is
  not meaningfully discriminating between the two classes, just calling
  almost everyone positive.

- Both models' top predictors are dominated by ARM/ARMCD (treatment
  assignment) and dose_total (cumulative dose administered), rather than
  by baseline clinical characteristics. This is an important caveat: ARM
  and total dose are largely a function of how long/how much a subject was
  actually treated, which is mechanically entangled with whether they
  discontinued early (a subject who discontinues sooner generally
  accumulates less total dose and, on this particular pilot dataset, arm
  assignment happens to correlate with discontinuation risk). This makes
  the model's practical AUC advantage look partly like it is picking up on
  a proxy for the outcome itself rather than on genuinely predictive
  pre-treatment or very-early risk factors. A natural next step (and one
  we generated a variant prompt/scripts for separately) is to re-run the
  pipeline with ARM/ARMCD and any total/cumulative-dose feature excluded
  from the feature table, to see how much signal remains from baseline
  labs, vitals, medical history, QS scores, and AE/exposure-duration
  features alone.

- After the ARM/dose-related predictors, the next most influential
  features are individual QS (questionnaire/rating-scale) items and,
  for the SVM, AE severity and site -- suggesting patient-reported/
  clinician-rated scale items and early adverse-event severity carry real,
  if modest, standalone signal for early discontinuation risk.

- The anomaly-detection layer flags 14.6% of subjects as outliers on the
  combined feature space, most strongly driven by Hemoglobin A1C and
  several QS items. This is a useful complementary, unsupervised check
  (e.g., for data-quality review or highlighting atypical subjects) but is
  not itself a discontinuation predictor -- it was not used as a target or
  fed back into the supervised evaluation above.

- The outcome is modeled as a binary discontinuation flag rather than
  time-to-event; a survival/Cox-model formulation on the same cutoff-window
  features would better account for censoring and differential follow-up
  and is a reasonable extension of this analysis.

- Overall model discrimination is modest (AUC 0.58-0.66). With a very
  short 0.57-week feature-accumulation window (dictated by the fastest
  observed discontinuation in this dataset) and a small cohort (254
  subjects, 328 model-ready predictors after encoding), these results
  should be read as a proof-of-concept pipeline rather than a
  clinically validated early-warning model.

================================================================================
Reproducibility note
================================================================================
All results above were generated with RANDOM_SEED = 123 (config.py) under
Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1,
and pyreadstat 1.3.5 -- the exact versions verified as installed at
execution time. Re-running modeling.py in this environment against the same
sdtm/ input files will reproduce these numbers exactly.
