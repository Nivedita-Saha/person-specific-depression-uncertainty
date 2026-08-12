# Person-Specific, Uncertainty-Aware Depression Assessment from Expressive Behaviour

Estimating depression severity (PHQ-8) from expressive-behaviour features - facial action units, gaze, and head pose - using a **person-specific** design: each session is modelled as an individual's deviation from their own behavioural baseline, rather than against a population average.

The project adds a trustworthy-AI layer on top of the estimator:
- **Calibrated uncertainty** on every prediction (MC dropout + conformal intervals)
- **Explainability** via action-unit attribution
- **Fairness** analysis across demographic groups
- An **honest baseline comparison** against a direct behaviour-to-label model

## Status
Pipeline under active development and validated on synthetic data; clinical evaluation on DAIC-WOZ / E-DAIC in progress (dataset access requested).

## Data
DAIC-WOZ / E-DAIC (Gratch et al., 2014; Ringeval et al., 2019), released to academic researchers on request. Raw data is never committed to this repository.