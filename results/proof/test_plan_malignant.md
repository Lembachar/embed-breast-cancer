# Malignant vs benign: decisions fixed before opening the test set

Written 2026-11-01, before any malignant vs benign test score was computed. All choices below come from dev only.

- Main image-only model: average of the baseline and second-pass stacks (6 models). Dev AUC 0.737.
- Also reported: images with exam type and age; images with exam type, age and the radiologist's recorded findings.
- Also reported for comparison: baseline stack alone, second-pass stack alone.
- Thresholds and the feature list come from dev and are not changed after the test is opened.
- The test set is opened once. No model, threshold or input is changed because of a test result.
