# Fix log

Every pipeline failure and its fix, in order.

## 2026-11-01 00:34, dry run, stage train_a_resnet50_fold0
- Error: URLError SSL UNEXPECTED_EOF_WHILE_READING downloading ImageNet weights from pytorch.org through the VPN.
- Fix: new stage s0_weights.py caches and hash-checks all weights before training, with retries and a curl fallback.
- Effect on results: none.

## 2026-11-01 00:45, dry run, stage evaluate_a_final
- Error: planted on purpose (bootstrap set to the text "many") to test stop, wake-up and resume.
- Result: stopped at once, woke the session, resumed without redoing finished stages.
- Effect on results: none.

## 2026-11-01 01:11, full run, fetch
- Error: AttributeError in the download retry handler. ProxyConnectionError carries response=None, which crashed the handler instead of retrying.
- Fix: handler accepts a missing response. Connections reduced from 64 to 32, because measured speed was equal (6.4 vs 6.2 MB/s) and it loads the VPN less.
- Effect on results: none. 220 images already fetched were kept.

## 2026-11-01 02:30-03:05, Kaggle jobs
- Measured on Kaggle (2x T4, 4 CPUs): download 205 MB/s, 34 img/s; decode 2.2 img/s per process, so about 7-8 img/s overall. PC through VPN: 1.37 img/s.
- Decision: BI-RADS (3 runs) and part of malignant vs benign (convnext_tiny folds 0-4, efficientnet_v2_s folds 2-4) moved to Kaggle. PC keeps running and downloading as the fallback. kaggle/import_results.py copies finished runs in and writes their stage markers, so the PC skips them. Runs the PC has already started stay local.
- Error: kaggle_secrets returned HTTP 400 in CLI-pushed notebooks. Fix: 24-hour presigned download links in a private dataset. Each link holds the AWS access key ID and a signature, not the secret key.
- Error: Kaggle unpacked links.csv.gz on upload, FileNotFoundError in fetch. Fix: plain links.csv.
- Error: kaggle datasets create failed on a relative path containing a slash. Fix: run the CLI from inside the folder.
- Error: VPN reset the connection during kernels push. Fix: pushed again with retries.
- Effect on results: none. Same preprocessing, splits, seeds, hyperparameters and code; loader_workers is 2 on Kaggle. Hardware per run is recorded in config.json.
- 03:08: tried a third job to run BI-RADS ConvNeXt in parallel. Refused: "Maximum batch GPU session count of 2 reached". Running jobs not affected; make_jobs.py put back to what is running.

## 2026-11-01 03:26, VPN data limit
- Problem: user's VPN plan is 150 GB and is 52% used; traffic may also go through a phone hotspot. The local S3 download was using about 32 GB per hour and the remaining images would have needed about 73 GB, more than what is left.
- Fix: stopped the local download at 11,209 images. Added run_pipeline.py --no-fetch: it starts no download, and BI-RADS waits for the runs imported from Kaggle instead of training locally. EfficientNetV2-S fold 0 resumes from its last saved epoch.
- Effect on results: none for malignant vs benign, all its images are local. The PC is no longer a fallback for BI-RADS. Local PNGs cover only part of BI-RADS, so figures that use local images must not be read as the full cohort.
- 03:27 error on resume: TypeError "RNG state must be a torch.ByteTensor". The checkpoint is loaded onto the GPU, and the saved random state went with it. This was a real bug in mid-run resume. Fix: move it to the CPU before restoring (s3_train.py). Running Kaggle jobs still have the old line, so a crash there would not resume until re-pushed. Effect on results: none, training continued from its saved epoch.

## 2026-11-01 03:32, crop problem found (not fixed yet)
- Check: scripts/check_crop.py on 40 random original DICOMs, picture in outputs/proof/crop_check.png, numbers in crop_check.csv and crop_check.json.
- Result: median 0.5% of non-black pixels fall outside the crop box, mostly markers and labels. But 4 of 40 had 7.6%, 18%, 32% and 52% outside. The picture shows real tissue cut in an implant image and in MLO views where only the bright part was kept.
- Cause: the Otsu threshold in preprocess() in s2_fetch_images.py can separate a very bright region (implant, dense tissue) from the rest of the breast, and the crop then follows only that region.
- Affects: all images made so far, on the PC and in both running Kaggle jobs. Runs so far must be treated as a baseline with this flaw.
- Measured sizes (medians): original 4096x3328, crop 3147x1550. Shrink from crop: x4.1 at 768x512, x6.2 at 512x384, x2.1 at 1536x1024.

## 2026-11-01 03:46, crop fixed, second pass started (v2)
- Crop fix: crop_box() in s2_fetch_images.py uses a fixed low level (10/255) instead of Otsu. Same 40 images: worst outside share went from 52.4% to 5.2%, median from 0.48% to 0.16%. Picture in outputs/proof/crop_check.png. Also the VOI window now gets integer pixels, which removes a pydicom warning about possibly wrong results on LUT images.
- Resolution: malignant vs benign 1536x1024 (batch 4, accum 6), BI-RADS 1024x768 (batch 8, accum 3). Effective batch stays 24. PNG long side 1536 and 1024, set by EMBED_LONG_SIDE.
- Old Kaggle jobs (version 2) stopped from the Kaggle page because they used the flawed crop. New jobs are version 3: all 15 malignant vs benign runs and 3 BI-RADS runs. Each job also makes its own dev metrics and figures, since the images are there. Test set stays locked.
- Results land in outputs_v2/ through kaggle/import_results.py. outputs/ stays as the first-pass baseline (768x512, old crop) from the PC, no VPN use.
- First-pass BI-RADS at 512x384 was dropped, so there is no BI-RADS baseline to compare with.

## 2026-11-01 04:05, conversion made about 3x faster
- Found: Kaggle converted only 3 img/s, not the 7-8 I estimated. Timing per image on the PC: decode 0.13 s with dicomsdl (identical pixels on 12 of 12 images, 2.7x faster than GDCM), windowing 0.55 s before and 0.11 s now through a lookup table (identical on 10 of 10 images), total 0.28 s.
- embed-birads restarted as version 4 with this code, 20 minutes of fetch lost. embed-malignant version 3 kept running on the older code because it had nearly finished its images. Pixels match, so the two jobs are comparable.

## 2026-11-01 BI-RADS job finished; report step failed on Kaggle
- Found: s4_evaluate exit 1 on Kaggle, KeyError 'Manufacturer'. The Kaggle metadata is trimmed and has no covariate columns. Training results were safe.
- Fix: evaluate on the PC. Added EMBED_RUN and EMBED_CONFIG (configs/kaggle_v2.json), load_metadata always reads outputs/cohort, import_results skips the trimmed cohort and merges fetched.csv.
- The running malignant job has the same report failure. Harmless; same fix applies after import.
- Figures: scripts/fetch_figure_images.py fetches only the gallery images with the fixed crop (72 images, 431 MB for BI-RADS). s5 galleries use images on disk.

## 2026-11-01 Grad-CAM finding: attention on markers, not only tissue
- Seen in outputs_v2/figures/b/gradcam_dev.png: heat on triangle skin markers, nipple BBs and burned-in marks at the image edge.
- Measured (scripts/metadata_baseline.py, dev, no images): exam type alone reaches accuracy 42.8%, macro AUC 0.735; image stack reaches 42.7%, 0.759.
- Cause: BI-RADS 0 and 1 come from screening, 3, 4, 5 from diagnostic exams (crosstab in the log). The images reveal the exam type through markers.
- So BI-RADS image scores overstate reading skill. Not fixed in this run; masking markers needs retraining (next time).

## 2026-11-01 Improvement with no training: exam type and age in the stacker
- Tried pooling too (scripts/try_stack_inputs.py): max vs mean logits within noise, not applied.
- Added stack_clinical beside the image-only stack in s4, chosen on dev before the test was opened, reported separately. Figure clinical_compare_*.

## 2026-11-01 Radiologist's reading added as a separate score (no training)
- The clinical table holds lab results (path1-10, grade, stage, receptors, surgery): these are the answer and are never inputs. It also holds what the radiologist recorded before biopsy: mass shape, margin, density, calcifications, other findings, BI-RADS.
- Added stack_reading in s4, malignant vs benign only, chosen on dev with the test closed. Feature columns come from dev only. Not used for BI-RADS, where it would hand the model its label.
- Cancer count (scripts/clinical_field_trial.py): the open table has 658 patients with confirmed cancer; Task A uses 517. Patients in EMBED cannot grow this much.

## 2026-11-01 Marker test on the laptop GPU (outputs_masked/)
- Question: does malignant vs benign lean on skin markers? On dev, exams with a detected triangle or circle marker are 41% cancer against 31% without.
- Method: ConvNeXt-Tiny, same folds, config and images as the PC baseline, with burned-in marks blacked out and hollow markers inpainted (scripts/s2_fetch_images.py remove_marks, EMBED_MASK_MARKS). Compared on dev by scripts/compare_masked.py. Test not read.
- Limits seen in outputs_masked/proof/mask_zoom.png: removes triangles and circles well, also some rim calcifications that are real findings, and misses solid BBs. Detected on 8% of images.

## 2026-11-01 Two waiters downloaded the same output at once
- Found: first version of the malignant waiter (started 03:01) was never stopped when the job was re-pushed. Both it and the 03:46 waiter ran `kernels output --force` into the same folder at 12:38, then both restarted from scratch at 12:47 and 12:48 after a failure.
- Effect: about 2.5 MB/s for 17 min (about 2.5 GB of VPN data) with 0.45 GB kept. Both stopped at 12:55.
- Fix: kaggle/fetch_output.py: one process, resumes a file after a drop, keeps finished files, shows percent. Skips last.pt (about 4 GB saved); best.pt can warm-start and last.pt stays on Kaggle (--all fetches it).
- Rule: stop the old waiter whenever a job is re-pushed; check with a process listing.

## 2026-11-01 Marker test result
- ConvNeXt-Tiny dev AUC 0.698 baseline, 0.688 with markers removed; difference -0.009 (95% CI -0.037 to 0.017). No clear evidence the malignant vs benign score relies on the removed markers; the test could not rule out a small effect. BI-RADS was not tested.

## 2026-11-01 Malignant vs benign finished
- Report step failed on Kaggle as expected (trimmed metadata); evaluated on the PC.
- Dev: baseline stack 0.719, Kaggle stack 0.723. Averaging both raised dev AUC to 0.737 (difference +0.014, CI 0.002 to 0.025), so both were combined. Decisions written to outputs_v2/proof/test_plan_malignant.md before the test was opened.
- Test (348 exams, 112 cancers, opened once): combined images 0.776 (0.725-0.826), + exam type and age 0.799, + radiologist's findings 0.826. Test scores run above dev, partly because each test exam gets the average of 5 fold models. CIs are wide.
- Removed duplicates after checking file counts: kaggle/embed-*_out (imported to outputs_v2), dry_run checkpoints and images. data/embed/png kept until lesion-finding is decided.
