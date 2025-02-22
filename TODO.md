@Shawn:
- [ ] A few concrete examples of drift vs no drift

- [x] What fraction of units show drift? 
     - 15% of annotated units (50% no drift, 35% unsure)

- [x] Which QC features correlate with annotated drift
    - presence_ratio (lower = more drift)
    - amplitude (larger = slightly more drift)
    - consistency of spike count over blocks, in baseline and in response

- [x] Describe the LDA classification metric
    - spike count in 3s from stim onset, Pearson corr with abs time of trial (https://codeocean.allenneuraldynamics.org/capsule/6421158/tree)
    - leave-one-out cross-validation on 5000 annotated units (excl. unsure)
    {
        'vis_response_r2': 1.438650980981107, 
        'aud_response_r2': 1.8042454303396904, 
        'presence_ratio': -1.511989383309867,
    }
- [x] Which brain areas have most drift? Directly assess with annotation, and also apply LDA to whole dataset.
- [x] Which experiments have the most drift? Directly assess with annotation, and also apply LDA to whole dataset.
    - 715710_2024-07-17 the most 
- [ ] Is KS4 better than KS25 regarding drift? 
    - need to actually annotate
- [x] Do KS4 have a lower LDA metric on average?

---
@Corbett:
- [x] Find a unit that appeared to be split in KS25 that's merged in KS4
        - 05392fb2-557a-4185-990b-8b94026d7eae
        - labelled as 'mua' (probability = 0.76)
        - (ks4 cluster id = 293, units table id = 574)
        
            - 742903_2024-10-21_E-179
            - 742903_2024-10-21_E-180

- [ ] Make a spike amplitude metric

---
@Ben 
- What does LDA predict for units annotated UNSURE - make a scatter plot with images
- How does amplitude correlate with consistency metric?
- Does amplitude variability correlate with consistency?

---
- [x] fix dashboard (5000 row limit)
- [x] launch another KS4 session with high drift
    - 715710_2024-07-17 (finishes Friday afternoon)
- [x] get spike times parquet for KS4 sessions 
    - [x] get timing data from npc_sessions, adjust times in units table
    - [x] export data asset with three sessions
- [x] create extra metrics for KS4 sessions 
    - [x] created branch
    - [x] waiting for export of data asset with third session parquet
- [x] make an annotation app for KS4 sessions
- [ ] explore difficulty of getting spike amplitudes
- [ ] what do false positives look like
- [ ] compare ks2.5 vs ks4 after thresholding on presence ratio
    - many more ks4 units with near-zero presence ratio