# Research synthesis: Compare ML methods for literature screening

> **Flagged as unverified.** accuracy retry budget exhausted - claims below have not passed full validation and require manual checking.

| Source | Methodology | Dataset | Reported accuracy |
|---|---|---|---|
| 10.26226/morressier.5f55fb7d6fdcfc6871991734 | not reported | not reported | not reported |
| 10.1021/acs.jpca.0c02647.s001 | not reported | not reported | not reported |
| 10.7717/peerj.19246/fig-2 | not reported | not reported | not reported |
| 10.31234/osf.io/nc8hs | nine different machine learning algorithms | abstracts from the field of psychology | not reported |
| 10.1021/acsaem.5c02609.s001 | not reported | not reported | not reported |

## Run trace

  0. **orchestrator** -> planner - no plan yet
  1. **planner** -> plan_created - 4 steps
  2. **orchestrator** -> retrieval - step 0 needs evidence
  3. **retrieval** -> retrieved - 5 papers via ['crossref', 'openalex']
  4. **orchestrator** -> processing - evidence retrieved, not yet processed
  5. **processing** -> extracted - 3 papers processed; 2 failed: ["10.26226/morressier.5f55fb7d6fdcfc6871991734: Malformed JSON in model response: Expecting ',' delimiter: line 1 column 168 (char 167)", "10.1021/acs.jpca.0c02647.s001: Malformed JSON in model response: Expecting ',' delimiter: line 1 column 144 (char 143)"]
  6. **orchestrator** -> validator - extractions await validation
  7. **validator** -> fail:accuracy - 1 of 3 claims could not be traced
  8. **orchestrator** -> processing - accuracy retry 1/3
  9. **processing** -> extracted - 5 papers processed
 10. **orchestrator** -> validator - extractions await validation
 11. **validator** -> pass - all claims traced to source
 12. **orchestrator** -> retrieval - step 1 of 4
 13. **retrieval** -> retrieved - 5 papers via ['crossref', 'openalex']
 14. **orchestrator** -> processing - evidence retrieved, not yet processed
 15. **processing** -> extracted - 5 papers processed
 16. **orchestrator** -> validator - extractions await validation
 17. **validator** -> fail:accuracy - 3 of 3 claims could not be traced
 18. **orchestrator** -> processing - accuracy retry 2/3
 19. **processing** -> extracted - 5 papers processed
 20. **orchestrator** -> validator - extractions await validation
 21. **validator** -> fail:accuracy - 3 of 3 claims could not be traced
 22. **orchestrator** -> processing - accuracy retry 3/3
 23. **processing** -> extracted - 5 papers processed
 24. **orchestrator** -> validator - extractions await validation
 25. **validator** -> fail:accuracy - 3 of 3 claims could not be traced
 26. **orchestrator** -> human_gate - escalated: accuracy retry budget exhausted
 27. **human_gate** -> approved - 5 validated papers | FLAGGED UNVERIFIED: accuracy retry budget exhausted
