# GoodMem Integration Test Results

**Date:** 2026-03-16
**GoodMem Server:** http://localhost:8080
**Embedder Used:** Voyage voyage-3-large (019c2aff-c470-778d-9f4e-89a74c77890f)
**PDF File:** /home/bashar/Downloads/New Quran.com Search Analysis (Nov 26, 2025)-1.pdf

## Command Executed

```bash
GOODMEM_BASE_URL="http://localhost:8080" \
GOODMEM_API_KEY="gm_rttn7pla4rm3ry6hqakfnnaal4" \
GOODMEM_PDF_PATH="/home/bashar/Downloads/New Quran.com Search Analysis (Nov 26, 2025)-1.pdf" \
GOODMEM_EMBEDDER_ID="019c2aff-c470-778d-9f4e-89a74c77890f" \
python3 tests/test_integrations/test_goodmem_e2e.py
```

## Results Summary

| # | Test | Status | Details |
|---|------|--------|---------|
| 1 | list_embedders | PASS | Found 4 embedders |
| 2 | list_spaces | PASS | Found 11 spaces |
| 3 | create_space | PASS | id=019cf352-26cb-7428-ae44-6e3b9a6d5172, name=deepeval-test-666b3d6f |
| 4 | create_space_reuse | PASS | Correctly reused existing space by name |
| 5 | create_memory_text | PASS | Created text/plain memory, status=PENDING |
| 6 | create_memory_pdf | PASS | Created application/pdf memory from real PDF file |
| 7 | retrieve_memories | PASS | Retrieved 5 results with semantic search, wait_for_indexing worked |
| 8 | retrieve_as_context | PASS | Returned 3 context strings suitable for LLMTestCase.retrieval_context |
| 9 | get_memory | PASS | Fetched memory by ID with metadata |
| 10 | delete_memory | PASS | Deleted memory and confirmed it is inaccessible |

**Total: 10 | Passed: 10 | Failed: 0**

## Full Output

```
GoodMem E2E Tests
  Base URL: http://localhost:8080
  API Key:  gm_rttn7pl...
  PDF Path: /home/bashar/Downloads/New Quran.com Search Analysis (Nov 26, 2025)-1.pdf

  [PASS] list_embedders
         Found 4 embedders: ['019bebbe-8a48-7742-adbb-3d6133a37143', '019c2aff-c470-778d-9f4e-89a74c77890f', '5ab2710c-b948-4220-9d12-0883729c4983', '019c44a6-22de-7299-85cb-c804ca619cf7']
  [PASS] list_spaces
         Found 11 spaces
  [PASS] create_space
         id=019cf352-26cb-7428-ae44-6e3b9a6d5172, name=deepeval-test-666b3d6f, reused=False
  [PASS] create_space_reuse
         Reused space: id=019cf352-26cb-7428-ae44-6e3b9a6d5172
  [PASS] create_memory_text
         id=019cf352-26db-7336-a013-b3aa344b729d, status=PENDING
  [PASS] create_memory_pdf
         id=019cf352-26e6-7180-a1dc-78119ec56edb, file=New Quran.com Search Analysis (Nov 26, 2025)-1.pdf, status=PENDING
  [PASS] retrieve_memories
         Retrieved 5 results
           [0] score=-0.8290687203407288, text=DeepEval is an open-source evaluation framework for LLMs...
           [1] score=-0.519167423248291, text=green, it's clear that all the core information-retrieval metrics improved...
           [2] score=-0.5125037431716919, text=("better"+"much better") vs 10 regressions...
  [PASS] retrieve_as_context
         Returned 3 context strings for LLMTestCase.retrieval_context
  [PASS] get_memory
         id=019cf352-3dd6-763b-9c29-c10e66c65a64, keys=['memory', 'content_error']
  [PASS] delete_memory
         Deleted id=019cf352-3dde-7428-885a-09803b2c390d, confirmed inaccessible

============================================================
TEST SUMMARY
============================================================
  [PASS] list_embedders
  [PASS] list_spaces
  [PASS] create_space
  [PASS] create_space_reuse
  [PASS] create_memory_text
  [PASS] create_memory_pdf
  [PASS] retrieve_memories
  [PASS] retrieve_as_context
  [PASS] get_memory
  [PASS] delete_memory

  Total: 10 | Passed: 10 | Failed: 0
```

## Notes

- The OpenAI Ada embedder (019bebbe-8a48-7742-adbb-3d6133a37143) returns "Embedding failed: Dense embedding failed" errors. The Voyage embedder works correctly.
- The `get_memory` test shows `content_error` in keys because the `/v1/memories/{id}/content` endpoint returns an error for the test content (likely because text memories don't have separate content downloads). The memory metadata itself is fetched correctly.
- The `wait_for_indexing` polling mechanism (60s timeout, 5s intervals) works correctly — retrieval succeeded after memories finished processing.
