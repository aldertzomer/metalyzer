# Why BioSample benchmark performance differs from TSV input

Diagnostic report, 12 October 2026. This report uses the six completed 1,320-row
benchmarks, the corresponding local BioSample metadata, reconstructed model
records, and small controlled NLI experiments. Run IDs were joined explicitly;
BioSample accession order and the duplicate request were checked. Production
classification code and benchmark outputs were not changed during this audit.
No additional Mistral API requests or network metadata queries were made.

**The main finding is a mixture of data disagreement and genuine model weakness.**
The lower BioSample score does not mean every additional error is a failure to
retrieve or classify the local metadata. Some reference labels disagree with
explicit BioSample source fields. NLI also has a substantial, independently
demonstrated weakness with scientific host names and record formatting.

| Method | TSV correct | BioSample correct | TSV correct → BioSample wrong | TSV wrong → BioSample correct |
|---|---:|---:|---:|---:|
| NLI | 1,137 | 805 | 368 | 36 |
| Local Ministral | 1,215 | 1,133 | 134 | 52 |
| Mistral API | 1,284 | 1,174 | 119 | 9 |

These are the existing reference-label scores, not revised biological labels.
See [error_summary.tsv](error_summary.tsv) and the complete
[record comparison](record_comparison.tsv).

**1. The reference labels and BioSample source metadata disagree in identifiable groups.**

There are **44 reference-goat records** whose local BioSample metadata says:

- `host = Gallus gallus`
- `ifsac_category = chicken`
- original submitted host also `Gallus gallus`

For these same runs, the TSV has `host_scientific_name = Capra hircus` and
`host_tax_id = 9925`. All 44 are resolved as goat by TSV taxonomy. Both local
Ministral and Mistral API call all 44 chicken from BioSample metadata. That is
consistent with the metadata they receive, although scored wrong against the
reference label.

Example: `SRR26351296` → `SAMN37735409`. The sample identifier/alias also agrees
between inputs (`009LG01AL`); the source fields do not. This is a source-data
conflict, not a demonstrated accession lookup or row-order error. The BioSample
row has a later update date than its original submission, but this audit cannot
establish whether the TSV, the current BioSample annotation, the reference label,
or an upstream association is biologically correct. Review the original study
and provenance before changing any labels.

There are also **41 reference-unknown records with explicit human evidence** in
BioSample `host`, `description`, or `source_type`. Mistral API calls all 41 human;
local Ministral calls 37 human, with the other four remaining unknown. Examples:

| Run | BioSample | Reference | BioSample evidence |
|---|---|---|---|
| `ERR3145269` | `SAMEA5326284` | unknown | `host = Homo sapiens` |
| `SRR13169053` | `SAMN16961722` | unknown | `source_type = human`, while host/isolation source are missing |
| `ERR11003616` | `SAMEA112765475` | unknown | description: `Campylobacter jejuni isolates from human` |

The TSV lacks these explicit fields in the corresponding examples. Here,
`unknown` appears to reflect unavailable evidence in one metadata representation,
which is different from establishing that the sample's biological source is
unknown in all representations. The other seven reference-unknown records called
human by the API are not explained by these three explicit evidence criteria;
they need separate review rather than automatic relabeling.

The 44 goat conflicts and 41 human-evidence/unknown cases are disjoint. Together,
they contain **81 of 187 scored local LLM errors** and **85 of 146 scored API
errors**. This is not permission to mark the predictions correct: it identifies
cases where the reference benchmark requires a provenance audit. No adjusted
accuracy or automatic ground-truth correction has been applied.

Supporting tables: [goat_host_conflicts.tsv](goat_host_conflicts.tsv),
[unknown_human_evidence.tsv](unknown_human_evidence.tsv).

**2. NLI genuinely struggles with scientific species names and terse metadata text.**

Large NLI error groups include turkey→cat (61), other_animal→environment (53),
wildbird→environment (53), sheep→goat (28), and pig→goat (18). Generative methods
classify all 101 turkey records correctly from the same BioSample input.

I tested minimal texts with the unchanged NLI model, full `sources.tsv` candidate
labels, and threshold 0.2. The following contrasts change the surface form of the
host name while retaining the same biological meaning and isolation source:

| Input contrast | NLI prediction with Latin host | NLI prediction with common host |
|---|---|---|
| `Meleagris gallopavo` vs `turkey`; isolation source `Animal-Turkey-Young Turkey` | goat | turkey |
| `Gallus gallus` vs `chicken`; isolation source `feces` | goat | chicken |
| `Ovis aries` vs `sheep`; isolation source `feces` | goat | sheep |
| `Sus scrofa domesticus` vs `pig`; isolation source `feces` | sheep | pig |

`Felis catus` works in this small test, so this is not a universal Latin-name
failure. Formatting also matters: `host: Non-human primate; isolation source:
feces` was classified wastewater, while `The sample is feces from a non-human
primate.` was classified other_animal. `isolation source: Goose faeces` was
classified wastewater, whereas an explicit sentence specifying a wild bird
was classified wildbird. The goose sentence adds information about wildness,
so that contrast is not purely a formatting experiment.

These tests demonstrate a model/text-representation problem independently of
Parquet retrieval. They do not identify an internal neural mechanism or establish
that replacing all names or rewriting all records would improve a full benchmark.
The long candidate descriptions and scientific-name recognition merit a dedicated
validation set.

See [nli_sentinel_texts.tsv](nli_sentinel_texts.tsv).

**3. The extra BioSample administrative fields contribute noise, but removing them is not a complete fix.**

BioSample text contains IDs, timestamps, owner, model, package, links, and original
attribute aliases alongside the biological evidence. Median rendered records
contain 40 fields / 1,538 characters, versus 13 fields / 562 characters for TSV.
There are 114 distinct `original_attributes.*` columns across the selected data;
submitted aliases often repeat already-present canonical values.

For 642 records, the package display name is `Pathogen: environmental/food/other;
version 1.0`, regardless of whether their true source is a turkey, goose, dog,
or another host. Generic registration-profile text is not evidence that the
sample was collected from an environmental source. It occurs alongside the
similarly generic title and `model = Pathogen.env`, which can compete with the
actual host/source text.

In a deliberately selected diagnostic set of 20 records (18 scored failures and
two correct controls), rerunning the original text reproduced all 20 saved NLI
calls. Removing only `title`, `model`, `package`, and `package_display_name`
raised reference correctness from 2/20 to 5/20: three other-animal→environment
errors were corrected. Most other errors remained. Reducing text to source
fields alone produced 3/20, including an accidental reference-goat call despite
`host = Gallus gallus`; it was not a reliable remedy.

This supports a limited causal contribution from profile fields in those selected
cases. The sample was chosen to study errors and is not a representative accuracy
estimate. Do not solve this by deleting raw metadata, imposing an attribute
whitelist, or assuming an aggressively reduced prompt is always better.

See [nli_ablation.tsv](nli_ablation.tsv).

**4. BioSample and TSV contain different useful information.**

BioSample does not include all sequencing experiment and study fields present in
`benchmark.tsv`. Missing context explains some otherwise ambiguous source calls.

Example: `SRR9030816` → `SAMN11605015`, reference water. Both inputs have
`isolation_source = Troups Creek Wetland`. The TSV additionally has study title
`Understanding the sources and risks of Campylobacter to stormwater constructed
wetlands`. BioSample lacks that stormwater context and contains generic
`Pathogen.env` registration fields. All three BioSample methods call environment.
This is consistent with an information difference and a water/environment
boundary ambiguity; a controlled study-title addition would be needed to prove
which field caused the different call.

Conversely, BioSample sometimes supplies *more* informative source evidence than
TSV, as the reference-unknown examples above demonstrate. It is therefore not
valid to characterize one input as uniformly richer or more correct.

**5. Host taxonomy protection is largely unavailable in the BioSample representation.**

All six runs loaded taxonomy successfully. TSV taxonomy resolved 215 records,
all matching their reference labels. No BioSample rows had `host_tax_id` in the
extracted dataframe, so those records instead reached the text models.

The adapter retains a canonical field named `host_taxid`, which is not recognized
by the existing deterministic stage. This is a real integration gap. However,
only two rows have nonempty values in that field (`9913` and `8839`); using it as
an alias produced one configured taxonomy assignment, which was already correct
under every saved model. Replacing that one saved call has zero effect on the
reported accuracies. Fixing this spelling alone does not restore the TSV's 215
assignments.

| Records resolved by TSV taxonomy | NLI BioSample correct | LLM BioSample correct | API BioSample correct |
|---|---:|---:|---:|
| 215 | 133 | 171 | 171 |

The 44 generative errors in this group are exactly the goat/chicken metadata
conflicts described above. Taxonomy protection therefore overlaps with the data
conflict issue; its effect must not be added to those 44 as an independent loss.
Outside this group, BioSample still loses a net 250 correct NLI calls, 38 local
LLM calls, and 66 API calls compared with TSV. Missing numeric host taxonomy is
not the whole explanation.

**6. Truncation and score filtering are secondary factors, not demonstrated root causes of the full gap.**

- Metalyzer's 2,000-character limit affected 109 BioSample records and zero TSV
  records. Those 109 records had 99 correct NLI calls and 103 correct calls for
  each generative method—better than the rest of the dataset. This association
  does not prove truncation is harmless, but contradicts treating it as the main
  observed explanation for poor overall accuracy.
- With the cached NLI tokenizer's 512-token limit, 412 BioSample records exceed
  the premise-plus-hypothesis limit for at least one full candidate label; zero
  TSV records do. NLI truncates the premise while retaining the hypothesis.
  Median record token counts are 432 for BioSample and 154 for TSV. Token
  truncation is an additional risk, especially for later attributes, but these
  counts are not a causal estimate. Priority host/source fields already come
  first, and the minimal-text experiments fail without any long context.
- Local LLM filtering produced 75 low-score abstentions. The rejected label
  agrees with the reference in 21 cases and disagrees in 54; 11 rejected cases
  have reference unknown. The default 0.75 threshold sacrifices some recall
  while preventing incorrect calls. Lowering it is not a general solution to
  conflicting metadata or high-confidence wrong predictions.

**7. NLI verification is not an independent ground-truth check.**

Against the existing references, 202 wrong NLI calls, 78 wrong local LLM calls,
and 90 wrong API calls have verification scores ≥0.9. Some of these are metadata
conflicts where NLI reasonably supports the explicit input but the reference
differs. There are also low scores for correct predictions: the API's correct
wildbird call for `SAMN06270804` has verification around 0.0036. Verification
uses the same NLI model and cannot resolve an external data discrepancy or be
assumed to correct that model's own recognition failures. It is currently logged,
not used as a classification veto.

**What to address first**

1. Audit the 44 goat/chicken conflicts and 41 reference-unknown/human-evidence
   records against original study/sample provenance. Decide whether references
   represent biological source or what could be inferred from a particular TSV
   export. Keep the original scores visible; do not silently relabel cases.
2. Normalize genuine host-taxonomy field aliases, then investigate resolving
   unambiguous scientific host names through the local taxonomy. Preserve the
   submitted values, handle conflicts explicitly, and do not substitute the
   sequenced-organism `tax_id` for host taxonomy.
3. Develop a BioSample-specific model-text representation while retaining every
   raw metadata field: avoid repeating canonical values through aliases, identify
   administrative registration profiles, and keep informative unexpected fields
   such as `source_type` and `ifsac_category`. Validate each change separately;
   the source-fields-only diagnostic was not a successful blanket fix.
4. Validate NLI with scientific/common-name pairs and realistic structured
   records before treating it as a reliable source classifier or verifier for
   this input. Any naming or hypothesis changes require an independent test set
   and a fresh full benchmark, rather than tuning to these selected failures.
5. Measure remaining information gaps, such as study/experiment context and
   token truncation, without enriching BioSample benchmarks with reference labels
   or secretly borrowing TSV host assignments.

The supported conclusion is that **the current benchmark mixes representation
mismatch, reference-data disagreement, loss of deterministic host information,
and real model limitations**. The controlled evidence is strongest for scientific
name/format sensitivity in NLI and for the explicitly conflicting source fields.
Administrative noise is partly causal in tested examples; the full contribution
of context length and missing study metadata remains to be measured.
