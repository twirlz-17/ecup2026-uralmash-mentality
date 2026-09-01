# SUBMISSIONS — every slot spent

LB slots are a shared, scarce resource. **Before** submitting: add a row
with your expected score and what question the submission answers (a
submission that answers no question is a wasted slot). **After**: fill in
the actual. Claim the slot in chat before pushing.

| date | who | tag | config | question it answers | expected | actual |
|---|---|---|---|---|---|---|
| — | max | lexical | lexical probe | baseline floor | — | 0.2586 |
| — | max | v6 | GBDT (pooled training) | GBDT level | — | 0.35502 |
| — | max | CE-solo | crippled CE alone | CE viability | — | 0.4150 |
| — | max | v9 | v8 GBDT + crippled CE, rank blend w=0.5 | blend value | — | 0.44108 |
| 2026-08-19 | max | v10 | GBDT-only, human-only A+B rows (t99 export) | critical experiment: E_real school vs E_mix school on pooling | 0.368 (E_real) / 0.322 (E_mix) | 0.33346 — **E_mix wins, pooling stays**; transfer ~0.65 |
| 2026-08-20 | max | **v11** | v8 GBDT + t98b e5-base ep1 CE, rank blend w=0.1 | does the completed CE curriculum + re-fitted blend weight beat the crippled champion | **0.478 (band 0.46–0.55)** | **0.47542 — NEW CHAMPION**, +0.03434 over v9; the 0.355 transfer predicted +0.0371 and delivered +0.0343 (miss 0.0026) |

| 2026-08-20 | max | **v13** | v8 GBDT + bge-reranker-v2-m3 CE @ max_len **192**, w=0.1 | does oleg's o-bge backbone gain (+0.0220) carry to our stack | **0.487-0.492** (E_real line 0.4872, val-gap 0.4919) | **0.4688363317 — REGRESSION, -0.0066 vs v11.** Both instruments missed by ~0.02. Archive verified faithful (tools/tok_check.py). Confounded: changed backbone AND window; 256 is the only window that has ever worked on our board. See LEDGER v13-lb, linedomain |

| 2026-08-30 | max | **v45** | v39 with ONE entry changed: CE-1 `t167llm2x-ep1` -> `t176full-ep1` (poolfull full-pool draw, 5.08x Stage-A volume at 23.4% positive mix) | does the poolfull MIX change carry to the board, on an axis where the SOLO local read said 'flat'? | **not expected to move** -- solo E_real +0.00030, 8/20, CI through zero; I ranked it LAST of four and the pre-registered `lastslot` bar would have BLOCKED it | **0.5331887134560345 — NEW CHAMPION, +0.0101432254 over v39.** Biggest step since v24. The SOLO read missed by 34x; the CASCADE read (+0.00272, sha `51c14596c45fccc1`) had the sign right and is now 2/2 across v41 and v45. Lesson in LEDGER `v45lb`: never price a container change solo |

## Queued

### 2026-08-30 — four one-variable candidates built against v39 (0.5230454881)

All four are `submission_v39.zip` with EXACTLY ONE entry changed, proven
entry-by-entry by CRC against the base (`tools/build_swap.py` /
`tools/build_ce3.py`), not asserted. The builder was validated by rebuilding
v41 and reproducing its shipped sha256 `7b9acffa2afef604` bit-for-bit.

WHY FOUR DIFFERENT AXES RATHER THAN THE ONE WITH THE BEST LOCAL DELTA.
`transfergap` now has four receipts and is SIGN-wrong 2/4 and MAGNITUDE-wrong
4/4, so a local point estimate barely ranks candidates. Slots expire worthless
and the private score comes from SELECTED submissions, so v39 remains the floor
whatever these score. Under those conditions the quantity to maximise is AXIS
DIVERSITY, not local delta.

NONE OF THESE CLEARS THE PRE-REGISTERED `lastslot` BAR of +0.010 E_real. That
bar was derived for CE-1 swaps from two board receipts; it is recorded here as
failed rather than quietly re-scoped. Agents never submit -- the slots are the
operator's.

| tag | sha256 | the ONE change | local E_real vs v39 cascade | container risk |
|---|---|---|---|---|
| **v43** | `b070848854439da2` | `models/ce-2/model.safetensors` -> Alexander's Stage-B bge | +0.00161, 11/20, 90% CI [-0.00247,+0.00604], P(>0)=0.752 | **none** -- weights blob only, no code change. `config.json` byte-identical, tokenizer vocab (all 250,002 entries), normalizer, pre_tokenizer and post_processor all EQUAL to the shipped bge; base tokenizer kept |
| **v44** | `6a3e8f48d830b5fb` | `run.py`: `W_CE` default `0.7` -> `0.55` | +0.00247, 15/20, 90% CI [+0.00003,+0.00469], P(>0)=0.950 | **none** -- one numeric literal, verified present in the built archive. Zero inference cost at any value |
| **v42** | `7cb7a6810762749f` | adds `models/ce-3/` + a cascaded CE-3 stage in `run.py` | **+0.00565**, 15/20, 90% CI [+0.00169,+0.00907], P(>0)=0.988 | **UNVALIDATED -- DO NOT SUBMIT AS-IS.** See below |
| **v45** | `51c14596c45fccc1` | `models/ce-e5-base/model.safetensors` -> `t176full-ep1` | +0.00030, 8/20, CI through zero | **none** -- weights blob only. `config.json` AND `tokenizer.json` both byte-identical to the shipped CE-1, so the v40 rope trap cannot apply |

**v42 IS THE ONLY ONE THAT EDITS A CODE PATH, AND THAT PATH HAS NEVER RUN.**
The single grader-image test died inside CE-1 -- unchanged v39 code, on a local
RTX 3050 -- at 334s, before CE-3 was ever reached, and Docker then wedged and
could not be restarted from a non-interactive session. v37 scored 0.3611536
from a container shipped unrun. What HAS been checked: the patched `run.py`
compiles, and `tools/check_ce3_scope.py` proves every name the new block loads
(`ce_scores`, `_band_rank`, `_alarm`, `_ce1_secs`, `deadline_ts`, `HERE`, ...)
is bound in scope. That removes the NameError class, not the runtime class.
Gate before submitting: `bash tools/grader_run.sh submission/dist/submission_v42.zip <workdir>`
must complete with a `CE-3 ok` line inside the stage budget.

**AND THE FAILURE MODE IS WORSE THAN "IT JUST SKIPS", WHICH I INITIALLY GOT
WRONG (see `watchdogkills`).** `FORCE_CE` defaults to `1` on any scored stage,
which sets `deadline_ts = None`, which makes the "skip if short on budget"
branches in CE-2 *and* CE-3 INERT -- they never fire. The only real limit is the
SIGALRM cap, and a timeout raised anywhere in the CE section propagates to the
outer handler and discards EVERY cross-encoder score, shipping the GBDT alone.
v42's own stage has its own try/except and would survive its own overrun, but
any change that adds inference work raises the chance of the global failure.


| tag | config | question it answers | expected | owner |
|---|---|---|---|---|
| v11 | v8 GBDT + **t98b e5-base ep1** CE, rank blend **w=0.1** | does the fully-trained CE beat the crippled one, and does the re-fitted blend weight hold up | **BUILT + SMOKE-TESTED** (`submission/dist/submission_v11.zip`, 654.7 MB, blend confirmed running at w=0.1). On the leak-free E_real: **0.61529** vs v9's config **0.51067** = +0.1046, from three stacked gains — fp16 fix +0.044, human Stage B +0.125, weight 0.5→0.1 +0.033. Inference cost identical to v9, so no timing risk. **Projected LB 0.478 (band 0.46-0.55)** via the only CE-touching within-family anchor (CE-solo->v9 transfers at 0.355, i.e. the ruler over-reports 2.8x). No ruler fits LB level across families (R2 <= 0.45, E_mix slope negative, orderings at chance) — so this submission also BUYS the first CE-quality->LB anchor. | max |
| v12 | v8 GBDT + **t104 ep3** e5-base CE, w=0.1 | **HOLD — answers a question we can already answer for free.** The calibrated line called v11 to ±0.0003, so it will call v12 to 0.4793 and a 4th anchor 0.004 from the 3rd adds nothing to an R2=0.9995 fit. Built and smoke-tested; keep as INSURANCE if t114bge misses its gate | **BUILT, NOT QUEUED** `submission/dist/submission_v12.zip` (654.7 MB). leak-free E_real 0.62613 vs v11's 0.61529, +0.01115 sd 0.00217, positive 5/5. Projects **0.4793 = +0.0039**, no timeout risk | max |
| v13 | v8 GBDT + **t114bge** CE (bge-reranker-v2-m3, max_len 192, our Stage A), w tuned | **THE question worth a slot**: does o-bge's +0.0220 backbone gain COMPOUND with our Stage-A curriculum and GBDT blend, or was it specific to oleg's stack? Cross-family, so no ruler we own can answer it (COOKBOOK: cross-family is LB-only) | pending the t114bge gate (>+0.006 over t104 ep3 on leak-free E_real, 5/5) and the throughput benchmark | max |
| v13 | v8 GBDT + t102 e5-large CE, w=0.1 | is 4x the inference cost worth it | **projects 0.4819 = +0.0064** on the calibrated line (R2 0.9995). Downside if the CE misses its deadline is GBDT-only ~0.365, a −0.11 loss, so this needs P(timeout) under ~5%. **UNBLOCKED (RULES.md:90): private = 13 min for ~275k pairs on an H100.** Measured on the box, e5-large needs ~10.5 min of that against e5-base's ~5.4 — it fits only if model load + tokenisation + the GBDT feature path stay under ~2.5 min, and that path is exactly what caused 7 historical timeouts. Also needs a 3h retrain first: the t102 checkpoint did not survive the box refresh. Do not spend a slot on this before a clean single-job throughput benchmark with an H100-vs-RTX-6000 margin. | max |
| v8 | clean GBDT-only (135.5 MB, smoke-tested) | 2nd E_mix→LB delta anchor; decomposes table/depth vs rows axes | ~0.365 (E_mix ladder × 0.65) | max |
| **v14** | v8 GBDT + **t116bge256 ep1** CE (bge-reranker-v2-m3, max_len **256**), w=0.1 | **Does o-bge's backbone gain carry at the ONE window that has ever worked on our board?** v13 moved backbone AND window together and regressed, so it could not say which lost. v14 holds the window at 256 and changes only the backbone against the champion - `run.py` is **byte-identical to v11's** (sha adee33f28f80), so the CE weights are the single variable. It is simultaneously the registered test of the additive-gap model (LEDGER `gapmodel`). | **0.5055** if the gap model holds (val 0.7888 - gap 0.2833). **0.4730** if the gap instead stays at v13's measured 0.3158 - which is BELOW v11's 0.47542. The two hypotheses land 0.033 apart on opposite sides of the champion, so this slot cannot come back uninformative. **PRE-REGISTERED FALSIFICATION:** val >= 0.785 shipping under 0.4900 refutes the additive-gap model alongside the E_real line, and we go LB-only with no exceptions. Local reads, both of which are SILENT on this axis by construction: leak-free E_real 0.64913 at w=0.1 (a tie with v13's checkpoint, -0.0002); the window is worth +0.0046 on shared-split val and +0.00064 on E_real, and +0.00064 is the ruler's measured fp16-vs-fp32 noise floor (LEDGER `reorderprobe`). **BUILT + SMOKE-TESTED** (`submission/dist/submission_v14.zip`, 1186.9 MB, 39 files; both stages pass, blend confirmed running at w=0.1). Checkpoint preserved off-box as kaggle `gordeevmax/ecup26-t116bge256-ep1`. No timing risk: same backbone and window as v13, which scored inside budget. | max |
| **v15** | v14 **plus a second CE**: bge@256 (CE-1) + e5-base from v12 (CE-2), rank-blended at w_ce=0.7, then the GBDT at w=0.1 | **Does cross-family CE ensembling transfer to the board, and do BOARD-PREVALENCE ruler deltas transfer at all?** Two questions in one slot, both first-time. t106 measured this idea at native ~25% prevalence, got +0.0025..+0.0065 and dismissed it; boardprev showed the board runs near 2.5% where effects run ~2x. This is also the FIRST test of whether a delta measured on the new board-prevalence ruler transfers -- that ruler orders models well but its LEVEL does not match the board (it reads 0.382 for a checkpoint that scored 0.4886), so only its ORDERING has ever been used. | **> 0.48855 (v14).** Local: +0.00320 paired-holdout at the CORRECTED board prevalence ~0.045 (the 0.025 first quoted came from leaky rows; see boardprev's self-correction), +0.00358 at 0.025, w chosen on one half and scored on the other, 15/16 splits positive, sd 0.00304 (LEDGER ensboard). NO point forecast is offered -- the transfer rate of board-prevalence deltas is exactly what this buys. **FALSIFICATION, AMENDED 2026-08-22 (LEDGER `v15gate`) — READ THE EXACT-MATCH TEST FIRST:** the original rule was *"if v15 lands at or below v14, cross-family ensembling is dead AND board-prevalence deltas do not transfer"*, and it had a hole. CE-1-alone is v14 **bit-for-bit**, so if the runtime gate DECLINES CE-2 the score is *exactly* v14 — and the rule would have killed two live axes on an infrastructure event that says nothing about either. So: **score exactly `0.4885520439` → the gate declined, the experiment never ran, and this rule does NOT fire.** Any other value → CE-2 ran and the delta is the reading: **above** v14 → the axis transfers *and* board-prevalence deltas transfer, which licenses the e5-large retrain (+0.00588 vs this partner's +0.00320, LEDGER `ce2large`); **below** v14 → both axes die and go LB-only, as originally registered. `tools/v15_gate_sim.py` simulates the gate from the container's own constants over bench320's measured timings and says it **passes in all four scenarios** (public/private × grader-matches-our-card/30%-slower), so a decline is unlikely — but private-at-30%-slower clears by only **5s**, and if it does decline that is evidence against the cost model, not against the ensemble. **SAFETY, PROVEN NOT ASSUMED:** the degradation ladder is both CEs -> ensemble, CE-2 slow/missing -> CE-1 alone, CE-1 fails -> GBDT. tools/test_v15_fallback.py ran v14 and v15-with-SKIP_CE2 on identical synthetic input: **outputs are IDENTICAL bit-for-bit**, and the ensemble path measurably differs (spearman 0.972). So the worst case of this slot is v14's score. CE-2 runs only if CE-1's MEASURED elapsed time leaves room (x0.35 cost ratio, x1.6 safety), which adapts to the grader's hardware rather than trusting a constant. **COST:** 8.90 of 13.0 min with the 30% H100 margin, from bench_infer's measured 1.59 CE-min for e5-base on 275k pairs -- comfortable. e5-large scores higher (+0.00750) but needs a ~3h retrain and costs 11.90/13 with no margin, so it is NOT this submission. **BUILT + SMOKE-TESTED + FALLBACK-TESTED + RE-VALIDATED after a pair-text cache was added** (ce_scores rebuilds item texts internally, so CE-2 was repeating a parquet read plus two regex passes over the whole item pool -- 33.4s measured on 711k items, 13% of the headroom; the archive now memoises them and the smoke log shows ONE item-load instead of two) (`submission/dist/submission_v15.zip`, 1706.1 MB, 45 files; ZERO GPU -- CE-2 was lifted out of submission_v12.zip). | max |

## oleg's account — a SEPARATE quota, not a shared slot

These ran on a different competition account, so they never consumed a slot
from the table above; the board and the metric are the same, so the levels
are directly comparable. Full detail: `oleg/docs/RESULTS.md`,
`oleg/EXPERIMENTS.md`.

| tag | config | question it answers | actual |
|---|---|---|---|
| v1 / v2 | CatBoost on hand features + LLM concept map (+ char n-grams) | how far does a no-GPU classical pipeline get | 0.2659 / 0.2859 |
| v3 | CE `e5-small` retrained on human labels | is the CE lane alive | 0.3932 |
| v4 → v6 | CE score as a feature in the per-category GBDT; then two CEs | does stacking on a CE pay | 0.3999 → 0.4118 |
| v6w | **one** warm-started `e5-base` CE, raw score, refit on 365k | one CE or two | **0.4282** |
| v7 | two CEs + per-category rank-normalised predict | is rank normalisation safe as an output transform | 0.3910 — **no** (LEDGER `o-rank`) |
| v8 | v6w recipe, explicit refit, no ranks | equivalence check on v6w | 0.4272 (confirms, noise ~0.001) |
| v9 / v9.1 | CE **trained** at max_len 384; then same weights, faster inference | is a longer window a win, and is any loss just timing | 0.4177 / 0.4177 — **no**, and not timing (LEDGER `o-len384`) |
| v10 | v8 + tuned GBDT ensemble (8-seed bag + LGBM) | does GBDT tuning still pay over a CE | 0.4322 |
| v11 | **pure `bge-reranker-v2-m3` CE, no stack** | does a relevance-pretrained backbone transfer better | **0.4542** — best gap on that track, 0.3253 (LEDGER `o-bge`) |
| v-ibm | hand features + IBM Model 1, no CE at all | do the classical layers stand alone | 0.2697 — below v2 (LEDGER `o-classic`) |
| v12 | GBDT ensemble **over** the bge CE | does the tree layer add on top of a strong reranker | **0.457877 — that track's champion** |
| v13s | pure CE on a **selectively** warm-started bge (v11 archive, weights swapped) | does zeroing the LLM warmup in the four weak categories help the anchor | predicted 0.4555 → **0.446082**, −0.00813 vs v11 |
| v13 | GBDT stack over the same selective anchor (v12 recipe bit-for-bit) | same question, one layer up | predicted 0.4577 → **0.452926**, −0.00495 vs v12 |
| **v14** | GBDT stack n25 over **bge-reranker-v2-m3 @ max_len 256** (v12 recipe bit-for-bit, single variable = the CE checkpoint) | does the window law (256 is the only window that ever worked on the board) pay on THIS track, whose champion still runs at 192 | **0.4665801183 — NEW CHAMPION of this track, +0.00870 over v12** (submitted 2026-08-21 by Олег). No forecast was written, per `linedomain` — the pre-registered accept rule was only "beats v12", and it does. Honest val was **0.7892**; transfer +0.0171 val → +0.0087 LB (~0.51, the stack-family band, not the backbone's near-1). Gap grew 0.31422 → 0.32262 while max's v14 gap *shrank* to 0.30025 — same change, opposite gap moves. Together with max's v14 (0.48855, +0.0131) this is the positive-direction confirmation of the window law on **both** pipelines. LEDGER `o-v14lb`, `o-256cmp` |
| **v16c** | схема max'а на нашем аккаунте: CE-1 = t158mlmA2b-ep1 (mmBERT-base @1024) НЕСУЩИМ, полное покрытие + токен-sortkey + charcap 900, head-first обход, ранг-бленд 0.9*CE + 0.1*GBDT | переносится ли класс «carrier» кросс-трек | **0.4901711006800616 (2026-08-29, Олег) — НОВЫЙ ЧЕМПИОН НАШЕГО ТРЕКА, +0.0229327 к v16g. Крупнейший одиночный ход трека на доске; разрыв с командой был 0.0558, закрыт на 41%, остаток 0.0329.** Пререг был мис-калиброван (порог «<0.50 = не переносится» привязан к абсолютному уровню, а не к дельте против нашей базы) — по механизму перенос ЕСТЬ и он большой, но ЧАСТИЧНЫЙ, поэтому REVISIT, не REJECT. Остаток объясняется единственной структурной разницей: у нас ОДИН CE, у чемпиона КАСКАД декоррелированных (runens2 мерил этот переход в +0.0282..+0.0304 на его линейке — тот же порядок, что наш остаток 0.0329). LEDGER `v16clb` |
| **v16g** | same n25 stack, retrained over the **graphfix CE** (Stage A on min-flip-corrected labels, enriched draw; v12/v14 recipe bit-for-bit, single lever = the CE checkpoint) | does correcting provably-contradicted Stage-A labels IN PLACE pay on the board (the only living form of the label-cleaning lane, o-v13lb) | **0.4672384458 — NEW CHAMPION of this track, +0.00066 over v14** (submitted 2026-08-22 by Олег). Local container read was **−0.0006** vs v14 (0.7886 vs 0.7892) — board and local disagree in sign and BOTH are within noise (unpaired ruler sd 0.00337; single LB read), so the honest claim is «v16g ≈ v14, champion badge moves on a coin-edge». Gap 0.3214 vs v14's 0.3226. The MECHANISM verdict (graphfix vs control, bar +0.003) still awaits the control arm's finish — this slot does not decide it. Chain: invariant PASS 0.999994 → stack retrained on the soup-free npz → bit-verified np-export → smoke 600/600. **Roles split (anti-ratchet): v16g is the SHIPPED artifact; the PRICING baseline for future arms stays «v14 == v16g, either» — promoting the baseline on a positive coin-flip and never on a negative one walks it upward on pure noise (the time-extended winner's curse, per ozon-5b).** LEDGER `v16g-lb` |
| ~~v14q~~ | same stack over **Qwen3-Reranker-0.6B @256** | is a newer reranker backbone worth a slot | **BLOCKED — DO NOT SUBMIT.** The stack layer built and exported fine (honest val 0.7619), but the CE artifact is desynchronised from the scores that trained it: `best/` weights reproduce their own `ce_bge_scores.npz` at corr **0.898** (mean\|Δ\| 0.099) where bge@256 reproduces its own at **0.99999**. `last/` fails identically (0.897), the file is complete (311/311 tensors, score head present), and CPU and MPS agree to max\|Δ\|=0 — so the instrument is sound and the molab run simply did not push the model it scored with. Scored by its OWN shipped weights that checkpoint reads ~0.655, not 0.7430. Shipping it = a stack trained on scores its CE cannot produce, i.e. exactly the silent desync invariant #1 exists to catch. Fix: re-score all 365,654 pairs with the pushed weights (GPU minutes, or ~12 h local MPS) and retrain the stack on those. Stack artifacts kept at `oleg/submission_v14q_stack/`. LEDGER `o-qwendesync` |
| v15pc_spec | wbp-ярус: bge-стек + колонка euro-СПЕЦИАЛИСТОВ (t165pc, 20 дельт поверх общего ствола) | локальный REJECT (percatstack) — слот вопрос не покупал | **0.4672172063 == v16g (2026-08-27, Олег). Euro-ярус на грейдере НЕ ЗАПУСКАЛСЯ**: образ несёт transformers **5.14.1** (боксы 4.57.3), vendored EuroBERT-код падает `KeyError 'default'` (ROPE) при инстанцировании, общий except в run.py:597 молча оставляет wb → предсказания == v16g бит-почти-в-бит (разница LB 2.1e-5 = fp16-джиттер). Слот прочитал ноль про euro-колонку. Пререг-строки до сабмита не было — протокольный пропуск. LEDGER `v15lb`, note `gradenv` |
| v15ctl | wbp-ярус: bge-стек + колонка ПУЛИРОВАННОГО euro-ствола t152euroA (локально +0.0153 macro, 0.8039) | euro-колонка на доске (crosstrack-семейство) | **0.4672214376 == v16g (2026-08-27, Олег), тот же механизм**: wbp ни разу не запустился, вопрос НЕ куплен → REVISIT, не REJECT. Ирония, зафиксированная докер-репро: KeyError СПАС слот — с одним rope-фиксом euro-форвард под 5.x даёт corr **0.2597** с обучающими скорами (маски), колонка была бы вредной. Euro-колонка в контейнерах заблокирована до мостика масок + паритет-гейта. LEDGER `v15lb`, `gradenv` |
| **v15t158** | bge-стек + t158mlmA2b-колонка (заявляемая 0.8069, stackcol ACCEPT) | лучшая одиночная колонка по декорреляционной лестнице | **СОБРАН, но сабмитить ТОЛЬКО `submission_v15t158fix.zip`**: оригинальный зип несёт 5.x-конфиг ствола (sliding rope theta 160000 вместо обучальных 10000 — артефакт fp16-экспорта под kaggle-venv 5.14.1) → в образе грейдера колонка воспроизводит обучающие скоры на corr **0.97682** (тихий десинк класса v14q). Фикс = одно поле конфига, проверен В САМОМ ОБРАЗЕ докером: corr **0.99998**, бит-в-бит с локальным бейзлайном. Build-копии trunk-конфига тоже пропатчены. Остаточный риск: пропускная способность spec-яруса @1024 на public (деградация мягкая через рейт-гейты; дизайн-дыра первого арма записана в note). LEDGER `gradenv` |
| ~~v15t158 (оригинал, 2 слота)~~ | ТОТ ЖЕ оригинальный зип, сабмитнут ДВАЖДЫ 2026-08-28 (до/мимо предупреждения строкой выше) | — | **0.44801749978 и 0.45426147123 — ОБА НИЖЕ v16g (−0.0192 / −0.0130). Прогноз десинка подтверждён доской**: в отличие от euro (падал → безопасный откат на wb == v16g), mmBERT-ствол в образе ГРУЗИТСЯ — фолбэка нет, wbp-стек получает колонку corr-0.977 и ПОРТИТ каждую покрытую категорию. Разброс 0.0063 между двумя прогонами ОДНОГО файла = недетерминизм таймингова каскада (сколько категорий успел покрыть spec-ярус, столько испорчено; wb-пол при этом уцелел — обвала к 0.36 нет, т.е. бюджет wb не украден). Слот-дисциплина: два слота на неисправленный артефакт при готовом фиксе в корне репо — сабмиты шли без чтения SUBMISSIONS.md. LEDGER `v15t158lb` |

**v13/v13s verdict: the selective warmup is refuted on the board, 2/2** — and it is the LB
confirmation of `t105`. Both archives changed exactly one thing, the CE weights. Subtracting LLM
label mass from Stage A costs board points even where the local ruler says otherwise; the removed
rows are the *ambiguous* ones, which is t105's mechanism and these are the ambiguous categories by
construction. The unweighted bge anchor (v11/v12) stands. Self-inflicted lesson: the local read was
+0.0014, **below that track's own 0.003 epsilon**, and two slots were spent on it anyway.
LEDGER `o-v13lb`.

## Rules

- Never two submissions testing the same question.
- Expected score written down BEFORE the result comes back — the
  expected-vs-actual gap is our ruler calibration data; don't destroy it.
- Final-week slots are reserved by team decision at the weekly call.

## 2026-08-22 slot guidance (max) — v16 HELD by the human, v17 HOLD (superseded below)

**v16** (`dist/submission_v16.zip`, sha256 cb6a8d5cfdb44789, 1706.1 MB, 45 entries)
re-verified 2026-08-22: all archive checks pass, CE-1 weights byte-identical to
v14's proven floor. Local paired delta over v15 = **+0.0014** (CE-2 swapped
t104-ep3 -> t104-ep4). Caveat on the expectation: the 1.010 transfer figure was
measured on the cross-FAMILY ensemble axis; this is a same-family partner swap,
a different axis whose transfer is unmeasured. +0.0014 is the local reading, not
a board prediction.

**v17** (`dist/submission_v17.zip`, sha256 312724fd501c435a) is built and passes
six container-vs-measurement checks, and should NOT get a slot. It buys **zero
accuracy** over v16 — the two-CE cascade measures +0.00013 on the leak-free
slice, inside the 0.0002 tie-ordering noise floor (`tiecost`). The cascade exists
to make room for a THIRD cross-encoder; at two CEs, v16 already fits at 8.26 of
13.0 CE-minutes. Spending a slot on v17 would put never-container-executed code
on the critical path for nothing and return a number we can already predict.

**v18** is where the cascade earns its place (+0.00409 over v16, 16/16). It will
change two things at once — cascade and CE-3 — which is accepted rather than
ideal: the cascade's local accuracy effect is +0.0001 so it cannot explain a
large miss, and a cascade BUG would most likely trip the degradation ladder
(falling back to v17 or v14) rather than degrade silently. Anyone who wants that
risk removed can spend a slot on v17 first; it is defensible and it is not the
recommendation.

**Two gates before v18 goes anywhere**, both already written and neither
optional: `tools/verify_ce3.py` (the delivered checkpoint must reproduce
`handoff/oleg_bge256_scores.parquet` at pearson >= 0.999 — his repo carries six
revisions of that run scoring 0.7747-0.7860) and `box/queue15.sh` (the cascade
timing; 9.97 CE-minutes is arithmetic on full-pass rates, and a 40% subset pass
does not amortise per-batch overhead the same way).


## 2026-08-22 slot guidance, REVISED (max) — hold v16 too

**The human's call, and it is the right one:** v16 is not worth a slot.
Its expected delta is **+0.00138**, less than half the smallest board effect
this project has ever cleanly resolved (v15's +0.00323). A slot spent there buys
position, not information — `v16predict` said exactly that before the decision,
and the decision agrees with it.

Standing state:

| build | status | why |
|---|---|---|
| **v15** = 0.49178410751467555 | **CHAMPION, shipped** | |
| v16 | built, **HELD** | +0.00138 is under our demonstrated board resolution |
| v17 | built, verified, **HELD** | +0.00013 accuracy over v16; it exists to buy the CE-minutes a third cross-encoder needs, and is only worth a slot as part of v18 |
| v18 | built, verified at every rung, **waiting on one file** | +0.00409 over v16 at 16/16 — the largest unshipped delta we own |

**What a slot should be spent on next** — not a rung we already believe. The two
candidates that would actually *resolve* something:

1. **v18**, once Олег's `ecup26-oleg-bge256` checkpoint arrives as a Kaggle
   dataset. Run `tools/verify_ce3.py` against it FIRST — his HF repo carries six
   revisions scoring 0.7747–0.7860, and shipping ep3 while quoting a number
   measured on ep2 would be silent and would cost most of the gain. Pass is
   pearson ≥ 0.999. Still also gated on queue15's cascade timing.
2. **A window arm at 320**, if t117 finishes to ep1 on the next box. `windowband`
   makes this the best-motivated unshipped change we have: the local read is
   +0.00857 at 15/16 *and* the effect sits exactly where the mechanism requires,
   with a byte-identical-input control stratum pricing the noise. But
   `windowlbonly` stands — a local read cannot accept a window change, which is
   precisely what makes this a question worth a slot.

**Slots are the human's. Nothing here submits anything.**


## 2026-08-23 slot guidance (max) — v19 is built, verified, and is the candidate

**`dist/submission_v19.zip`, 2754.7 MB, sha256 `13b69820cf8cf964`.**

| build | local macro | vs shipped | CE-min of 13.0 | status |
|---|---|---|---|---|
| v15 (champion, board 0.49178) | 0.49754 | — | 5.68 | shipped |
| v16 | 0.49892 | +0.00138 | 5.68 | **held** — under our board resolution |
| v17 | ~0.49767 | +0.00013 | 8.26 | held — exists to buy CE-minutes, not accuracy |
| **v19** | **0.50773** | **+0.00986, 16/16** | **6.97** | **built + verified** |

v19 = bge@320 promoted to CE-1, with e5-base (t104-ep4) and **Олег's bge@256**
both cascaded at 40% coverage. Every gate that could be cleared locally is
cleared: `verify_ce3` matched his checkpoint to the scores it was priced on at
pearson 0.999996, and `verify_v19` executes the container's own fold block at
all three degradation rungs and finds each equals a weighting somebody measured.

**Two things are NOT cleared, and neither can be from here:**

1. **`queue15`'s cascade timing has never run.** Every CE-minute number above is
   arithmetic on full-pass rates. The container has a runtime gate that drops a
   rung rather than time out, so the failure mode is losing +0.00283, not a
   zero — but it is unmeasured.
2. **`windowlbonly`**: no local read may ACCEPT a window change, and CE-1 at 320
   is one. The precedent is in our favour — when the board last judged a window
   it paid **+0.01972** while the local ruler read −0.0053..+0.0020, i.e.
   under-read it tenfold — but a favourable precedent is not a measurement.
   This is exactly the kind of question a slot is *for*.

**Honest forecast.** The local ruler is not the board scale (`levelcal` REJECT):
v15 reads 0.49754 here and 0.49178 there. Deltas transfer at ~0.65 for
cross-family changes, so +0.00986 local projects to roughly **0.498–0.502**,
with genuine upside if the window under-reads the way it did last time.

**Also in flight:** `t126mm1024` — mmBERT (ModernBERT multilingual) at a
1024-token window, where **0.6% of pairs truncate against our shipped 61.9%**,
benched at 8.44 of 13.0 CE-minutes as a sole cross-encoder. If it lands it is a
bigger change than v19 and the window ladder closes.

---

## v20 — one weight swap on v19, at identical cost

`submission_v20.zip` · **2.75 GB** · sha256 `8f30424e1d4f73f870e22a8019d5f14534b60ca8c1726219ed32cb15da59fd25`

**The whole diff is one slot.** v19 ships `t117@320` (CE-1) with `t104-ep4` and
`oleg@256` cascaded at 40%. v20 replaces **only** the `oleg@256` slot with
`t120pw0.134-ep1`. CE-1, the other rung, the window, the degradation ladder, the
CE-3 budget ratio and every tokenizer are byte-identical, because both
checkpoints are bge-reranker-v2-m3 at window 256 with the same architecture, the
same tokenizer, and the same file size — **1,135,559,698 bytes exactly**.

**Cost is unchanged: 6.97 of 13.0 CE-minutes**, same as v19. This is the reason
to prefer it over bigger containers (below).

> **CORRECTED.** The first version of this entry quoted **+0.00660**, measured
> on 16 disjoint 1/16 folds (~2284 rows, ~29 positives per category). That
> design inflates deltas by ~35% and inverted a sign elsewhere (`foldsize`).
> Re-measured on this project's own convention — `ensboard`'s resampled
> half-splits of 18,271 rows — the two columns agree to 0.00001.

| container | CE-min | full-slice Δ vs v19 | half-mean Δ | splits+ |
|---|---|---|---|---|
| v19 as built (t104 + oleg) | 6.97 | — | — | — |
| **v20 — t104 + t120a** | **6.97** | **+0.00489** | +0.00490 | **16/16** |
| **v20b — t120a + t120b** | 7.97 | **+0.00931** | +0.00873 | **16/16** |
| v20 without the GBDT | 6.97 | +0.00529 | — | — |
| two rungs only (gate declines) | — | −0.00322 | — | — |

Leak-free E_real at prevalence 0.045. Local macro **0.51264** against the
shipped blend's 0.49754. **v20's +0.00489 does NOT clear the standing +0.006
bar** — it is a 16/16 directional result of smaller size than first reported.
**v20b does clear it (+0.00931, 16/16)** but costs 1.00 more CE-minute and, per
the runtime note below, would need `need2` repatched from 0.35 to 0.87 because
its CE-2 becomes a bge rather than an e5-base.

**Why not the +0.00984 arm.** It costs 1.64 more CE-minutes, and `queue15` — the
cascade **timing** measurement — has still never run, so every CE-minute figure
in this project is arithmetic rather than an observation. Spending unvalidated
minutes to buy a further +0.0032 is the wrong trade while that gate is open.
v20 moves no timing at all. If timing is ever measured end to end,
`t120a + t120b + t104` is the arm to promote and it needs no new training.

**Where t120pw0.134-ep1 came from,** since its name says "loss sweep" and that
would be the wrong way to read it. It was the treated arm of a pos_weight A/B;
`t120ep2` REJECTed pos_weight and was right to. `epochpair` then showed the
sweep's **plain-BCE control arm gains just as much** (+0.00967 vs +0.01033), so
this is not a pos_weight effect — both arms are simply better models, because
both descend from `t119vol2x`'s **2x-volume Stage A**. It is shipped for the
lineage, not the loss.

**What was checked before this was called shippable**

1. **Selection bias.** The container was first found as the argmax of 512
   configurations scored on the rows it was chosen on. Redone with selection on
   one half and scoring on the disjoint other: bias term **−0.00110**, i.e. the
   search did not inflate it.
2. **Leak.** The leak-free mask lives in `out-t102` and every reader assumes it
   matches `train_ce.py`'s split. Rebuilt from the primary parquet with
   train_ce.py's own recipe — **100.0000% agreement, 0 of 36,542 rows
   contaminated**.
3. **Ruler validity.** `stageavol2x` pre-registered, before the run, that E_real
   is one-sided on this lineage: it flatters human-leaning arms and penalises
   LLM-leaning ones, so a **win is credible** and a loss is uninformative. This
   is a win, and `stageavol2xA` read the same Stage A at +0.0045 on human-val.
4. **The archive computes what was measured.** `verify_v20` — 9 checks, all
   pass, fold arithmetic exact to 1.1e-16 at every rung.
5. **The fp16 cast is free.** `push_ckpt` stores fp16, so the weights in the
   archive are not the fp32 the container was priced on. Scored the SHIPPED
   fp16 weights on the full universe and re-ran the whole comparison on them:
   **+0.00664 at 15/16 against the fp32 pricing's +0.00660**, paired difference
   **+0.000038 +- 0.000250** -- inside E_real's ~0.0006 noise floor. pearson
   0.999994 / spearman 0.999928 against the fp32 dump, on par with the 0.999996
   that cleared oleg's CE-3.
6. **The right checkpoint went in.** Check 9 of `verify_v20` is new and is the one that matters:
   oleg's CE and t120pw0.134-ep1 are the **same size**, so the size assertion
   cannot separate them and shipping the wrong one would pass every v19 check
   silently. v20 hashes the shipped weights against the checkpoint on disk —
   `4695498b8f98d043…` — and they match.

**Known risks, unchanged from v19**

1. `queue15`'s cascade timing has never run; the CE-minute figures are
   arithmetic. v20 does not increase this exposure — it costs exactly what v19
   costs.
2. `windowlbonly`: CE-1 at 320 is still a window change no local read may
   ACCEPT. Inherited from v19, not introduced here.
3. This drops oleg's CE from the container. That is a ranking of checkpoints on
   our ruler, not a judgement of his lane; `crosstrack`'s cross-family argument
   stands and `partnerprice` gives the bar his CE needs to clear as a partner.

### What the recheck found — read this before spending a slot

**1. "6.97 of 13.0 CE-minutes" was the wrong frame, and I propagated it.** The
13 minutes are TOTAL runtime: `timelimit` records that they must also cover
model load, tokenisation, the GBDT feature path and I/O, and `bench320` puts
that non-CE work at ~165 raw seconds. Simulated from the container's own
constants (`total_budget` 780s, `reserve` 0.92 → deadline 718s,
`need3 = ce1*0.87*1.6*0.40`):

| grader | non-CE + CE-1 | CE-3 gate | finishes | scores |
|---|---|---|---|---|
| matches our card | 165 + 282 = 447s | left 232s vs need 157s → **RUNS** | 584s | 0.51264 |
| 1.30× slower | 215 + 367 = 582s | left 86s vs need 204s → **SKIPPED** | 631s | 0.50453 |

**This is not a v20 risk — it is the container class.** v19 shares CE-1 and CE-2
exactly, so in the slow scenario both degrade to the *same* 0.50453. v20 weakly
dominates v19 everywhere: equal if the gate declines, +0.00489 if it fires.

**2. The GBDT is contributing nothing, and it is what costs us the rung.**
Inside the three-rung container the GBDT is worth **+0.00040 full-slice,
−0.00008 by half-split mean, 7/16 splits** — under the noise floor and worse
than a coin flip. That does not contradict `gbdtworth`'s +0.00492: that was
measured against a CE-1-only baseline in the v15-era container, and a weak
partner (solo 0.2934) stops paying once the cross-encoder side reads 0.51264.

Meanwhile it consumes essentially the whole non-CE budget. Swept rather than
assumed: **the third rung survives a slow grader once ≥60% of the 165 non-CE
seconds are GBDT-specific**, and `gbdtworth` puts the GBDT half at ~185s.

Note the direction: **+0.00040 is what *removing* the GBDT is worth** on the
full slice (0.51303 against 0.51264), while the half-split mean says removing it
costs −0.00008 at 7/16. The two columns disagree in sign and both sit under the
~0.0006 noise floor — which is the finding. The GBDT is not a small positive
being sacrificed; it is indistinguishable from zero in either direction. So the
trade is: give up nothing measurable, gain **+0.00851 (16/16)** whenever the
gate would otherwise decline.

**Why I have not shipped that change.** The GBDT runs *first* and calls
`write()` before the cross-encoder starts, so it is also the container's safety
net: an incomplete CE-1 currently falls back to the GBDT result, and without it
that path writes a constant. Dropping it trades rare-catastrophe insurance for a
common-case gain — while making that catastrophe much less likely, since CE-1
alone would finish at 389s of 718s. That is a judgement about risk appetite on
someone else's submission slot, so it is registered (`gbdtdead`, REVISIT) rather
than shipped.

I earlier called **reordering** — running the GBDT after the cross-encoders on
leftover time — the "strictly better" fix. With the sign corrected that is no
longer true on accuracy: there is no +0.00040 to preserve, because removing the
GBDT is the direction that gains it on the full slice. Reordering would buy back
only the **fallback** for an incomplete CE-1, and nothing else, at the cost of a
much bigger diff to `run.py`. It is worth doing only if that fallback is judged
to matter — which is the same judgement v21 already puts in front of a human.

---

## v21 — v20 with the GBDT switched off (RECOMMENDED over v19/v20)

`submission_v21.zip` · **2.75 GB** · sha256 `9314383035a1c8a32a134fc33e0f48c65a8022f3ac446c761c241b53fe85b49a`

The only container measured today whose **worst case equals its best case**.

| container | fast grader | 1.30× slower | worst case |
|---|---|---|---|
| v19 (built) | 3 rungs, 584s → 0.50775 | 2 rungs, 631s → 0.50453 | 0.50453 |
| v20 (built) | 3 rungs, 584s → 0.51264 | 2 rungs, 631s → 0.50453 | 0.50453 |
| **v21** | **3 rungs, 435s → 0.51303** | **3 rungs, 565s → 0.51303** | **0.51303** |
| v20b (t120a+t120b) | 3 rungs, 644s → 0.51705 | 2 rungs, 709s → 0.50453 | 0.50453 |

Every other option ties v19 at 0.50453 when the grader is slow. v21 is
**+0.00850 on the worst case** and within 0.004 of the best case of anything on
the list.

**The change is one flag.** `RUN_GBDT` defaults to `0` and gates the whole GBDT
block; `RUN_GBDT=1` restores v20 exactly, and the GBDT models are still in the
archive. Nothing else differs from v20 — same CE-1, same rungs, same windows,
same weights, same ce-3.

**Why it costs almost nothing.** Inside the three-rung container the GBDT is
worth **+0.00040 full-slice, −0.00008 by half-split mean, 7/16 splits** — under
the noise floor. That does not contradict `gbdtworth`'s +0.00492, measured
against a CE-1-only baseline in the v15-era container: a partner soloing 0.2934
stops paying once the cross-encoder side reads 0.51264.

**Why it buys the rung.** `v20gate` showed CE-3 is skipped at 1.30× because
~215 slow-seconds go to non-CE work before the cross-encoders start.
`gbdtworth` puts the GBDT half at "roughly 4 of the 13.0 private minutes", i.e.
essentially all of it. Swept rather than assumed: the rung survives once **≥60%**
of the 165 raw non-CE seconds are GBDT-specific.

**What is given up, plainly.** The GBDT ran *first* and called `write()` before
the cross-encoder started, so it was also the fallback for an incomplete CE-1 —
that path now writes a constant. The same change makes it far less reachable
(CE-1 alone finishes at 389s of a 718s deadline, not 582s). It is a trade, not a
free win, and it is the one judgement in this container that belongs to a human.

**Verified 6/6** (`verify_v21`). Check 4 is the one that matters: with `gb=None`
the blend takes a *different* branch, so the check executes that branch and
confirms it writes the folded three-rung result — byte-identical to v20's fold,
max|d| `0.00e+00` — rather than a constant.

**Two self-inflicted faults on the way**, recorded because they will recur: the
first zip ran the local disk to exactly 0 bytes and truncated silently, and then
`build_v21 --stage` was pointed at its own output directory, whose `rmtree`
deleted the tree it was about to copy from. Nothing was lost — everything
rebuilds from `submission_v20.zip` — and the script now refuses when source and
destination resolve to the same path, and takes `--from-zip` to skip the 3 GB
tree copy entirely.

**Note:** `submission_v19.zip` was deleted to make room. v20 strictly dominates
it (identical runtime, better accuracy, both verified), and it rebuilds from
`build_v19.py` plus the Kaggle-hosted checkpoints.

---

## v21 — BOARD RESULT: 0.48656258501230026 · **REGRESSION, −0.00522 vs v15**

| | board |
|---|---|
| v15 (champion) | 0.49178410751467555 |
| v14 | 0.4885520439 |
| **v21** | **0.48656258501230026** |

**Local predicted +0.01549 (0.51303 vs 0.49754). The board delivered −0.00522.
A miss of −0.02071, with the sign inverted.**

**The container is unattributable, and that is my error.** v21 moved four things
at once against the champion: CE-1 window 256→320, partner structure
full-coverage→40% cascade, CE-3 checkpoint oleg→t120a, and the GBDT removed. One
board number cannot separate four variables. This project had already recorded
the lesson — v13's entry reads *"Confounded: changed backbone AND window; 256 is
the only window that has ever worked on our board"*, and v14 fixed it by holding
the window and moving one variable. I flagged `windowlbonly` in v21's own entry
and then recommended the container on local numbers anyway.

**What the board's own deltas now say:**

| change | board delta |
|---|---|
| v11→v14 — backbone only, **window held at 256** | **+0.01313** |
| v11→v13 — backbone + window(192) together | −0.00662 |
| v14→v15 — add CE-2 at full coverage | +0.00323 |
| v15→v21 — window(320) + 40% cascade + no GBDT | **−0.00522** |

Every window the board has judged other than 256 has lost. That is now a
three-point pattern, not a slogan.

**Ruler damage.** E_real's *level* calibration held — it reads 0.49763 for v15's
container against the board's 0.49178 — so this is a **delta** failure on
container composition, consistent with `rulerhot` and `emix`. The specific
unlicensed step was treating a container that contains a window change as
locally comparable at all.

**What is not implicated.** The t120 checkpoints were never board-tested in
isolation, and both rulers rank every t120 arm above t116 as CE-1. That is what
v22 tests.

---

## v22 — v15 with ONE variable moved · **READY**

`submission_v22.zip` · **1.71 GB, 45 files** · sha256
`dad84b75263053af0572a0fb77c32afacab6bf0cc5d6630e37a0c2e6d200bc9c`

**The only thing that moves is the CE-1 checkpoint**: v15's `t116-ep1` becomes
`t120pw0.134-ep1`. Window stays 256, CE-2 stays e5-base at **full coverage**,
GBDT stays at w=0.1, runtime stays v15's **8.90 of 13.0** budgeted minutes — a
container the board has already executed successfully. No window question, no
cascade question, no timing question.

This is the v14 move repeated: v11→v14 changed only the CE-1 checkpoint with the
window held at 256 and paid **+0.01313**, our largest board gain.

**Measured in v15's exact container shape** (leak-free E_real at prevalence
0.045, full-slice level, 16 resampled half-splits):

| CE-1 candidate | E_real | vs v15 | splits+ | human-val |
|---|---|---|---|---|
| t116-ep1 *(what v15 ships)* | 0.49763 | — | — | 0.7888 |
| **t120pw0.134-ep1** | **0.50845** | **+0.01081** | **16/16** | 0.7909 |
| t120pw0.134-ep2 | 0.50517 | +0.00754 | 15/16 | 0.7851 |
| t120pw1.0-ep2 | 0.50247 | +0.00484 | 15/16 | 0.7868 |
| t120pw1.0-ep1 | 0.50221 | +0.00458 | 15/16 | 0.7930 |

**The two rulers disagree and it is not hidden.** human-val orders
pw1.0-ep1 > pw0.134-ep1 > pw1.0-ep2 > pw0.134-ep2 — nearly the reverse of
E_real. They **agree** every t120 arm beats t116 (0.7888), so the disagreement
is over which of two near-equivalent checkpoints, not whether to swap. The
variable separating the two arms is `pos_weight`, which `t120ep2` already
measured as a **null**. pw0.134-ep1 is chosen because E_real is one-sided-valid
in the *win* direction on this lineage (`stageavol2x`), it wins there at 16/16
by the largest margin, `epochpair` independently confirmed the 2x-Stage-A
lineage, and the artifact is already verified end-to-end.

### v22 also deletes the silent fallback — the second reason v21 was unreadable

The container shipped a **degradation ladder**: if the runtime gate estimated no
room it dropped CE-2 and scored CE-1 alone. That is insurance against a zero,
and it is a **bad trade for us**, because the degraded container scores ~0.4966
— about 0.003 from a genuine result, with no logs from the grader to tell them
apart. A timeout scores ~0 and is obvious; a declined gate is *invisible*. We
spent a whole analysis (`v21fallback`) on the question and could only reach
"unsupported, not refuted".

`FORCE_CE` already existed in `run.py`; only its default was wrong, and we
cannot set environment variables on the grader. v22 flips it, so `deadline_ts`
is `None` and **both cross-encoders always run**. The failure mode moves from a
silent ~0.4966 to the GBDT-only result at **~0.29** — unmistakable.

**Safe by arithmetic, not hope:** GBDT ~165s + CE-1@256 245s + CE-2 e5-base at
full coverage 95s = **505s raw, 657s at the 1.30× slow-grader margin**, against
the SIGALRM watchdog at 0.97×780 = **756s**. And v15 ran this exact container
inside budget on the real board.

**The bare flip was a defect, and the verifier caught it.** `force_ce` also
gates the **check-stage** skip (`elif n <= 10_000 and not force_ce`). A global
flip would start running the CE on the check stage — which carries a **60-second**
budget, not 780, and which is a *rejection* gate: overrun it and the submission
never reaches scoring at all. Trading a silent-fallback ambiguity for a chance of
never being scored is strictly worse. The forcing is therefore scoped with
`and n > 10_000`, so the check stage keeps v15 behaviour exactly. The third
branch `force_ce` gates — the no-CUDA skip — stays disabled on scored stages on
purpose: if CUDA vanished we *want* the loud GBDT-only ~0.29, and the watchdog
raises `TimeoutError` (an `Exception`), which the existing handler catches and
degrades to the GBDT result rather than crashing.

**Verified as a diff, not an argument** (`verify_v22`, 9/9): same file
inventory; **exactly three members differ** — the two CE-1 files and `run.py`;
`run.py` differs from v15's in **exactly one line**, and that line is the
`FORCE_CE` default (checked as a line-pair, not as "it changed"); no v18–21
patch markers; shipped CE-1 weights hash to `4695498b8f98d043` (the verified
checkpoint, not another bge of the same size); CE-1 tokenizer (4 files) and GBDT
models (25 files) byte-identical; the gate is **replayed out of the shipped
source** at all three stage sizes and behaves as designed —

| stage | `force_ce` | `deadline_ts` | behaviour |
|---|---|---|---|
| check n=1,000 | False | 55.2 | gate active — identical to v15 |
| public n=115,000 | True | None | both CEs forced |
| private n=275,000 | True | None | both CEs forced |

— and `deadline_ts` is assigned exactly once in the whole file, so nothing
downstream can resurrect the deadline the flip just nulled. A macro number could
not have caught a stray patched `run.py`, a tokenizer that travelled with the
checkpoint, or a gate that goes inert on the wrong stage — the diff can.

**Two variables, then, and it is stated rather than buried:** the CE-1
checkpoint, which the board will see, and one runtime line, which is
**score-neutral on completion** — it changes nothing the container computes when
it finishes, it only stops a *different* container being substituted silently.

**PRE-REGISTERED READING.** Above 0.49178 → the 2x-Stage-A lineage transfers,
and Stage-A volume becomes the funded axis (`t128vol2x2ep` is already queued).
At or below → the lineage does not transfer, E_real's deltas are dead for
checkpoint selection as well as for container composition, and the CE-1 slot is
settled at t116 until a genuinely different family lands.

### BOARD RESULT: v22 = 0.497252 — **NEW CHAMPION**, +0.00547 over v15

The pre-registered ACCEPT branch fires: *"above 0.49178 → the 2x-Stage-A lineage
transfers and Stage-A volume becomes the funded axis."* `t128vol2x2ep` has been
promoted from last to second in the GPU chain, behind only the queue15 timing
measurement.

**The board's own ladder, which is now a rule with receipts:**

| change | window | vars moved | board delta |
|---|---|---|---|
| v11→v14 backbone only | **held 256** | 1 | **+0.01313** |
| v11→v13 backbone + window | moved to 192 | 2 | −0.00662 |
| v14→v15 add CE-2 at full coverage | held 256 | 1 | +0.00323 |
| v15→v21 window + cascade + no GBDT | moved to 320 | 4 | −0.00522 |
| **v15→v22 CE-1 checkpoint only** | **held 256** | **1** | **+0.00547** |

Every single-variable change with the window held at 256 has won. Every change
that moved the window has lost. That is now five board points, not a slogan.

**E_real's delta transfer, measured a second time.** Local +0.01080 → board
+0.00547 is a ratio of **0.506**, against 0.628 for v14→v15 and `emix`'s ~0.65.
Two clean points, both with the correct sign, both in the 0.5–0.65 band — and
the one time the sign inverted (v21) was the one time the container crossed the
window axis E_real is known invalid on. So E_real is usable for *within-axis*
checkpoint selection at roughly half strength, and that is the honest reading.

**The level offset is NOT a constant, and this matters retroactively.** v15's was
+0.00585; v22's is +0.01118. `tools/v21_fallback_probe.py` leaned on that offset
being stable to price v21's rungs, so its conclusion was weaker than it looked —
already recorded as "unsupported, not refuted", and now known to rest on an
assumption the board has since contradicted. Use the *ratio*, not the offset.

**Did both cross-encoders actually run?** (`tools/v22_readback.py`) GBDT-only is
ruled out outright — it scores 0.29341 locally, ~0.2 below the observed board
score, so CE-1 certainly completed. The two remaining rungs sit 0.00118 apart
locally, which the ruler cannot separate, so the score alone does not *prove*
CE-2 ran. What did change is the mechanism: with `deadline_ts=None`, `left < need`
can never be true, so CE-2 is now **always attempted** and can only be missing if
something *threw*. Before, declining was normal designed operation. The
ambiguity moved from "by design" to "only on an exception" — which is the
difference between a container we cannot read and one we can.

## v23 + v24 — the mmBERT pair, built and verified 2026-08-23

| | v23 (safe rung) | v24 (the full swap) |
|---|---|---|
| shape | v22 CE-1 + **mmBERT@1024 as CE-2**, 30% cascade | **mmBERT@1024 as CE-1**, t120 bge demoted to 30% partner, e5 gone |
| local vs v22 | +0.00939 (16/16) | **+0.01634 (16/16)** — largest ever measured |
| raw min (our card) | 9.70 | 13.44 |
| sha256 | `53859d3ed27975f3…` | `18cf388b86e6a582…` |
| verified | 7/7 archive diff + gate replay | 7/7 + cross-check vs v23's mmBERT |
| smoked | full cascade in the shipping image, ALL PASS | same, reverse model order, ALL PASS |

Both carry the **cerace fix**: the shipped `ce.py` (since v15) runs 4 tokenizer
threads on one fast tokenizer and can die with `Already borrowed` on the first
concurrent wave — reproduced in the grader's own image by the smoke, closed
with a one-line warmup. Both keep the nofallback gate. Pre-registered readings
in `v23built` / `v24built`; v24's public score doubles as the grader-speed and
window-axis measurement no local ruler can make (`rulergrade`).

## v24 = 0.519802 — NEW CHAMPION (2026-08-24)

+0.02255 over v22 (0.497252): the largest single move ever recorded on this board.
The pre-registered >= 0.5000 branch fires: **the 1024 window transfers, mmBERT@1024 is
the funded main line, t130mmoleg ships in this container as v25 once it reads.**
Local container read (+0.01634) under-predicted the board (+0.02255) — the board's
near-dupe negatives reward the window more than our local universe does.
Public ran the full cascade inside the watchdog (score at prediction level). Private
timing still assumes H100 >= 1.07x our card; loud degradation path = mmBERT-alone.
v23 (mmBERT as 30% partner) is now superseded — do not spend a slot unless curiosity
is free; the next informative submission is v25 (t130 in the v24 container).

## v25 = 0.5213854511693545 — NEW CHAMPION (2026-08-25)

+0.00158 over v24 (0.519802). One variable: TEXT_CHAR_CAP 900 -> 2000 in src/ce.py
(two string literals; 42/43 archive entries byte-identical to v24 — `v25build`).
Local read was +0.00327 human-val (Kaggle T4 A/B, `charcap`), so the single-knob
transfer ratio is ~0.48 — opposite in sign to v24's container read, which
UNDER-predicted (+0.01634 local vs +0.02255 board). Working rule until more data:
human-val single-knob deltas halve on the board; container-level reads are the
better forecaster. Timing: uncapped text is ~7% more tokens and the cascade still
finished inside the public watchdog at prediction level — the "no fallback" design
holds. Next informative submission: v26 = v25 + `sortkey` token-sort patch once its
macro-invariance gate passes (then the freed ~30% wall-clock funds the backbone axis).

## v26 — built 2026-08-25, pending the image smoke (do not submit before it flips LANDED)

v25 + exactly one variable: `_score`'s batch sort key, characters -> true tokens
(`build_v26_sortkey.py`; 42/43 entries byte-identical, every diff hunk inside
`_score`). No accuracy delta claimed — the invariance gate read bit-exact
0.80296 = 0.80296 under token-sorted vs unsorted composition (kernel
`ecup26-sortinv`). What it buys: −41.6% padded tokens at 1024 ≈ −30% CE
wall-clock — private-timing margin (275k pairs / 780s) plus the compute budget
that funds the backbone-at-1024 axis. Gate before the slot: the cascade smoke
in the shipping image (the rewrite is concurrency-bearing — same class as
cerace), A/B'ing v26 against v25 on identical pairs.

**v26 smoke PASSED 2026-08-25** (shipping image, `tools/smoke_ce_v26.py`):
max |v26−v25| = 3.13e-07 / 4.77e-07 across both CE slots on identical pairs,
cascade contract exact, thread count inert, batches byte-equal to the
tokenizer's own output. **v26 is ready for a submission slot** — expected score
≈ v25 ± null; the payoff is ~30% CE wall-clock (private-timing margin + the
backbone-axis budget).

**v26 board = 0.5213519606588966 (2026-08-25)** — vs v25's 0.5213854511693545:
−0.0000335, parity at bit-noise level on the public set. The invariance gate's
bit-exact claim is now confirmed by the grader end-to-end. v25 remains the
nominal score champion by 0.00003; **v26 is the shipping baseline from here** —
identical score, ~30% less CE wall-clock, wider private-timing margin. v27+
build on v26.

## v27 — built and smoked 2026-08-25, READY (the first training gain since v22)

v26 with ONE variable: CE-1 weights swapped to **t136sa-ep1 (0.8049 human-val,
+0.0021 over the old champion, full provenance)** — the checkpoint that closed
`recipedrift` by completing Stage A past the mirror. 4-entry slot swap, 39
entries CRC-identical to the archive the board scored 0.5213520; tokenizer
proven id-identical in the shipping image; cascade smoke ALL PASS with the v26
token-sort active. Expected board ≈ +0.001 (0.5× single-knob transfer) — modest,
but it also promotes the reproducible lineage into the container, so every
future training win ships through a known base.

## v27 board = 0.5172128860585519 (2026-08-25) — REJECTED, −0.0041 vs v26

The +0.0021 human-val read on t136sa-ep1 sign-inverted on the board. Mechanism:
more Stage A = more LLM-label conformity, and the LLM labels err exactly on the
near-duplicate negatives the board tests — invisible to human-val's easy-negative
regime. **v25 (0.5213854) remains the nominal board best; v26 remains the
shipping baseline for all future builds.** Process change: container-shape
leak-free read gates every CE slot candidate before a submission; dose-direction
claims are board-only (the `mixaxisclosed` rule, now enforced without exception).


## v28 + v29 — built and verified 2026-08-26, queued for tonight's expiring slots

Both are ONE-CHANGE patches of v26 (the shipping baseline, 0.5213520); they are
independent — each reads against v26, so they can be submitted in either order
or both. PRE-REGISTERED BEFORE EITHER SCORE EXISTS.

### v28 — the blend constants, re-tuned on the container-shape read

`submission_v28.zip` (1.64 GB; 1 entry changed = run.py, 42 CRC-verified).
W_GBDT 0.1 -> 0.05, W_CE 0.7 -> 0.6, CE_COVER 0.30 -> 0.35 — constants fixed at
v24 build time and never re-tuned after the CE-1 strengthened. LEDGER
`arithsweep`: 74-config grid, SELECTED on seed-1 half-splits, CONFIRMED on
disjoint seed-2 splits at **+0.00292 sd 0.00134 16/16** (winner's curse
removed by construction). Inference-shape axis — the container read's home
turf (v24 forecast +0.016->board +0.023; v26 parity call exact). Timing: the
CE-2 skip-guard scales with CE_COVER; sortkey's freed ~30% dwarfs the +17%
CE-2 cost. **Expected 0.5234-0.5243. FALSIFICATION: v28 <= v26 -> the
container-shape ruler fails on its OWN axis; demote it to ordering-only,
revert constants, no further arithmetic slots.** If >= +0.0015: v28 becomes
the shipping baseline.

### v29 — CE-1 swapped to t148mlmA-ep1 (the MLM lineage's board question)

`submission_v29.zip` (1.64 GB; 4-entry CE-1 slot swap, 39 CRC-verified;
tokenizer id-identical to v26's on paired samples, raw-tokenizers check).
t148mlmA-ep1 = m0mmb MLM init + Stage-A(1) + B2, best local 0.8058 (+0.0030
over champion). DECLARED COMPOSITE vs the champion (MLM init AND dose A1-vs-A2
move together) — this slot answers "does the MLM lineage pay on the board",
a candidate question, not a mechanism isolation. Instruments disagree and
neither is trusted here: human-val +0.0030 (inverts on mix), container gate
-0.0049 (validated only for non-mix). Mechanism note: t148 carries LESS
Stage-A dose than the champion, so v27's failure mode (more LLM-conformity on
near-dup negatives) runs the OTHER way. **Expected band -0.004..+0.004.
READING: > v26+0.001 -> MLM lineage is the CE-1 line and the gate's sign was
wrong on this axis (matrix update); within +-0.001 -> keep v26, gate magnitude
miscalibrated; < v26-0.001 -> board rejects, gate vindicated, MLM stays
training-axis only. Whatever lands, the slot buys the axis answer + a gate
calibration point.**

Slots: 5 available today; these spend 2. No third gate-clean candidate exists
tonight — remaining slots stay unspent rather than manufactured.


## v28 = 0.5186466 / v29 = 0.5176789 (2026-08-26 evening) — BOTH REJECTED, both falsifications fired as written

**v28 (arith re-tune): −0.00271 vs v26.** The container-shape read said
+0.00292 sd 0.00134 at 16/16 on FRESH confirm splits — on its own home axis —
and the board inverted it. The select/confirm design removed winner's curse
within the ruler; the ruler itself is what failed. Mechanism (recorded as
hypothesis, not fact): the re-tune shifts weight between components with
different near-duplicate behavior (less GBDT with its code/attribute features,
more bge cascade), and the ruler's negatives are not the board's near-dups —
so the ruler rewards exactly the reweighting the board punishes. THE
PRE-REGISTERED CONSEQUENCE FIRES: container-shape ruler demoted; constants
stay 0.1/0.7/0.30; no further arithmetic slots.

**v29 (t148mlmA CE-1): −0.00367 vs v26.** The MLM lineage's +0.0030 human-val
does not transfer to the CE-1 slot. Per prereg: MLM stays a training-axis
fact; the mmBERT-t126 stack stays shipped. Note the twin sign-inversion with
v27 (t136sa +0.0021 local → −0.0041 board): two training-composition
candidates, two local-positive board-negative readings.

**THE STANDING CONCLUSION (supersedes the v27 process rule):** the
container-shape ruler's board sign record is now 2/4 (v24 right, v27 wrong,
t148 right, v28 wrong on its OWN axis) — a coin flip. No local instrument
prices container changes: not human-val (inverts on mix), not the container
gate (coin flip incl. its home axis). **Every slot candidate is board-only
from here.** Local rulers keep exactly two jobs: training-axis direction on
the SAME construction (the B-only ladders, mlmab, percat's paired control)
and timing/invariance engineering checks (sortkey's bit-parity held on the
board). Ladder unchanged: v25 0.5213854 nominal best, v26 0.5213520 shipping
baseline. Slots spent: 2 of 5 today; both bought instrument truth that no
local measurement could have.


## v30 + v31 — the last two expiring slots, 2026-08-26 late evening (user's call: "give us more information")

Board-only epistemology (slotruler) means expiring slots are the only
instrument left. PRE-REGISTERED BEFORE EITHER SCORE.

### v30 — CE-1 = t149baseA-ep0: DECOMPOSES v29's loss

t148 (v29, -0.0037) moved MLM init AND Stage-A dose together. t149baseA =
same A1+B2 construction, NO MLM, native @1024 (no window/remote-code risk),
local 0.8036. READING: v30 also ~-0.004 -> the A1-dose/local-family direction
is what the board rejects (MLM exonerated); v30 ~= v26 (+-0.001) -> the MLM
init was v29's loser; v30 > v26+0.001 -> LESS Stage-A dose PAYS on the board
(v27's conformity mechanism running in reverse) and the dose axis reopens
board-side. 4-entry slot swap on v26, ce1swap builder.

### v31 — CE-1 = t159euro610b-ep1 @768: RECON for t160 (score is secondary)

Buys tomorrow's unknowns tonight: (a) EuroBERT remote code under the baseline
image's transformers (code files ride in the slot dir, local auto_map,
offline-safe; trust_remote_code=True added in ce.py -- inert for slots
without custom code); (b) 610m@768 wall-clock inside the public budget (v26
sortkey active; CE_COVER guard adapts). Human explicitly accepted timeout
risk. READING: score >= ~0.50 -> euro ships mechanically + timing fits ->
t160's slot tomorrow is de-risked; ~0.29-0.45 or SIGALRM fingerprint -> the
container needs euro work (or quantization) BEFORE t160's slot -- learned on
a B-only throwaway instead of the 17h candidate; the t159 board point also
gives the euro family its first local->board anchor (local 0.7978).


## v30 = 0.5217320 / v31 = 0.3611549 (scored 2026-08-26 night) — both prereg readings fired as written

**v30 (t149baseA-ep0 CE-1): 0.5217319533 — NEW NOMINAL BOARD BEST**
(+0.00038 vs v26 0.5213520, +0.00035 vs v25 0.5213854). Within the prereg
+-0.001 band -> the reading is «the MLM init was v29's loser»: t149baseA
carries the SAME A1 dose as t148 minus the MLM init and lands at parity-plus,
so the A1-dose/local-family direction is exonerated and the MLM lineage loses
the slot (train-axis fact only, per v29). Three bonuses the slot paid out:
(1) first board proof that a NO-MLM A1+B2 construction equals the shipped
champion in-slot — t160's construction family is board-viable at the slot;
(2) it did so at ep0 (ONE Stage-B epoch) — the dose plateau holds on the
board side too; (3) nominal best. ANTI-RATCHET (v16g precedent): +0.0004 is
a coin-edge, the PRICING BASELINE stays «v26 == v30, either» — never promote
on a positive coin-flip. v30 is the shipped artifact of record for the
family; ladder: v30 0.5217320 nominal best, v26 0.5213520 co-baseline.

**v31 (t159euro610b-ep1 @768 CE-1): 0.3611549 — prereg band ~0.29-0.45
FIRES: the container needs euro throughput work BEFORE t160's slot.**
Decomposition from receipts, not vibes: the GBDT-only floor is a KNOWN
0.293 (v22 readback receipt, gbdt_only 0.29341 local ~0.29 board). 0.361
sits +0.068 ABOVE that floor -> the euro CE-1 **loaded and ran on the
grader** — a load crash lands exactly on the floor via run.py:253's
except-block (CE-2 is nested inside and dies with it). So the path that
fired is ce.py:188's deadline governor: it scored pairs until the projection
crossed the deadline, bailed, left the rest NaN, and GBDT filled the tail.
Coverage arithmetic: 0.293 + x*(0.52-0.293) = 0.361 -> x ~ 0.30 — the euro
scored roughly the shortest ~1/3 of pairs (traversal is length-sorted), i.e.
**throughput is ~3x short of the public budget**. On the user's «we removed
all fallbacks» point: the weak-model SUBSTITUTIONS are indeed gone (audited
the v31 zip itself); what remains by design are the three never-submit-Error
guards — budget-skip (run.py:163), deadline governor + NaN-fill
(ce.py:188), except->GBDT-only (run.py:253). The governor is the difference
between 0.361 and a wiped slot. WHAT THE SLOT BOUGHT: (a) euro remote-code
mechanics PROVEN on the grader image (load, trust_remote_code, @768 window
patch); (b) the entire gap is throughput, size ~3x, measured on a B-only
throwaway instead of the 17h t160; (c) a design flaw exposed — length-sorted
traversal makes a bail cover the LEAST valuable pairs; head-first traversal
(GBDT-rank-ordered, token-bucketed) turns any future shortfall from -0.16
into -0.0x, because PR-AUC lives in the head 30-40% (cascade receipt).
CONTAINER WORK QUEUE before t160's slot: (1) sortkey in TOKENS not chars
(41.6% padding-waste receipt) + (2) 900-char cap (+0.00327 confirmed; ship
together per the standing note), (3) head-first CE traversal, (4) a real
pairs/s bench of 610m@768 vs the shipped 210m@1024 on a box to size the
residual gap; quantization only if (1)-(3) do not close 3x.

Slots: all 5 of 2026-08-26 spent — v28, v29, v30, v31 scored; 4 prereg'd
questions, 4 answers, 1 nominal champion. Zero slots were spent blind.


## CORRECTION to the v31 postmortem (2026-08-27 morning) — the mechanism was WRONG; conclusions re-derived from the shipped code

The v30/v31 postmortem above claims v31's 0.361 came from ce.py's deadline
governor bailing after ~1/3 coverage. READING THE SHIPPED CODE REFUTES THIS:
under FORCE_CE=1 (the v26 default for scored stages, run.py:151)
deadline_ts=None, so ce.py:188's projection-abort is DISARMED; the only
bound is run.py's SIGALRM whose handler RAISES TimeoutError (run.py:52-53),
which the outer except (run.py:253) converts to "GBDT result retained" =
the 0.293 floor. v31 scored 0.361, not 0.293 -> THE ALARM NEVER FIRED ->
**the euro scored ALL 115k public pairs inside the window.** No fallback,
no partial coverage.

WHAT 0.361 ACTUALLY IS: the full-coverage quality of t159euro610b -- a
**B-only** checkpoint -- in the blend. Local 0.7978 -> board 0.361: the
training-mix axis sign-inversion at its most extreme (v27/v29's pattern,
magnified; B-only is the largest mix delta we ever put in a slot). The
"~3x throughput-short" reading is WITHDRAWN for the public stage.

WHAT SURVIVES, CORRECTED:
- Public timing FITS a full 610m@768 CE-1 pass (grader euro rate >= ~500
  pairs/s, >= 1.25x our Blackwell's benched 399/s -- bench610 row).
- PRIVATE (275k/780s) is the binding stage: a straight euro CE-1 there
  costs ~550s+GBDT -> over budget -> SIGALRM -> GBDT-only on the stage
  that decides the final. The cascade shape (euro as CE-2 on the head)
  remains the right slot design -- now for the private budget, not the
  public one. At the SHIPPED cover 0.30: euro CE-2 private ~120s, total
  ~640-690s < 780 with margin.
- t160euro610f (A1+B2, ep1 0.8089) carries the SAME mix construction v30
  just validated at the slot (t149baseA family, nominal board best) -- it
  fixes exactly what this correction says killed v31.

The bench610 and v31eurorecon LEDGER rows carry matching corrections.
Lesson filed: a Success score above the crash-floor is not proof a
governor fired -- read WHICH guards are armed under the shipped env
defaults before narrating a mechanism.


## v32 + v33 + v34 -- the t160 batch (2026-08-27 night, 3 slots, user's call)
## PRE-REGISTERED BEFORE ANY SCORE. Anchors: v26 0.5213520, v30 0.5217320.

All three carry t160euro610f-ep1 (610m A1+B2 @768, local 0.8089 all-time
best, weights sha 5a5862801a913a01). One variable apart from each other and
from their parents. The v31 correction governs the readings: 0.361 was
FULL-coverage B-only-mix quality, so mechanics+public timing are proven.

### v33 -- euro as straight CE-1 @768 (build_v31 machinery, t160 weights)
Public-information slot (a straight euro CE-1 likely overruns the PRIVATE
780s window, so v33 is not a final-pick candidate without quantization).
READING: >= ~0.50 -> the B-only mix WAS v31's killer; the euro backbone
family is slot-viable and its full-coverage public value is measured
directly. Still ~0.36-0.45 -> the container's transformers degrades euro
remote-code quality (H_code) -> quantize/vendor before any euro slot.
v33 vs v26 also prices backbone-at-CE-1 (mmBERT307@1024 -> euro610@768).

### v32 -- v26 + euro replaces bge in the CE-2 HEAD slot (cover 0.30 kept)
The private-safe design: CE-1 champion scores 100% (proven), euro re-scores
the top 30% of its ranking @768, batch 128; worst case = inner except ->
CE-1 alone (~v26-class), never a cliff. READING vs v26: > +0.002 -> the
head re-scorer pays, euro head architecture validated; within +-0.002 ->
euro head neutral at w=0.3 (weight/cover become the axis, board-only);
< -0.002 -> euro head hurts even at 30% coverage (H_code suspicion rises;
cross-check v33).

### v34 -- v30 + the same euro CE-2 head (the SHIP candidate)
Same one-variable delta as v32 but on the nominal-best base (CE-1 =
t149baseA-ep0). READING: v34-v30 should track v32-v26 (same mechanism on a
near-identical base); if v34 > all -> new nominal best and the default
final-pick candidate pending private-timing sanity (cover 0.30 private
fits per bench610: ~120s CE-2, total ~640-690s < 780s).

CROSS-READS bought by the batch: (v33 vs v26) backbone-at-CE-1; (v32 vs
v26) euro head on stock base; (v34 vs v30) euro head on best base; (v34 vs
v32) base equivalence under the new head -- 4 preregistered contrasts from
3 slots. Slots are the human's; agents never submit.

---

## HOLD ON v32 / v33 / v34 — do not submit until the docker parity gate passes
### Raised 2026-08-28 by max, after oleg's `gradenv` / `v15lb`.

**The grader image is `transformers 5.14.1`; our boxes train on 4.57.3.**
Confirmed here by running the image directly: `tf 5.14.1, torch 2.10.0+cu128,
py 3.12.3`. Two independent consequences, both measured, neither known when
the batch was preregistered above.

**1 — euro may be computing noise on the grader.** oleg scored our euro family
inside the image against its own box dump: with the ROPE patch it loads and
returns **corr 0.2597**. A model at corr 0.26 with its training scores is not a
weaker model, it is an unrelated one. All three archives carry euro:
- **v33** puts it at CE-1, scoring every pair. Expect a v31-class result.
- **v32 / v34** put it in the CE-2 head, re-scoring the top 30% of the
  champion's ranking. A noise head there does not degrade gracefully — it
  actively corrupts the band the metric cares about most.

**This is now the leading explanation for v31 = 0.3611549, and it is my third.**
I previously said (a) the deadline governor bailed at ~1/3 coverage — refuted by
reading the shipped code; then (b) it was full-coverage B-only-mix quality. But
euro-at-corr-0.26 predicts a score just above the GBDT floor (0.293), which is
what 0.361 is. Explanation (b) required a well-trained model to lose 0.16 to a
training-mix change; explanation (c) requires only that the forward is broken,
and oleg measured that independently. Treat (b) as superseded.

**2 — our CE-1 has a rope desync of its own, and it is not euro-specific.**
`outputs/ckpt-t149baseA-ep0/config.json` carries 5.x-form `rope_parameters`
with `sliding_attention.rope_theta = 160000` and NO 4.x-form
`local_rope_theta`. Verified with AutoConfig under 4.57.3: 4.x falls back to
the class default **10000** while 5.x reads **160000**, and `layer_types` is
full/sliding/sliding repeating — so two thirds of the layers. Isolating that
one field on the real weights (`tools/ropeparity.py`, 300 pairs):

    pearson 0.96751   spearman 0.97333   mean|d| 0.506   max|d| 2.280

Below the 0.999 bar, and worse than the 0.9768 that cost oleg his t158 column.
If the full image test agrees, **every slot since v24 has paid this tax**, and
the fix is one field: `rope_parameters.sliding_attention.rope_theta = 10000`,
which 4.x ignores entirely, so boxes are unaffected.

**Status:** `tools/run_parity.sh` is running the authoritative test — the same
300 pairs scored under local 4.57.3 and inside `odsai/...:1.0`, for both the
CE-1 champion and the euro CE-2. Gate: corr >= 0.999.
- euro FAILS  -> v32/v33/v34 are dead as built; do not spend slots on them.
- CE-1 FAILS  -> rebuild v30 with the one-field fix; that becomes the arm, and
  it is a free-points candidate rather than a new-model bet.
- both PASS   -> lift this hold, the batch reads as preregistered above.

Nothing above changes the `oofstack` result or the v30 ladder. It changes which
archives are safe to put in front of the board.

---

## v31 = 0.3611549 RESOLVED — it was the GBDT-only fallback. Third explanation, and the first one with receipts.
### 2026-08-28, max. The human said this on the day: *"seems that 0.36 cause of SOME FUCKING FALLBACK."* They were right, and I argued otherwise twice.

**The chain, every link measured this session, none inferred.**

1. The grader image runs **transformers 5.14.1** — ran it:
   `tf 5.14.1, torch 2.10.0+cu128, py 3.12.3`.
2. In 5.14.1, `ROPE_INIT_FUNCTIONS` = `['dynamic','linear','llama3','longrope','proportional','yarn']`.
   **There is no `'default'` key.** In our 4.57.3 there is. Ran the probe in both.
3. Our vendored `modeling_eurobert.py` line 263 sets `rope_type = "default"` when
   `config.rope_scaling is None`, then line 268 does `ROPE_INIT_FUNCTIONS[self.rope_type]`.
   Our euro config has `rope_scaling: null`. → **KeyError('default') at instantiation.**
4. That file is **byte-identical (sha `c800961405876db2`) in v31, v33, v34 and the
   local checkpoint.** So v31 shipped code that cannot construct its own model on the grader.
5. v31's `run.py:253` catches any cross-encoder exception:
   `log(f"ERROR: cross-encoder failed ({exc!r}); GBDT result retained")`.
   The CE contributes nothing and the archive scores **GBDT alone**.
6. GBDT-alone on the board: **0.33346 measured** (v10, the weaker human-only variant)
   and **~0.365 projected** for the pooled v8 GBDT this container carries.
   v31 scored **0.3611549**. That is the GBDT, not a damaged euro.

**Where my two earlier stories broke, so the error is reusable.** Both rested on one
sentence in the v30/v31 postmortem above: *"the GBDT-only floor is a KNOWN 0.293 (v22
readback receipt, gbdt_only 0.29341 local ~0.29 board)."* **0.29341 is a LOCAL number.**
It was promoted to "~0.29 board" with no board receipt, and then `0.361 != 0.293` was used
as proof that the CE must have run — which killed the fallback hypothesis and forced me
into first a deadline-governor story and then a training-mix story. The board's own
GBDT-only receipt, 0.33346, was in this same file at line 14 the whole time.

> **Rule.** Never compare a board number against a local number and read the gap as
> mechanism. A floor claimed for the board needs a board receipt. We have a ruler-domain
> memory for exactly this and I walked into it anyway.

**Consequences for the three queued archives — all three are dead as built.**

| archive | euro slot | what the grader will do | expected |
|---|---|---|---|
| **v33** | CE-1 | KeyError → `run.py:253` → GBDT retained | **~0.36, a repeat of v31** |
| **v32** | CE-2 head | KeyError → inner `except` at `run.py:240` → `"CE-2 failed, CE-1 alone"` | **≈ v26**, buys nothing |
| **v34** | CE-2 head | same inner except | **≈ v30**, buys nothing |

v32/v34 degrade safely — the cascade's inner except is doing its job — but a slot that
returns a number we already have is still a spent slot. **Do not submit any of the three.**

**The KeyError is currently protecting us.** Do not "fix" it with a bare rope shim: oleg
measured that euro WITH the rope patch forwards at **corr 0.2597** against its box scores
under 5.x (attention-mask path). Patching the crash without fixing the masks converts a
safe fallback into a harmful noise column — precisely what cost him two slots on t158,
which scored 0.448 / 0.454 against a 0.4672 champion because mmBERT *loads* and therefore
has no protective crash. Euro ships only after corr >= 0.999 in the image.

---

## Parity gate RESULTS, and the root fix that beats the config patch
### 2026-08-28, max. Numbers from `odsai/ecup26-matching-baseline:1.0` running locally.

**CE-1 champion (mmBERT @1024), 300 pairs, local 4.57.3 vs the image's 5.14.1:**

    pearson 0.99383   spearman 0.99258   mean|d| 0.2901   max|d| 1.2858   -> FAIL (gate 0.999)

**Correction to my own earlier entry:** the one-field proxy in the HOLD section
above read **0.96751** and I wrote it up as "worse than the 0.9768 that cost oleg
his t158 column." The image says **0.99383** — the proxy was pessimistic by a
wide margin, because it isolated one field under 4.x rather than running the
real 5.x path, and the rest of that path evidently compensates. The tax is real
and fails the gate, but it is **milder than oleg's t158**, not worse. Expected
board value of fixing it: positive, unknown, and small — we have no corr→board
transfer coefficient, and his single data point (0.9768 → −0.013/−0.019) came
from a stack column, not a CE-1 computing the whole ranking.

**euro-610m: no dump at all.** The image leg produced no scores — `KeyError('default')`
at instantiation, confirmed end to end, exactly as the ROPE_INIT_FUNCTIONS probe predicted.

### The root fix: vendor transformers 4.57.3 into the archive

Both failures are one cause — the container runs a different transformers than we
trained on. We already vendor `lightgbm` and `rapidfuzz` because the image lacks
them. Verified working inside the image:

    PYTHONPATH=/v  ->  transformers 4.57.3, hub 0.35.3, tokenizers 0.22.2 (image's own),
                       torch 2.10.0+cu128 (image's own), 'default' in ROPE_INIT_FUNCTIONS: True

61 MB of **pure-python wheels only**: transformers 4.57.3, huggingface_hub 0.35.3
(4.57.3 pins <1.0; the image ships 1.27.0), requests + urllib3/certifi/idna/
charset_normalizer (5.x dropped requests). No compiled dependency is added — the
image's own `tokenizers 0.22.2` already satisfies 4.57.3's `>=0.22,<0.23` pin, and
torch is untouched.

If the forward parity confirms, this is strictly better than the one-field config
patch, because it fixes **both** problems at the root and makes every local number
we own directly transferable instead of needing a per-model parity check forever:
- CE-1 reads its config the 4.x way it trained under -> the rope tax disappears;
- euro instantiates -> the euro lane has somewhere to ship, which currently it
  does not, and Box B is spending ~8 GPU-hours on `t165euroA2` right now.

**Still gated.** `tools/run_parity_full.sh` is measuring the forwards. Asserting a
fix from its mechanism is precisely the error that produced two wrong v31 stories,
so nothing ships until corr >= 0.999 is on the board for both models. Note also
that vendoring changes the container's import surface, which is a **container-shape
change and therefore BOARD-ONLY** under `slotruler` — the parity gate licenses the
engineering, not the score.

**Candidate ladder once parity lands:**
1. `v35` = v30 + vendored 4.57.3, nothing else. One variable, no new weights, and
   it prices the whole environment-desync axis in a single slot.
2. `v36` = v35 + the euro CE-2 head (what v34 was supposed to be) — only if euro
   passes parity under the vendored stack.
`submission/build_v35_ropefix.py` (the one-field alternative) stays as the fallback
if vendoring turns out to break something the parity test does not cover.

---

## PARITY RESOLVED: vendoring transformers 4.57.3 gives bit-exact parity on BOTH models
### 2026-08-28, max. Measured in `odsai/ecup26-matching-baseline:1.0`, 60 pairs @256.

| model | environment | pearson | spearman | mean abs d | |
|---|---|---:|---:|---:|---|
| CE-1 mmBERT | image, stock 5.14.1 | 0.99390 | 0.99111 | 0.2854 | FAIL |
| CE-1 mmBERT | **image + vendored 4.57.3** | **1.00000** | **1.00000** | **0.0000** | **PASS** |
| euro-610m | image, stock 5.14.1 | — | — | — | `KeyError('default')`, no dump |
| euro-610m | **image + vendored 4.57.3** | **1.00000** | **1.00000** | **0.0000** | **PASS** |

The stock euro traceback, for the record:

    File ".../modeling_eurobert.py", line 268, in __init__
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    KeyError: 'default'

**`submission_v35.zip` is built** — v30 + `vendor/` + one `sys.path.insert`, 1.65 GB
(+13 MB compressed), every other entry byte-identical to v30, sha `764afaaae4075b08`.
Builder: `submission/build_v35_vendored.py`.

**Two corrections to my own earlier entries in this file.**
1. "We already vendor lightgbm and rapidfuzz" — **not true of v30**, which ships no
   `vendor/` at all (its GBDT is pure numpy/pandas). v35 adds the directory and the
   wiring; it is a new mechanism, not a reuse.
2. The one-field proxy read corr 0.96751; the image says 0.99383 @1024 and 0.99390
   @256. The proxy was **pessimistic**, so the CE-1 tax is milder than oleg's t158
   (0.9768), not worse — I had it backwards.

**Gates still open before any slot.**
- Re-run `tools/run_parity_full.sh` at SHIPPING length (CE-1 @1024, euro @768). The
  1.00000 above is 60 pairs @256 and proves the code path, not the shipping config.
- Smoke the extracted v35 layout inside the image (attempted, blocked by a full disk).
- `slotruler`: container shape is BOARD-ONLY regardless of how clean the parity is.

**Candidate ladder, in order.**
1. **v35** = v30 + vendored 4.57.3. One variable, no new weights. Prices the entire
   environment-desync axis in one slot, and is the only one of these that can be read
   cleanly against v30 0.5217320.
2. **v36** = v35 + the euro CE-2 head (what v34 was meant to be). Only meaningful
   *because* of v35 — euro cannot instantiate without it.
Both supersede the held v32/v33/v34, which stay dead as built.

---

## v37 = v35 + a preflight that refuses to fall back on a build error
### 2026-08-28, max. Built after the human asked, again, why there are fallbacks in the submission.

They were right to ask. v22 removed the deliberate degradation paths
(`FORCE_CE=1` disarms the deadline governor and the no-CUDA skip). It did not
remove the exception handlers, and `run.py:253` — a bare `except Exception`
around the whole cross-encoder — is what turned v31's `KeyError('default')`
into a GBDT-only CSV scoring 0.3611549.

**v37 adds a preflight before the GBDT** that, for every CE slot, resolves the
config, imports the remote code, and constructs the model on torch's `meta`
device (no memory, no weights, but the exact path that failed in v31), and
loads the tokenizer. On failure it raises `SystemExit` and nothing catches it:

    PREFLIGHT FAIL models/ce-2: KeyError('default') -- this is a BUILD error,
    the archive cannot run this model in this environment. Refusing to fall
    back to a partial container and report a plausible score (that is what
    v31 did: 0.3611549 was the GBDT alone).

The timeout fallback is deliberately KEPT — on the private stage a partial
result genuinely beats nothing. What is now impossible is shipping a model that
cannot load and getting a number back that looks real.

`submission_v37.zip`, sha `c65ce158f00609e2`, 1.65 GB. Everything except
`run.py` byte-identical to v35.

**OUTSTANDING — the negative test.** The preflight has been proven to build and
parse, NOT to fail on a broken model. A safety net that has only been seen
passing is not yet a safety net (COOKBOOK: validate with a negative test — the
checkpoint guard rule exists for this reason). The test is: run the preflight
against the euro slot under STOCK 5.14.1 in the image and confirm it raises.
Blocked on Docker Desktop, which is wedged (`docker ps` returns empty, rc=0).

**Also rebuilt:** v35 had mixed line endings in `run.py` (277 CRLF + 8 bare LF)
because my insert used LF into an all-CRLF file. Python-legal but a needless
diff from a known-good artifact. Both builders now match the base's endings and
refuse to emit a mixed file. New v35 sha `c9eeacb56c9edcc2`.

### dist/ cleanup, human-authorised
Deleted 10 superseded archives, **freed 17.6 GB** (disk 99% -> 95%). Kept only
`v26` (co-baseline), `v30` (nominal best and the base v35/v37 are built from),
`v35`, `v37`. All deleted archives are reproducible from a builder in git plus
weights on Kaggle, or downloadable from the competition platform.

---

## v32 = 0.5156269726390332 (scored 2026-08-28) — the euro head did not run, and that priced the bge head
### My numeric prediction was wrong; the mechanism prediction held.

I wrote above that v32 would land "≈ v26". It landed **0.005725 BELOW** v26
(0.5213520). The mechanism was right — euro cannot instantiate on the grader
(`KeyError('default')`, reproduced directly in the image on the byte-identical
shipped file `sha c800961405876db2`), so `run.py:240`'s inner except fires and
the container runs **CE-1 alone**. What I got wrong is the arithmetic: **v26's
CE-2 head is doing work**, so CE-1-alone is v26 *minus the head*, not v26.

**Which makes v32 an accidental but clean ablation, and the first board price
we have ever had for the cascade head:**

    v32 (no head)  0.5156270
    v26 (bge head) 0.5213520
    -------------------------
    bge CE-2 head at cover 0.30 is worth +0.005725 on the board

Clean because the other two v32 deltas (`MAX_LEN_2` 256→768, `BATCH_SIZE_2`
512→128) only touch the CE-2 path that never executed. Against a board scatter
of ~0.0004 (v25 vs v26 differ by 0.00038 and we call that noise), −0.0057 is
~14x that. LEDGER `ce2headworth`.

**Three things this buys.**
1. The head **architecture pays** — +0.0057 — so a working euro head has a
   concrete target rather than a hope.
2. A **third independent confirmation** that euro is not running on the grader,
   after the ROPE_INIT_FUNCTIONS probe and the in-image traceback.
3. It **sharpens the unspent predictions**, so neither needs a slot:
   - v33 (euro at CE-1) → outer except → GBDT alone → **~0.36**, a v31 repeat.
   - v34 (v30 + euro head) → CE-1 alone → **0.5217320 − 0.0057 ≈ 0.5160**.

### v35 rebuilt — the tokenizer would have reproduced the v31 failure
`tools/test_preflight.py` (the negative test) caught it before a slot: our
checkpoints were exported under a 5.x venv, so `tokenizer_config.json` carries
`"tokenizer_class": "TokenizersBackend"` (a 5.x class) **and**
`extra_special_tokens` as a **list** (4.57.3 wants a dict, and a list raises
`AttributeError: 'list' object has no attribute 'keys'`). `src/ce.py:86` calls
`AutoTokenizer.from_pretrained()` with **no fallback**.

So v35-as-first-built — vendoring 4.57.3 to fix euro — **would have died at
tokenizer load, been caught by run.py:253, and scored GBDT-only**: the exact
failure the change was meant to prevent, reintroduced by the change itself.

Fixed in the build (`build_v35_vendored.py`), verified two ways: the tokenizer
now loads under 4.57.3, and `tools/tokclass_probe.py` shows it produces token
ids **identical** to the raw-`tokenizer.json` loader that scored the 1.00000
image parity. New v35 sha `49503e45c4c86a68`; v37 = v35 + preflight, sha
`1f3eea79aa0ded6d`.

**Preflight negative test now 3/3** (good config returns in 3.8s; nonexistent
rope_type raises; missing config raises). Still outstanding: the real
5.14.1 `KeyError('default')` path, which needs the grader image — Docker is
wedged. Until that runs the preflight is verified in mechanism but not against
the exact failure it was written for.

---

## Candidate ladder after the v32 reading — v35 then v36, in that order
### 2026-08-28, max. All four container defects found today are now fixed and composed.

| archive | = | sha | what it changes vs its base |
|---|---|---|---|
| `v35` | v30 + vendored tf 4.57.3 + CE-1 tokenizer fix | `49503e45c4c86a68` | the container runs the transformers every checkpoint trained under |
| `v37` | v35 + preflight | `1f3eea79aa0ded6d` | a build error becomes fatal instead of a silent GBDT fallback |
| `v36` | v37 + euro in the CE-2 head | `ae5dad849b18f410` | swaps the bge head (board-priced at +0.0057) for our best model |

**v36 composition verified, 13/13:** vendor present, `sys.path` wired before
the numpy import, preflight present and called before the GBDT, CE-1 tokenizer
fixed both fields, euro in ce-2 with its remote code, `ce.py` carrying
`trust_remote_code`, `MAX_LEN_2` 768 / `BATCH_SIZE_2` 128, bge's stray
`special_tokens_map.json` dropped, single line-ending style.

**v36's own preflight run against v36's own slots, locally under 4.57.3:**

    preflight OK models/ce-e5-base (modernbert, 8.1s)
    preflight OK models/ce-2      (eurobert,  1.6s)

~10s against a 360s/780s budget. Both slots actually visited, not skipped.

### Order, and why it is not "just ship v36"
**Spend the first slot on v35.** It moves ONE thing — the environment — against
v30 0.5217320, so it answers two questions at once with a clean attribution:
is the vendoring safe in the grader, and was the CE-1 rope tax (corr 0.9939)
real on the board?

v36 bundles the environment fix AND the head swap. If it wins, fine; if it
loses we cannot tell which half did it — and `ship one variable` is CONFIRMED
board history here, not a preference (v21 moved four and lost, v22 moved one
and won +0.0055).

**Expected values, stated before the readings so they can be wrong in public:**
- v35 − v30: small positive. Removes a corr-0.9939 desync on CE-1. No transfer
  coefficient exists from corr to board points, so the size is a guess; the
  SIGN is the claim.
- v36 − v35: the interesting one. The bge head is worth +0.0057 (v32's
  accidental ablation) and euro is far stronger locally (0.8089 vs bge's tier),
  so a working euro head should beat it. This is the first container in which
  euro can actually execute.

**Timing:** bench610 puts euro CE-2 at cover 0.30 near 120s on private, total
~640-690s against 780s, plus ~10s preflight. Fits, with less margin than v30.

**Residual risk on all three, unchanged:** the shipping-length parity
(CE-1 @1024, euro @768) has not been re-run since Docker wedged. Today's
1.00000 is 60 pairs @256 and proves the code path, not the shipping config.

### v36 static audit — the two ways vendoring could silently fail, both cleared
`tools/check_vendor_order.py`, run on the shipped archive.

**1. A pre-import would defeat the whole thing.** Python caches by name in
`sys.modules`, so if ANY module imported transformers before `vendor/` reached
`sys.path`, the container would run the image's 5.14.1 while every log line
claimed 4.57.3 — a silent failure of exactly the class we spent today chasing.
Audit of the shipped `run.py`: vendor insert at **line 42**; the only
module-level imports before it are `argparse, os, sys, time` (lines 19-22), and
`numpy`/`pandas` come after at 44-45. None of `src/*.py` imports transformers at
module scope, and `src/ce.py` imports it **inside a function** (line 78), which
runs long after. **Nothing preloads transformers.**

**2. The preflight must not run on the check stage.** 1,000 pairs against a 60s
budget, where ~10s of model construction would be 17% of it — and the CE is
legitimately skipped there anyway. Shipped guard, verified in the archive:

    if n > 10_000 and os.environ.get("SKIP_CE", "") != "1":
        _preflight_ce(HERE, log)

So the check stage skips it and both scored stages (115k public, 275k private)
run it. That matches `force_ce`'s own condition, so the preflight can never
disagree with the thing it is protecting.

Every static risk on v35/v36 that can be checked without Docker is now closed.
What remains is the shipping-length parity, which needs the image.

---

## v37 = 0.3611536 SOLVED — `reference_compile`. Reproduced in the image, fixed, and proven end-to-end.
### 2026-08-28, max. No inference this time: every line below is a command output.

**ROOT CAUSE.** ModernBERT in transformers **4.x** `torch.compile()`s its
embeddings when `reference_compile` is unset (it defaults on for CUDA).
That compile dies in the grader image:

    File ".../modeling_modernbert.py", self.compiled_embeddings(input_ids)
    File ".../_inductor/compile_fx.py", raise InductorError(...)

**5.14.1 does not take that path.** So v30 (stock 5.14.1) scored 0.5217320 and
v37 — same weights, same window, only *vendored to 4.57.3* — died at forward
time, hit `run.py:253`, and kept the GBDT: 0.3611536.

**Measured in the image on a real GPU, 256 pairs @1024:**

| stack | result |
|---|---|
| stock 5.14.1 | 75.2 pairs/s |
| vendored 4.57.3 | **InductorError** |
| vendored 4.57.3 + `reference_compile: false` | **90.6 pairs/s** |

So the vendored stack is ~20% FASTER than stock once the compile path is off.
The fix is one config field, no `ce.py` change.

**END-TO-END PROOF** — rebuilt v37, grader image, `--gpus all`, 11,000 pairs:

    ce: model loaded on ... GPU dtype=torch.bfloat16
    ce: scored 11000/11000 complete=True
    wrote 11000 rows
    rows 11000 | unique preds: 1484 | min 0.0001 max 0.9929

(The GBDT half reports missing `lightgbm` — that is absent from the *public*
image I pulled, not from the grader, which ran it for v30.)

### Why my preflight did not catch it — a design error, not a bug
The v37 preflight constructed each model on torch's **`meta` device**: fast, no
weights, no forward. **The failure was at FORWARD time.** A check that never
runs a forward is structurally incapable of seeing it, so it passed and the
container proceeded to fail exactly as before.

Rewritten to load **real weights** and run a **real forward** on the real
device. Verified in the image on GPU against v37's own slots:

    preflight OK models/ce-e5-base on cuda (12.4s, logit -0.9033)
    preflight OK models/ce-2      on cuda (16.1s, logit -5.7070)

Marginal cost ~28s (the models load twice), against 360s public / 780s private.
That is the price of not shipping a third 0.36.

**Rule this adds:** a preflight must exercise the SAME operation the container
performs. Construction, config parsing and tokenizer loading are each necessary
and none is sufficient. If the thing you are protecting is a forward, preflight
a forward.

### Rebuilt, with the fix
| archive | sha | contents |
|---|---|---|
| v35 | `6767fbc4501d7c4e` | v30 + vendored 4.57.3 + tokenizer fix + `reference_compile: false` |
| v37 | `aa816a7d55ee66d9` | v35 + real-forward preflight |
| v36 | (rebuilt on v37) | v37 + euro CE-2 head |

**What is still NOT proven:** that any of these beat v30 on the board. Only a
slot answers that. What IS now proven is that the container runs its
cross-encoder to completion in the grader's own image on a GPU — which was true
of neither v31 nor the first v37.

### Preflight narrowed to CE-1, and v36 proven end-to-end with the euro head

**CE-1 ONLY.** A CE-2 failure already degrades SAFELY -- `run.py`'s inner except
logs "CE-2 failed, CE-1 alone" and every pair still gets scored by the champion,
which is v30-class, not 0.36. Only CE-1 failure drops the container to the GBDT.
So preflighting CE-2 buys no protection and costs real budget:

    two-slot preflight (v36 first build)   293.2s
    CE-1-only preflight (v38 / v36)         52.7s

5.6x cheaper, same protection. Principle: **protect what fails hard; let what
degrades gracefully degrade.**

**v36 END-TO-END, grader image, `--gpus all`, 11,000 pairs** -- the euro head
executing in the cascade for the first time:

    ce: scored 11000/11000 complete=True
    CE-2: cascade to top 3300/11000 (30%) of the CE-1 ranking
    ce: model loaded on ... GPU dtype=torch.bfloat16
    ce: scored 3300/3300 complete=True
    CE-2 ok (CE-1 82s)
    wrote 11000 rows
    done in 243s (stage budget 360s)

Both stages complete, inside budget, on a laptop GPU.

### The ladder, all proven in the image on GPU
| archive | sha | preflight | status |
|---|---|---|---|
| v37 | `aa816a7d55ee66d9` | both slots (60s) | end-to-end proven |
| v38 | `9803c9464229511e` | CE-1 only (53s) | same as v37, leaner |
| v36 | v38 + euro head | CE-1 only | end-to-end proven, `CE-2 ok` |

**Order:** v38 (or v37 -- they differ only by ~7s of preflight) to price the
environment fix against v30 as one coherent change; then v36 to add the euro
head, whose bge predecessor v32 priced at +0.0057.

Neither is proven to BEAT v30 -- only a slot answers that. What is now proven,
and was true of neither v31 nor the first v37, is that the container runs its
cross-encoder to completion in the grader's own image on a GPU.

---

## v39 — SUBMITTED 2026-08-29 — **0.5230454881 — NEW CHAMPION**

**+0.0013135348 over v30 (0.5217319533).** Ends five consecutive losing slots
(v31, v32, v34, v36, v37). The archive is the v30 container with exactly one zip
entry changed: `models/ce-e5-base/model.safetensors`, CE-1 weights swapped from
`t149baseA-ep0` to `t167llm2x-ep1`. Stock runtime — v39 does **not** carry the
vendored 4.57.3 tree; that stays v40's variable.

**Read the projection against the outcome before trusting the next one.**
`ce1swap` projected "~+0.006 board at emixboard's 0.65 transfer". It paid
+0.00131, i.e. a transfer of **0.078** on the E_real container read. See
`transfergap` in the LEDGER: at nearly the same local size (+0.01634) the
v22→v24 move paid +0.02255, a ratio of 1.38. **Two moves of the same local
magnitude, board outcomes 17× apart.** The usable conclusion is negative —
a local container delta does not predict board magnitude, in either direction.

**It is not a clean one-variable receipt, despite being one changed file.**
t167llm2x-ep1 differs from t149baseA-ep0 in *two* training variables: 2× Stage-A
LLM dose (`dose2x`) and Stage-B ep0→ep1. `sbep1` prices the epoch leg alone at
+0.00896 solo E_real, so the dose leg carries the remainder. Fine as a slot,
wrong as an attribution.

**The margin is 1.88× the upper bound of a floor nobody has measured.**
`receiptcensus` puts the determinism floor in [3e-05, 7e-04]; there has still
never been a repeat submission. A byte-identical resubmission of v39 measures it
for one slot and cannot lower our best score.

---

## v40 — SUBMITTED 2026-08-29 — **0.522171008743644 — REJECT**

**−0.0008744794 against v39.** v40 is v39 plus the vendored transformers 4.57.3
runtime and *nothing else* — same CE-1 weights, same CE-2, same GBDT, same
container arithmetic, same window, verified entry-by-entry by `v38diff`. **This
is the cleanest one-variable board experiment this project has run**, and the
answer is that the environment fix costs points.

The two slots decompose exactly:

```
CE-1 swap alone   (v39 − v30) = +0.0013135348
runtime alone     (v40 − v39) = −0.0008744794
both together     (v40 − v30) = +0.0004390554     ← the legs sum to the joint
```

**The rope story was right about the mechanism and wrong about the value.** The
bug is real: 5.14.1 reads ModernBERT's rope config by the 5.x path and gives
CE-1 a sliding theta of 160000 where training used 10000. Correcting it moves
the board *down*. corr 0.9939 was always the tell — the two runtimes agree on
almost the whole ranking, so the residual was never worth much either way, and
`envfixlocal`'s CI95 [−0.00413, +0.01081] crossed zero. The board picked the
negative half.

**Is it noise?** Partly, and it must be said: |0.00087| is 1.25× the *upper*
bound of the determinism floor `receiptcensus` bracketed at [3e-05, 7e-04] and
never measured. The sign is not beyond doubt. What *is* established is that the
fix is not worth a slot, because its entire effect lives at the scale of a floor
we have never measured.

**Consequences.** Do not spend a slot on **v38** (env fix alone on the v30 base
— it would land ≈0.5209, below v30). Drop the vendor tree: 2,530 files and +85
lines of run.py for a board negative. And euro-610m *cannot construct* under
5.14.1 (`euroconstruct`, `v31resolved`), so it **requires** this runtime — any
future euro container starts 0.00087 in the hole, on top of v36's −0.0074.

**What it vindicates: `shiponevariable`.** Bundled into one submission these two
changes would have read +0.00044 and taught us nothing about either leg — which
is exactly what happened to v36, unreadable ever since. The human split them.

---

## Superseded: the v39/v40 build notes (both now have receipts)

`ce1swap` marked this ACCEPT on 2026-08-28 and it was never assembled — both
slots after it went to the euro batch (v36 0.5143414, v37 0.3611536). The
checkpoint is `gordeevmax/ecup26-t167llm2x-ep1`.

| archive | base | sha | changed vs base | notes |
|---|---|---|---|---|
| v39 | v30 | `f7fed5e03e5b8113` | **1 entry** (CE-1 weights) | one variable vs the champion |
| v40 | v38 | `ab885aca79811049` | **1 entry** (CE-1 weights) | also carries the env fix |

Verified independently of the builder: exactly one entry differs in each,
`models/ce-e5-base/model.safetensors`, zero added, zero removed. Both carry the
same weights (CRC 2776531587).

**Price** (`ce1swap`, ACCEPT): E_real container **+0.01690** (0.50916 → 0.52606)
from a solo +0.02217; E_mix container +0.00951 ⇒ ~+0.006 board projected. **THAT PROJECTION IS NOW SETTLED AND IT WAS 4.6× TOO HIGH: v39 paid +0.00131.** For v40 the same arithmetic on `envfixlocal`'s +0.00308 gives ~+0.00024 board at the v39 transfer — *below* the upper bound of the unmeasured determinism floor — or +0.0042 at the v24 transfer. The two receipts disagree on whether v40 is worth a slot, and no local measurement can break that tie. The change class has a receipt: **v22 → v24 was a CE-1 swap and paid
+0.0225**. The training change behind it (`dose2x`, ACCEPT) read E_real +0.01469
with its CI lower bound (+0.00849) already above the bar.

**It survives the 2026-08-29 ruler audit.** Both readings are on E_real, the
non-mined pool, so `eminediso` does not touch it — unlike `neg6x`, `blendwhard`
and `partnersweep`, which all died on that test the same day.

**Choosing the base is a real tradeoff, not a formality.** v39 moves one thing
against the champion so a result attributes cleanly, but runs on v30's broken
runtime (rope theta 160000 against the 10000 training used) while the +0.01690
was measured under the correct one. v40 runs the CE-1 in the runtime its
measurement assumed and stacks a second verified improvement, at the cost of two
moving parts. `ship-one-variable` argues v39; "measure it where you measured it"
argues v40.

**Trap caught during the build:** the new checkpoint's `config.json` and
`tokenizer_config.json` are byte-identical to **v30's**, so they lack
`reference_compile=false` and carry the old `tokenizer_class`. Swapping them
straight into the v38 base would have silently reverted both environment fixes
and re-created the InductorError that scored v37 at 0.3611536. v40's CE-1 dir
takes weights + `tokenizer.json` from the new checkpoint and the two fixed files
from v38, with asserts before the build ran.

Neither archive has been run in the grader image. v38's vendor tree is
board-proven via v36; v30's stock runtime is board-proven directly.

---

## v41 — BUILT + GRADER-VALIDATED 2026-08-30, awaiting a slot (last submission day)

**`submission/dist/submission_v41.zip`, sha256 `7b9acffa2afef604`, 1,760,205,825 B.**

**One variable, and it is proven rather than asserted.** 43 entries; 42 copied
byte-for-byte from the v39 archive that scored 0.5230454881; exactly 1 replaced.
Verified by entry-by-entry CRC comparison — the only differing entry is
`models/ce-e5-base/model.safetensors` (CE-1 weights), same file_size 615,076,194.
`config.json`, `tokenizer_config.json` and `tokenizer.json` are byte-identical
between the raw checkpoint and v39's shipped CE-1, so the v40 build trap (a
checkpoint's own config silently reverting environment fixes) cannot apply.

CE-1: `t167llm2x-ep1` → **`t177vol-ep1`** (`volceiling`: Stage A at 2.18× volume,
champion mix held).

**It is the first archive this project has ever run in the grader image before
submitting it.** v37 scored 0.3611536 precisely because a container shipped unrun,
and this file records "Neither archive has been run in the grader image" for v39
and v40. v41 was run in `twirlz/ecup26-matching:1.0` via `docker --gpus all` on a
local RTX 3050. The full path exercised, no exception and no fallback:

```
ce: model loaded ... dtype=torch.bfloat16
ce: pairs sorted by TOKEN length (mean 333 tok, max 1024)
CE-2: cascade to top 3300/11000 (30%) of the CE-1 ranking
CE ensemble: cascaded partner, w_ce=0.7
wrote 11000 rows (blend w=0.1)
done in 251s (stage budget 360s)
```

**Paired container A/B on the 11,000-pair labelled repro set** (same image, same
data, same environment; `tools/score_graderepro.py`):

| container | MACRO AP | rows matched |
|---|---|---|
| v39 | 0.44813 | 11082/11082 |
| **v41** | **0.45232** | 11082/11082 |
| | **+0.00419** | |

**Three instruments, none negative, and they disagree on size:** container
+0.00419, E_mix +0.00836 (19/20), E_real +0.00024 (9/20). The container number is
the closest thing to the board we have ever had before spending a slot, but it
rests on ~1,183 positives across 20 categories and is noisy. **`transfergap`
forbids projecting any of them onto the board** — no board number is predicted
here in either direction.

**Rope deliberately left as v39 ships it** (`sliding_attention.rope_theta` 160000).
`ropeconfound` shows the value is wrong and live (87/96 rank swaps), but v40 ran
the corrected 10000 and lost 0.00087. Changing it would make this two variables
and would move the one with a losing receipt.

**Expectation, stated plainly:** t177vol is only a **1.09×** volume increment over
the incumbent, and `volceiling` measured the volume axis FLAT beyond 2.0×. This is
not expected to be a large move. The case for the slot is the asymmetry — private
score comes from *selected* submissions, so with v39 held as a selection the
downside is bounded — not the point estimate.

**Slots are the operator's. Agents never submit.**

---

## v41 — SUBMITTED 2026-08-30 — **0.5219365998823058 — REJECT**

**−0.0011088882 against v39.** The one-variable CE-1 swap to `t177vol-ep1`
costs about a board point. **v39 (0.5230454881) remains champion.**

`t177vol-ep1` is dead as a CE-1 replacement. `volceiling` as a *training* result
is untouched — it cleared its registered bar (+0.01314 E_real at matched ep0,
17/20) and that reading stands. Deployment and attribution are separate questions
(`incumbentmoved`).

### Three instruments predicted this, and the best-looking one was wrong

| instrument | predicted | board | |
|---|---|---|---|
| grader-image container AP (11k labelled pairs) | **+0.00419** | −0.00111 | **sign wrong** |
| E_mix, 19/20 categories | **+0.00836** | −0.00111 | **sign wrong** |
| E_real, CI [−0.00667, +0.00747], 9/20 | +0.00024 | −0.00111 | **right; truth inside the CI** |

I rated the container A/B highest — "the closest thing to the board we have ever
had before spending a slot" — and it failed on its first test. **Being the real
container, the real image and the real metric does not fix sample size:** 11,000
pairs at a 0.1076 positive rate is ~1,183 positives over 20 categories, ~59
positives per category-AP. I wrote "it is NOISY and is not a board prediction" in
the same breath as leaning on it.

**E_real was right and I discounted it.** It said "no difference"; the truth was
−0.0011, inside its interval. Its flatness was information, not weakness. Treating
2-of-3 as agreement was wrong when the dissenter was the honest one.

**E_mix's board-anchor status is damaged.** `emixboardanchored` records ~0.65
transfer; here its strongest possible signal (19/20 categories) implied −0.13.
Category unanimity is not evidence of board direction.

**`transfergap` fourth receipt, second sign flip:** ratios now +1.380, +0.078,
−0.284, −0.265. Local→board has failed on sign in 2 of 4 and on magnitude in 4 of 4.

**What survives:** the grader-image harness itself. It proved the container runs
end-to-end (CE-2 cascade, `w_ce=0.7`, 251s of 360s) — a real guard against the v37
class of failure (0.3611536 from an unrun container). Demote its AP number, keep
the harness.

**Consequence for the final slot:** `poolfull` reads on **E_real** at matched ep0,
the instrument this receipt just vindicated. Hold that bar strictly — a +0.008
E_mix and a +0.004 container both became −0.001 on the board.
