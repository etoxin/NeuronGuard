# NeuronGuard Product Requirements Document

## Document status

- Status: Draft
- Product: NeuronGuard
- Target release: Stabilization release following `0.2.0`
- Primary audience: Maintainers and contributors

## 1. Product summary

NeuronGuard is a compact, CPU-first associative classification engine implemented in Rust and exposed through a high-level Python SDK. It maps sparse sensory tokens to weighted class associations, supports incremental updates, and exposes token-level contributions for interpretable predictions.

The preferred product positioning is:

> An extremely fast, sparse CPU classifier with online updates, compact model storage, and transparent feature contributions.

The project currently combines two related identities:

1. A low-level, cache-aligned event and associative-memory engine.
2. High-level text and tabular classifiers for Python users.

This stabilization effort will make the classifier use case production-ready while clearly separating or documenting lower-level experimental engine components.

## 2. Problem statement

NeuronGuard has a compelling architecture and approachable API, but several correctness, safety, persistence, testing, and documentation gaps prevent its strongest claims from being relied upon in production.

The most important issues are:

- The Rust memory model permits mutable access through shared references and relies on unconditional `Send` and `Sync` implementations.
- Parallel training can silently discard updates when two workers contend for the same sensory neuron.
- Predictions use shared motor potentials across multiple calls, allowing concurrent predictions to interfere with one another.
- Raw model files have no format header, dimension validation, versioning, integrity check, or safe read-only inference mode.
- Tabular models using feature interactions do not restore their complete configuration when loaded.
- Exact machine-unlearning and some performance/cache claims are stronger than the present implementation can guarantee.
- Input validation, lifecycle semantics, automated testing, release metadata, and repository hygiene need strengthening.

## 3. Goals

### 3.1 Primary goals

- Make the Rust core memory-safe under its documented concurrency model.
- Make parallel training lossless, reproducible, and testable.
- Make prediction atomic and safe for concurrent callers.
- Define a validated, versioned model format with reliable round trips.
- Align public claims with measured and demonstrable behavior.
- Establish clear API validation and classifier lifecycle semantics.
- Add continuous integration and substantially expand automated coverage.
- Clarify the product boundary between the classifier SDK and experimental event-engine modules.

### 3.2 Success criteria

- Miri and concurrency-oriented tests find no undefined behavior in supported operations.
- Sequential and parallel training produce equivalent weights for the same ordered input, or any documented nondeterminism is bounded and explicitly tested.
- Concurrent predictions on a shared fitted classifier cannot contaminate one another.
- Saved models reject truncated, incompatible, or corrupt data with clear errors.
- All supported classifier configurations survive save/load round trips without behavioral changes.
- Every externally visible performance claim links to a reproducible benchmark.
- Pull requests run Rust, Python, formatting, linting, and packaging checks automatically.
- Supported Python and platform versions match the wheels actually published.

## 4. Non-goals

- Competing with general-purpose deep-learning frameworks.
- Adding GPU support during the stabilization release.
- Introducing new classifier families before the core safety work is complete.
- Claiming exact regulatory compliance without external legal and technical validation.
- Preserving the existing raw binary model format indefinitely; a one-time migration or compatibility loader is acceptable.
- Guaranteeing identical floating-point or timing behavior across unrelated hardware architectures.

## 5. Target users and use cases

### 5.1 Target users

- Python developers who need fast, interpretable text or tabular classification.
- Edge and embedded developers working within CPU and memory constraints.
- Streaming systems that need inexpensive incremental model updates.
- Researchers exploring sparse associative classification.

### 5.2 Core use cases

- Low-latency text categorization with token-level explanations.
- Bucketed tabular classification with optional feature interactions.
- Batch inference on large collections of sparse records.
- Incremental correction of an existing model.
- Compact, fast model loading for local inference.

## 6. Functional requirements

### 6.1 Rust memory and concurrency model

#### Requirements

- Mutable neuron state must not be exposed through a safe shared reference unless it uses a valid interior-mutability abstraction.
- `Send` and `Sync` may only be implemented when their invariants are documented and enforced.
- Concurrent inference and training behavior must be explicitly defined as supported, serialized, or rejected.
- Immutable inference data should be separated from mutable training state where practical.
- Memory-mapped inference should prefer read-only mappings.
- Unsafe blocks must include concise safety comments identifying the invariant relied upon.

#### Candidate designs

- Use atomics for independently mutable fields.
- Use `UnsafeCell` behind a narrowly scoped abstraction with enforced leases.
- Partition neurons so each worker has exclusive ownership during batch training.
- Maintain a mutable trainer representation and freeze it into an immutable inference representation.

#### Acceptance criteria

- No method returns `&mut ThreadBoundedNeuron` from `&self` through a generally callable interface.
- Miri passes the supported core test suite.
- ThreadSanitizer or an equivalent concurrency test strategy reports no supported-operation races.
- The concurrency contract is documented in the Rust and Python API references.

### 6.2 Lossless parallel training

#### Requirements

- A training update must never be silently discarded because a neuron is temporarily leased.
- Batch training must define ordering and determinism guarantees.
- Contention handling must avoid unbounded starvation.
- Invalid token or class identifiers must return an error rather than silently consuming connection capacity or disappearing.

#### Candidate designs

- Group batch updates by sensory token and process each group exclusively.
- Partition sensory-token ranges across workers.
- Accumulate per-thread deltas and merge them deterministically.
- Use bounded retrying only if its progress and ordering behavior can be guaranteed.

#### Acceptance criteria

- A high-contention test applies the expected number of updates exactly.
- Sequential and parallel training produce equivalent final synapses under the documented ordering rules.
- Repeating the same batch test produces stable results.
- Benchmarks show the chosen design still provides a meaningful batch throughput improvement.

### 6.3 Atomic prediction

#### Requirements

- Reset, accumulation, argmax, and score retrieval must form one logical operation.
- Per-prediction scores must not use shared mutable state.
- Single and batch prediction must use consistent scoring and tie-breaking rules.
- An API should return both the predicted class and scores without recomputing the prediction.

#### Acceptance criteria

- Multiple threads can call prediction on the same fitted model without cross-request contamination.
- Concurrent results match isolated results for every test input.
- The Python SDK no longer implements a prediction through separate reset, predict, and get-potential calls.
- Empty-input and score-tie behavior is documented and tested.

### 6.4 Model persistence

#### Requirements

The model format must include:

- Magic bytes.
- Format version.
- Model type.
- Sensory and motor dimensions.
- Relevant layout constants, including connection capacity.
- Endianness or a canonical byte order.
- Payload length.
- Integrity checksum.
- Classifier configuration required to reproduce tokenization and feature mapping.

Loading must validate the complete file before neuron memory is accessed. Truncated, oversized, corrupt, incompatible, or incorrectly typed models must produce descriptive exceptions.

Runtime synchronization bytes must not be persisted as learned model state. Saving should be atomic, using a temporary file followed by replacement where supported.

#### Tabular configuration

Tabular persistence must include at least:

- `num_classes`
- `num_features`
- `buckets_per_feature`
- `amplify_delta`
- `suppress_delta`
- `baseline_delta`
- `features_min`
- `features_max`
- `use_feature_interactions`
- `interaction_vocab_size`

#### Text configuration

Text persistence must continue to include vocabulary, tokenizer options, class names, vocabulary scoring, bigram configuration, and training deltas.

#### Acceptance criteria

- Text, tabular, and interaction-enabled tabular models produce identical predictions and scores before and after a save/load round trip.
- Read-only model files can be loaded for inference.
- Truncated, corrupt, wrong-version, wrong-dimension, and wrong-model-type fixtures are rejected.
- A migration policy for existing raw `weights.bin` models is documented.

### 6.5 Input validation and lifecycle semantics

#### Requirements

Constructors must validate:

- Positive sensory, feature, vocabulary, bucket, interaction-space, and class counts.
- The relationship between class count and per-neuron connection capacity.
- Delta values that fit the Rust representation.
- `class_names` length when provided.
- Supported vocabulary-scoring values.

Training and evaluation must validate:

- Class labels are in range.
- Feature and column counts match configuration.
- Records contain finite supported values or follow a documented missing-value policy.
- Epochs and class weights are non-negative and within safe bounds.
- Empty datasets and records have defined behavior.

Prediction must reject an unfitted classifier with a clear error.

Repeated `fit()` calls must have explicit semantics. The recommended default is to replace existing learned state. Incremental changes should use `update` or `partial_fit`.

Methods typed as accepting `Iterable` must not require indexing. In particular, tabular update and unlearn operations must work with generators.

#### Acceptance criteria

- Invalid public inputs produce stable Python exception types and actionable messages.
- Repeated fitting behaves as documented and is covered by tests.
- Generator-based update and unlearn operations work correctly.
- NaN, infinity, missing values, empty inputs, and out-of-range labels have documented outcomes.

### 6.6 Connection-capacity behavior

#### Requirements

- The effect of the fixed eight-connection capacity must be visible to users.
- The system must either enforce a compatible maximum class count or support a configurable/dynamic representation.
- Eviction policy and tie-breaking must be deterministic and documented.
- Capacity eviction must be included in explanation and unlearning limitations.

#### Acceptance criteria

- A model cannot silently accept a configuration it cannot faithfully represent.
- Tests cover full-capacity insertion, eviction, ties, saturation, and more than eight classes.

### 6.7 Online learning and unlearning

#### Requirements

- Online learning must clearly state whether vocabulary and feature boundaries remain fixed.
- Unlearning must be described as inverse delta updating unless exact reversibility is proven for the active configuration.
- Exact unlearning must not be claimed after saturation, eviction, dropped updates, or later overlapping updates.
- If exact record deletion becomes a product goal, the model must retain sufficient update provenance or support replay from an auditable training log.

#### Acceptance criteria

- Documentation distinguishes approximate inverse updating from certified deletion.
- Tests cover updates before and after saturation and eviction.
- Public documentation does not make an unqualified GDPR-compliance claim.

### 6.8 Explainability

#### Requirements

- Explanations must use the same tokenization, hashing, scoring, and tie-breaking logic as prediction.
- Negative contributions and collisions must be represented rather than silently omitted.
- Hashed feature explanations must indicate that multiple source features may share a bucket.
- Calling `explain()` must perform prediction only once.

#### Acceptance criteria

- Summed explanation contributions match returned class scores for supported model types.
- Bigram and feature-interaction contributions are included or explicitly marked unavailable.
- Explanation output documents hash collisions and negative weights.

## 7. Performance and benchmarking requirements

### 7.1 Benchmark suite

Create a reproducible benchmark suite covering:

- Training throughput in records and tokens per second.
- Single-record prediction latency at p50, p95, and p99.
- Batch-prediction throughput.
- Peak resident memory.
- Serialized model size.
- Cold and warm model-load latency.
- Contended parallel-training performance.
- Concurrent-inference scalability.
- Accuracy, macro F1, and per-class metrics on fixed public datasets.

Every result must record:

- CPU model and core count.
- Available cache sizes.
- Operating system and architecture.
- Rust, Python, and package versions.
- Dataset version and preprocessing.
- Model configuration and thread count.
- Warm-up and repetition methodology.

### 7.2 Baselines

Where applicable, compare against:

- Multinomial Naive Bayes for sparse text.
- A linear or stochastic-gradient classifier.
- A simple frequency or majority-class baseline.

### 7.3 Documentation rules

- Claims such as “100,000 samples in under three seconds” or “loading in under one millisecond” must link to a reproducible benchmark.
- Cache-line alignment may be described as improving locality, but not as guaranteeing deterministic hardware prefetching.
- The project must not claim an entire multi-megabyte field fits in L1/L2 cache merely because individual neurons are cache-line aligned.
- “Zero overhead,” “completely eliminates,” and similar absolutes should be replaced with measured, scoped statements.

## 8. Python SDK and packaging requirements

### 8.1 Import behavior

- Import behavior without a compiled extension must be intentional.
- If documentation or type checking is supported without the extension, `tokenizer.py` and other high-level modules must not unconditionally import it.
- Otherwise, imports must fail immediately with a clear installation/build message.

### 8.2 Supported versions

- Python metadata must match the versions tested and distributed.
- Either build wheels for every declared supported Python version or use a suitable stable ABI target.
- Rust crate, Python package, and runtime `__version__` values must follow one release version.

### 8.3 Package metadata

Add or verify:

- Description and README metadata.
- License metadata.
- Project URLs.
- Author or maintainer information.
- Appropriate Python, operating-system, and development-status classifiers.
- Typed-package metadata if public type hints are treated as supported API.

### 8.4 Acceptance criteria

- Wheels install and import successfully across the declared matrix.
- Source distributions build in a clean environment.
- Version values are generated from or checked against one source of truth.
- Documentation builds without relying on an undeclared locally compiled artifact.

## 9. Testing and continuous integration

### 9.1 Required test categories

- Rust unit tests for neuron layout and weight behavior.
- Property tests for saturation, eviction, serialization, and token bounds.
- Miri tests for unsafe abstractions.
- High-contention concurrency tests.
- Sequential-versus-parallel equivalence tests.
- Python API unit tests for validation and lifecycle behavior.
- Save/load round-trip and corruption tests.
- Concurrent prediction tests.
- Generator and streaming-input tests.
- Integration tests for every documented getting-started example.
- Wheel installation smoke tests.

### 9.2 Pull-request CI

Every pull request must run:

- `cargo test`
- `cargo fmt --check`
- `cargo clippy` with warnings treated as errors
- Python unit tests
- Python formatting and linting
- Type checking if type hints are part of the supported API
- A development wheel build and import smoke test
- Documentation build and link/example validation

Release CI must run only after the test workflow succeeds and must publish artifacts for the complete supported Python/platform matrix.

### 9.3 Acceptance criteria

- CI runs on pull requests and protected branches, not only version tags.
- Release publication depends on successful test and wheel-validation jobs.
- Critical concurrency and persistence regressions have dedicated tests.

## 10. Documentation and positioning

### 10.1 Required documentation changes

- Use “cache-aligned sparse associative classifier” as the primary technical description.
- Explain what is neuromorphic about the design and what is conventional sparse classification.
- Document fixed vocabulary, feature boundaries, hash collisions, connection capacity, saturation, and eviction.
- Document concurrency guarantees separately for inference, training, and mixed workloads.
- Replace exact machine-unlearning and regulatory claims with scoped behavior and limitations.
- Link all quantitative performance claims to benchmarks.
- Remove or repair references to nonexistent examples, including the Melbourne Cup task.
- Clearly distinguish stable APIs from experimental modules.

### 10.2 API documentation

Every public method must document:

- Valid input ranges.
- Fitted-state requirements.
- Thread-safety behavior.
- Mutation behavior.
- Error types.
- Complexity where meaningful.

## 11. Architecture and repository organization

The current `guard`, `memory`, `queue`, `run`, and `train` modules represent a lower-level or earlier event-engine architecture distinct from the Python classifier implementation.

Choose one of the following:

1. Integrate them into a documented public engine with clear ownership and concurrency invariants.
2. Move them behind an `experimental-engine` Cargo feature.
3. Move them into a separate crate.
4. Remove them if they are no longer part of the product direction.

The default build should contain only maintained, tested functionality needed by the supported product.

## 12. Repository hygiene

### Requirements

- Remove tracked `.DS_Store`, `__pycache__`, and `.pyc` files.
- Add macOS and Python generated artifacts to `.gitignore`.
- Keep generated documentation only if the repository intentionally publishes it that way; otherwise generate it in CI.
- Define contributor commands for setup, build, test, lint, benchmark, and documentation.
- Ensure README example commands refer to existing tasks.

## 13. Delivery plan

### Phase 1: Correctness and safety

- Redesign neuron access and document unsafe invariants.
- Remove unsupported data races.
- Make parallel training lossless.
- Make prediction use request-local potentials.
- Add validation for dimensions, labels, tokens, and deltas.
- Add concurrency and equivalence tests.

Exit criterion: supported training and inference operations have a defined, tested concurrency model.

### Phase 2: Persistence and lifecycle

- Introduce the versioned model format.
- Add strict load validation and read-only inference mappings.
- Fix tabular interaction persistence.
- Define repeated-fit, update, and unlearn behavior.
- Add round-trip, migration, and corrupt-file tests.

Exit criterion: all supported classifier configurations round-trip reliably and incompatible files fail safely.

### Phase 3: Packaging and CI

- Add pull-request CI.
- Reconcile versions and supported Python metadata.
- Build and smoke-test the supported wheel matrix.
- Fix import behavior and documentation builds.
- Clean generated artifacts from the repository.

Exit criterion: a clean checkout can be built, tested, documented, packaged, and installed using documented commands.

### Phase 4: Benchmarks and documentation

- Add reproducible performance and accuracy benchmarks.
- Compare against appropriate baselines.
- Rewrite claims using measured results.
- Clarify algorithm, capacity, concurrency, and unlearning limitations.
- Repair example documentation.

Exit criterion: public claims are reproducible and accurately scoped.

### Phase 5: Product consolidation

- Decide the status of experimental event-engine modules.
- Publish a stable-versus-experimental API policy.
- Add deprecation and model-format compatibility policies.
- Prepare the next stable release.

## 14. Release readiness checklist

- [ ] Rust memory-safety redesign reviewed.
- [ ] Supported concurrency model documented.
- [ ] No silent training-update loss.
- [ ] Concurrent prediction isolation verified.
- [ ] Versioned model format implemented.
- [ ] Corrupt and incompatible models rejected safely.
- [ ] Text and tabular round trips verified.
- [ ] Feature-interaction round trip verified.
- [ ] Public input validation complete.
- [ ] Repeated-fit semantics documented and tested.
- [ ] Unlearning claims corrected.
- [ ] Benchmark suite reproducible.
- [ ] README performance claims linked to results.
- [ ] Pull-request CI required and passing.
- [ ] Supported Python versions match published wheels.
- [ ] Package and crate versions synchronized.
- [ ] Generated artifacts removed from version control.
- [ ] Missing or obsolete example references fixed.
- [ ] Stable and experimental modules clearly separated.

## 15. Risks and open decisions

### 15.1 Risks

- A safe concurrency redesign may reduce headline parallel-training throughput.
- A portable model format may make files slightly larger or loading marginally slower.
- Correcting public claims may make the project sound less dramatic in the short term, but should improve technical credibility and adoption.
- Preserving compatibility with raw memory dumps may complicate the new loader.
- Dynamic connection capacity could compromise the current fixed 64-byte layout.

### 15.2 Open decisions

- Is concurrent training and inference on the same model a supported use case?
- Must parallel training be bit-for-bit deterministic, or only lossless with documented ordering?
- Should the eight-connection representation remain a defining constraint?
- Is the raw event engine part of the stable product or a separate experimental project?
- Should legacy raw weight files receive a migration command or only a compatibility loader?
- Which Python versions and platforms will be officially supported?
- Is exact record deletion a future product requirement requiring provenance, or is approximate inverse updating sufficient?

## 16. Initial verification note

During the review that produced this PRD, the documented Rust and Python checks could not be executed because `cargo` and `uv` were not available on the active shell path and the local `mise.toml` had not been trusted. A fallback Python test could not import the unbuilt native extension. This PRD therefore treats successful clean-environment setup and automated execution as explicit delivery requirements.
