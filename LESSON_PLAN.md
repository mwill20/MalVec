# MalVec Lesson Plan (Skeleton)

## 🎯 Purpose of This Document

This lesson plan outlines what we'll teach AFTER the tool is built. It guides implementation by showing:
1. What concepts need to be demonstrable
2. What components need clear boundaries
3. What metrics need to be measurable
4. What decisions need documentation

**Note:** Actual lessons are generated AFTER Phase 10 of the build.

---

## 📚 Novice Track (~16 Lessons)

**Target Audience:** Aspiring malware analysts, SOC analysts, security students

**Prerequisites:** Basic Python, command line comfort, motivation to learn

**Learning Outcome:** Foundation in malware analysis + ability to use MalVec + entry-level interview prep

---

### Module 1: Malware Detection Fundamentals (3 Lessons)

**Lesson 01: Why Antivirus Fails - The Signature Arms Race**
- **Concept:** Signature-based detection
- **Problem:** Polymorphic malware
- **Domain:** Evolution of malware detection
- **Tool Connection:** Why MalVec exists
- **Hands-On:** Compare signature vs embedding detection
- **Interview Prep:** "What are the limitations of signature-based AV?"

**Lesson 02: Machine Learning Basics - Teaching Computers to Recognize Patterns**
- **Concept:** Supervised learning, features, labels
- **Problem:** How to generalize from examples
- **Domain:** ML in cybersecurity
- **Tool Connection:** MalVec's classification approach
- **Hands-On:** Train a simple classifier
- **Interview Prep:** "How does machine learning improve detection?"

**Lesson 03: Embeddings Explained - Turning Code Into Geometry**
- **Concept:** Vector representations, semantic similarity
- **Problem:** How to measure "similar" for code
- **Domain:** Representation learning
- **Tool Connection:** MalVec's core innovation
- **Hands-On:** Generate your first embedding
- **Interview Prep:** "What are embeddings and why do they work?"

---

### Module 2: MalVec Tool Mastery (7 Lessons)

**Lesson 04: Architecture Tour - How the Pipeline Works**
- **Concept:** System architecture, data flow
- **Components:** All 6 MalVec components
- **Domain:** Security system design
- **Hands-On:** Trace a sample through the pipeline
- **Interview Prep:** "Describe a malware detection system architecture"

**Lesson 05: Feature Extraction - What Makes Malware Different**
- **Concept:** Static analysis, PE/ELF structure
- **Component:** `extractor.py`
- **Domain:** Binary file formats
- **Hands-On:** Extract features from a sample
- **Interview Prep:** "What static features indicate maliciousness?"

**Lesson 06: Embedding Generation - Creating Vector Fingerprints**
- **Concept:** Feature → text → embedding
- **Component:** `embedder.py`
- **Domain:** Text embeddings for code
- **Hands-On:** Generate and inspect embeddings
- **Interview Prep:** "How do you create embeddings from binaries?"

**Lesson 07: Similarity Search - Finding Malware Neighbors**
- **Concept:** Cosine similarity, K-NN search
- **Component:** `store.py`
- **Domain:** Vector databases
- **Hands-On:** Find similar samples in vector space
- **Interview Prep:** "How does similarity search work?"

**Lesson 08: Classification - Making the Malicious Call**
- **Concept:** K-NN voting, confidence scoring
- **Component:** `classifier.py`
- **Domain:** Decision thresholds, false positives
- **Hands-On:** Classify an unknown sample
- **Interview Prep:** "How do you balance false positives vs negatives?"

**Lesson 09: Reading Results - Confidence, Clusters, False Positives**
- **Concept:** Interpreting ML outputs
- **Domain:** Operational decision-making
- **Hands-On:** Analyze classification results
- **Interview Prep:** "When should you trust ML predictions?"

**Lesson 10: Running MalVec - Complete End-to-End Lab**
- **Concept:** Full workflow
- **Components:** All scripts
- **Hands-On:** Train DB, classify samples, visualize clusters
- **Interview Prep:** "Walk me through a malware analysis workflow"

---

### Module 3: Domain Knowledge (3 Lessons)

**Lesson 11: Malware Families - Understanding the Taxonomy**
- **Concept:** APT groups, ransomware families, trojans
- **Domain:** Threat landscape
- **Tool Connection:** Why clusters map to families
- **Hands-On:** Identify family from cluster
- **Interview Prep:** "Describe major malware families"

**Lesson 12: Evasion Techniques - How Attackers Hide**
- **Concept:** Obfuscation, packing, polymorphism
- **Domain:** Adversarial techniques
- **Tool Connection:** What MalVec catches vs misses
- **Hands-On:** Test detection on packed malware
- **Interview Prep:** "How do attackers evade detection?"

**Lesson 13: Detection Trade-offs - Speed vs Accuracy vs False Positives**
- **Concept:** ROC curves, precision-recall
- **Domain:** Operational constraints
- **Tool Connection:** MalVec's performance characteristics
- **Hands-On:** Tune confidence threshold
- **Interview Prep:** "How do you evaluate a detection system?"

---

### Module 4: Career Preparation (3 Lessons)

**Lesson 14: Building Your Portfolio - What to Show Employers**
- **Concept:** Portfolio projects
- **Domain:** Career development
- **Hands-On:** Create GitHub showcase
- **Interview Prep:** "Tell me about a security project you built"

**Lesson 15: Interview Mastery - Malware Analyst Questions**
- **Concept:** Technical interviewing
- **Domain:** Career skills
- **Practice:** Mock interview scenarios
- **Topics:** All previous lessons synthesized

**Lesson 16: Next Steps - Tools, Certs, Communities**
- **Concept:** Continuous learning
- **Domain:** Industry resources
- **Resources:** VirusTotal, YARA, Cuckoo, communities
- **Career Paths:** SOC → Threat Intel → Malware RE

---

## 🏆 Professional Track (~22 Lessons)

**Target Audience:** Engineers deploying detection systems to production

**Prerequisites:** Novice track complete OR equivalent malware analysis foundation

**Learning Outcome:** Production implementation expertise + system design skills + staff+ interview prep

---

### Module 1: Architecture & Design (5 Lessons)

**Lesson 01: Embedding Model Selection - Sentence-Transformers vs Custom**
- **Decision:** Which embedding model for which use case
- **Trade-offs:** Speed, accuracy, domain specificity
- **Production:** Model serving infrastructure
- **Hands-On:** Benchmark multiple models
- **Interview Prep:** "How would you choose an embedding model?"

**Lesson 02: Vector Database Trade-offs - FAISS vs Milvus vs pgvector**
- **Decision:** Which vector DB for production
- **Trade-offs:** Performance, scalability, ops complexity
- **Production:** HA deployment patterns
- **Hands-On:** Compare DB performance
- **Interview Prep:** "Design a scalable similarity search system"

**Lesson 03: Index Optimization - Flat vs IVF vs HNSW**
- **Decision:** Index type for latency/recall requirements
- **Trade-offs:** Memory vs speed vs accuracy
- **Production:** Index tuning methodology
- **Hands-On:** Benchmark index types at scale
- **Interview Prep:** "Optimize for <500ms p99 latency with 95% recall"

**Lesson 04: Sharding Strategy - Scaling to Millions of Samples**
- **Decision:** How to partition vector space
- **Trade-offs:** Query latency vs operational complexity
- **Production:** Cross-shard consistency
- **Hands-On:** Implement sharding prototype
- **Interview Prep:** "Scale this to 100M samples"

**Lesson 05: Model Versioning - Managing Embedding Updates**
- **Decision:** How to update embeddings without downtime
- **Trade-offs:** Consistency vs availability
- **Production:** Blue-green deployment for models
- **Hands-On:** Zero-downtime model swap
- **Interview Prep:** "How do you version ML models in production?"

---

### Module 2: Adversarial Robustness (4 Lessons)

**Lesson 06: Polymorphic Malware - Cluster Stability Analysis**
- **Challenge:** Variants drifting in vector space
- **Solution:** Cluster monitoring, adaptive thresholds
- **Production:** Drift detection pipelines
- **Hands-On:** Test stability across mutations
- **Interview Prep:** "How do you detect model drift?"

**Lesson 07: Embedding Space Poisoning - Defense Against Training Attacks**
- **Threat:** Adversarial samples in training data
- **Defense:** Anomaly detection, curated datasets
- **Production:** Data validation pipelines
- **Hands-On:** Detect poisoned embeddings
- **Interview Prep:** "How would you defend against data poisoning?"

**Lesson 08: Evasion Detection - Recognizing Adversarial Samples**
- **Challenge:** Samples designed to fool embeddings
- **Solution:** Ensemble methods, uncertainty quantification
- **Production:** Multi-model voting
- **Hands-On:** Build ensemble classifier
- **Interview Prep:** "Detect adversarially crafted inputs"

**Lesson 09: Ensemble Methods - Multi-Model Defense**
- **Approach:** Combine multiple embedding models
- **Trade-offs:** Accuracy vs cost/latency
- **Production:** Parallel inference infrastructure
- **Hands-On:** Deploy 3-model ensemble
- **Interview Prep:** "When should you use ensembles?"

---

### Module 3: Production Engineering (6 Lessons)

**Lesson 10: Batch Processing - Throughput Optimization**
- **Goal:** Process 100K samples/day
- **Optimization:** Parallelization, batching, caching
- **Production:** Queue-based architecture
- **Hands-On:** Optimize from 1K to 10K samples/hour
- **Interview Prep:** "Scale batch processing"

**Lesson 11: Real-Time Detection - Sub-Second Classification**
- **Goal:** <500ms end-to-end latency
- **Optimization:** Pre-computed embeddings, hot caches
- **Production:** Low-latency serving infrastructure
- **Hands-On:** Achieve <200ms p99 latency
- **Interview Prep:** "Design a real-time detection API"

**Lesson 12: GPU Acceleration - CUDA Patterns for Embeddings**
- **Optimization:** Batch embedding generation on GPU
- **Trade-offs:** Cost vs speed
- **Production:** GPU resource management
- **Hands-On:** 10x speedup with CUDA
- **Interview Prep:** "When is GPU acceleration worth it?"

**Lesson 13: Cost Optimization - Cloud vs On-Prem Trade-offs**
- **Analysis:** TCO for different deployment models
- **Trade-offs:** Capex vs opex, control vs convenience
- **Production:** Multi-cloud strategies
- **Hands-On:** Calculate breakeven point
- **Interview Prep:** "Optimize for $X/month budget"

**Lesson 14: Monitoring & Alerting - Operational Observability**
- **Metrics:** Latency, throughput, accuracy, errors
- **Alerting:** SLO-based alerts, runbooks
- **Production:** Full observability stack
- **Hands-On:** Set up Prometheus + Grafana
- **Interview Prep:** "Design monitoring for this system"

**Lesson 15: Model Drift - Detecting and Retraining**
- **Challenge:** Accuracy degradation over time
- **Solution:** Continuous evaluation, automated retraining
- **Production:** MLOps pipelines
- **Hands-On:** Build drift detection system
- **Interview Prep:** "How do you maintain ML systems?"

---

### Module 4: Integration Patterns (4 Lessons)

**Lesson 16: SIEM Integration - FortiSIEM, Splunk, Elastic**
- **Integration:** Push detections to SIEM
- **Protocol:** Syslog, REST API, file-based
- **Production:** Reliable event delivery
- **Hands-On:** Send alerts to Splunk
- **Interview Prep:** "Integrate with existing SOC tools"

**Lesson 17: EDR Augmentation - Working With Existing Tools**
- **Pattern:** ML as second opinion for EDR
- **Integration:** Webhook callbacks, file sharing
- **Production:** Async processing pipelines
- **Hands-On:** Augment FortiEDR with MalVec
- **Interview Prep:** "Enhance existing security stack with ML"

**Lesson 18: Threat Intel Feeds - Enriching Detections**
- **Enhancement:** Map samples to IOCs, ATT&CK
- **Integration:** Pull from MISP, OTX, VirusTotal
- **Production:** Feed ingestion pipelines
- **Hands-On:** Enrich with threat intel
- **Interview Prep:** "Leverage threat intelligence"

**Lesson 19: API Design - Building a Detection Service**
- **Design:** REST API for classification
- **Requirements:** Auth, rate limiting, versioning
- **Production:** API gateway, load balancing
- **Hands-On:** Deploy production API
- **Interview Prep:** "Design a detection microservice"

---

### Module 5: Research & Innovation (3 Lessons)

**Lesson 20: State of the Art - Current Academic Research**
- **Papers:** Latest embedding techniques
- **Trends:** Multimodal learning, LLMs for code
- **Evaluation:** What's hype vs real
- **Hands-On:** Replicate a paper
- **Interview Prep:** "What's the future of malware detection?"

**Lesson 21: Future Directions - Multimodal Embeddings, LLMs**
- **Innovation:** Combine static + dynamic features
- **Approach:** Multi-encoder architectures
- **Potential:** Code LLMs for semantic understanding
- **Hands-On:** Prototype multimodal system
- **Interview Prep:** "Where is this field heading?"

**Lesson 22: Contributing - How to Extend MalVec**
- **Open Source:** Adding features, fixing bugs
- **Research:** Publishing improvements
- **Community:** Building around the project
- **Hands-On:** Submit your first PR
- **Career:** Building reputation

---

## 🎯 Lesson Development Guidelines

**For EVERY lesson:**

1. **Start with the problem** - Why does this matter?
2. **Show the tool solution** - How does MalVec handle this?
3. **Teach the domain concept** - What's the broader principle?
4. **Provide hands-on practice** - Let them DO it
5. **Measure success** - How do they know it works?
6. **Prepare for interviews** - What questions will they face?

**Novice-specific:**
- Use everyday analogies
- Build confidence incrementally
- Celebrate small wins
- Connect to career goals

**Professional-specific:**
- Focus on trade-offs
- Show production war stories
- Discuss failure modes
- Enable system design thinking

---

## 📊 Success Metrics for Lessons

**Novice Track:**
- Can explain malware detection to a colleague
- Can run MalVec end-to-end
- Can answer entry-level interview questions
- Understands when to use vs not use embeddings

**Professional Track:**
- Can design production detection system
- Can optimize for specific constraints
- Can debug complex failures
- Can answer staff+ interview questions

---

## 🔑 Key Insight

This lesson plan guides implementation by showing what needs to be:
- **Demonstrable** (hands-on labs need working code)
- **Explainable** (architecture must be teachable)
- **Measurable** (metrics must be observable)
- **Documentable** (decisions must be recorded)

**Build the tool to enable teaching. Teach the tool to reinforce learning.**
