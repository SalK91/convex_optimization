# 🏗️ System Design Deep Dive: AI for KYC/AML

This guide maps the functional KYC workflow to specific AI models, Data Infrastructure, and Scalability patterns. It is designed for technical system design discussions.

---

## 1. High-Level System Architecture
The "Golden" Tech Stack for Modern Fintech:

* ⚡ Ingestion: `Kafka` (Streaming events), `gRPC` (Low-latency Internal APIs).
* 💾 Storage:
    * Documents: `AWS S3` (Immutable storage).
    * Relations: `Neo4j` or `TigerGraph` (Graph DB).
    * Vectors: `Pinecone`, `Milvus`, or `Elasticsearch` (Vector Search).
    * Features: `Redis` (Online Feature Store).
* 🧮 Compute: `Kubernetes` (Orchestration), `Apache Flink` (Stateful stream processing).
* 🤖 AI Serving: `Triton Inference Server` or `TorchServe`.

---

## 2. Step-by-Step Breakdown: Models & Infrastructure

### Steps 1 & 2: Data Collection & Verification (The "Vision" Layer)
Goal: Convert unstructured physical atoms (ID card, Face) into structured digital bits.

* 🧠 AI Models:
    * OCR (Optical Character Recognition): Uses Transformer-based Vision models (e.g., TrOCR) rather than traditional Tesseract. specialized for reading MRZ (Machine Readable Zone) codes.
    * Document Forensics: A CNN (Convolutional Neural Network) trained to detect pixel anomalies (manipulated fonts, mismatched noise patterns) indicative of Photoshop.
    * Liveness Detection:
        * *Passive:* Single-frame analysis of texture (skin vs. mask) and depth maps.
        * *Active:* Landmarks tracking (`dlib`/`MediaPipe`) to verify movement instructions.
* ⚙️ Infrastructure:
    * GPU Nodes: Requires `AWS G4/G5` instances.
    * Serving: `Triton Inference Server` to batch requests and keep latency <200ms.

### Step 3: Screening (The "Language" Layer)
Goal: Find hidden risks in unstructured text (News, Sanctions) using semantic understanding.

* 🧠 AI Models:
    * NER (Named Entity Recognition): BERT-based models extract entities (Person, Location, Crime Type) from unstructured news text.
    * Semantic Search (Vector Embeddings): Converts names into vector space to solve fuzzy matching.
        * *Example:* "Osama Bin Laden" and "Usama bin Ladin" have high cosine similarity vectors, even if strings don't match.
    * Sentiment Analysis: Filters out positive news to focus on adverse media.
* ⚙️ Infrastructure:
    * Vector Database: `Pinecone` or `Milvus` for Approximate Nearest Neighbor (ANN) search on millions of vectors.

### Step 4: Entity Resolution (The "Graph" Layer)
Goal: Unmask the UBO (Ultimate Beneficial Owner) and detect shell company networks.

* 🧠 AI Models:
    * GNNs (Graph Neural Networks): Models like GraphSAGE or GCN.
    * Function: Propagates risk through connections. If *Company A* shares a director/address with *Company B* (Fraud), *Company A* gets a high-risk embedding.
* ⚙️ Infrastructure:
    * Graph Database: `Neo4j` or `TigerGraph`. Optimized for multi-hop traversal which SQL cannot handle efficiently.

### Step 5: Risk Scoring (The "Decision" Layer)
Goal: Synthesize all signals into a single probability score ($P(Fraud)$).

* 🧠 AI Models:
    * XGBoost / LightGBM: The standard for tabular risk data (fast, interpretable).
    * Autoencoders: Unsupervised Deep Learning for Anomaly Detection. Flags users with high reconstruction error (unusual behavior) even if no specific rule is broken.
* ⚙️ Infrastructure: The Feature Store
    * Tool: `Feast` or `Tecton`.
    * Online Store (`Redis`): Serves pre-computed features (e.g., "transaction_count_7d") to the model in <10ms.
    * Offline Store (`Snowflake`): Ensures Training-Serving Consistency (preventing skew).

### Step 6: Feedback Loop (Active Learning)
Goal: Improve models using human analyst decisions.

* Workflow: Analyst marks "False Positive" $\rightarrow$ Data labeled $\rightarrow$ Retrain Model.
* Deployment: New models run in Shadow Mode (Canary Deployment) alongside the live model to verify performance before switching.

---

## 3. ✏️ The "Whiteboard" Flow (For Interviews)

If asked to draw the system, visualize this linear pipeline:

1.  Ingestion: Mobile App $\rightarrow$ API Gateway $\rightarrow$ Kafka Topic.
2.  Parallel Processing:
    * *Consumer A:* Computer Vision Service (Triton) $\rightarrow$ DB.
    * *Consumer B:* Screening Service (Vector DB search).
    * *Consumer C:* Graph Service (Update Neo4j $\rightarrow$ Run GNN).
3.  Feature Assembly: Fetch realtime vectors from Feature Store (Redis).
4.  Scoring: Assemble features $\rightarrow$ XGBoost Model $\rightarrow$ Risk Score (0-100).
5.  Decision:
    * Risk < 10: Auto-Approve.
    * Risk 10-80: Human Queue (BPM Tool).
    * Risk > 80: Auto-Reject.

---

## 4. 💎 Key "Pro" Tips
* Entity Resolution is the bottleneck: Explain that resolving "J. Smith" vs "John Smith" requires complex Logic + Graph theory.
* Explainability (XAI) is non-negotiable: In Fintech, you must use SHAP values to explain *why* a model rejected a user (Regulatory requirement).
* The Feature Store is critical: Emphasize that without a Feature Store, you cannot perform real-time scoring reliably.