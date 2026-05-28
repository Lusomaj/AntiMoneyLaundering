# Production Architecture Guidance: Hybrid AML System

To move from a research prototype to a production-grade system, the following architectural recommendations should be implemented.

## 1. Streaming Architecture (Kafka & Spark)
For near-real-time detection of high-value or suspicious transactions, a streaming pipeline is essential:
-   **Data Ingestion**: Use **Apache Kafka** to ingest transaction streams from core banking systems.
-   **Stream Processing**: Use **Apache Spark Streaming** (or Flink) to:
    -   Perform windowed aggregations (e.g., transaction count per hour) in-memory.
    -   Maintain a "hot" state of account centrality features.
-   **Hybrid Enrichment**: Connect the Spark stream to a **Graph Database** (e.g., Neo4j or Amazon Neptune) to fetch SNA features in sub-second time.

## 2. Distributed Graph Computation
Calculating Betweenness Centrality on millions of nodes is O(V*E) and computationally expensive.
-   **Approximation**: Use **Apache Spark GraphX** or **GraphFrames** to compute SNA measures across a cluster.
-   **Incremental Updates**: Instead of recomputing the full graph, use incremental graph algorithms that update centrality based only on new edges.

## 3. Microservice Scalability
-   **Containers**: Use the provided `Dockerfile` to deploy instances to **Kubernetes (K8s)**.
-   **Horizontal Scaling**: Use K8s Horizontal Pod Autoscaler (HPA) to scale the `/predict` API based on CPU/Memory usage.
-   **Load Balancing**: Use an Ingress Controller to distribute traffic across nodes.

## 4. Explainability & Logging
-   **Asynchronous SHAP**: SHAP calculations can be slow. In production, run SHAP asynchronously and store the results in a "Reasoning Store" (e.g., Elasticsearch) for auditors.
-   **Audit Trails**: Log every flagged transaction, the model version, and the top 3 SHAP features for compliance with AML regulations.

## 5. Scalability Plan
| Component | Scaling Strategy | Tooling |
| :--- | :--- | :--- |
| **Ingestion** | Vertical & Horizontal Partitioning | Apache Kafka |
| **Logic** | Data Parallelism | Apache Spark |
| **Graph** | Distributed Storage | Neo4j / AWS Neptune |
| **Inference** | Microservice Replication | Docker + K8s |

