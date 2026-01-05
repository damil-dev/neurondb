<img src="assets/ai-with-database-on-prem/header.svg" alt="Running AI on premises with Postgres header" />

<h1>Running AI on premises with Postgres</h1>

<p><a href="https://github.com/neurondb-ai/neurondb">View on GitHub</a> | <a href="https://github.com/neurondb-ai/neurondb/releases">Download latest release</a> | <a href="https://neurondb.ai/docs">Documentation</a></p>

<p>Run your database, embeddings, indexes, and retrieval inside your network. Do this when you need data control, stable latency, and clear security boundaries. Keep the system small at first, then add parts as your load grows.</p>

<h2>Decide if you should run on premises</h2>

<p>Pick on premises when you must control where data lives. Pick it when you must keep traffic private. Pick it when you must hit a strict latency target. Pick it when costs grow with API calls and egress. If you need fast setup for a small pilot, start in cloud, then move the data plane later.</p>

<ul>
  <li>Compliance: HIPAA, GDPR, PCI, residency rules, audit rules</li>
  <li>Security: private networks, strict access, limited outbound traffic</li>
  <li>Latency: stable p95 and p99, fewer hops</li>
  <li>Cost: high volume usage where per call fees add up</li>
  <li>Control: standard Postgres and a clear ops surface</li>
</ul>

<h2>Cloud vs on premises, quick view</h2>

<img src="assets/ai-with-database-on-prem/quickreview.png" alt="On premises vs cloud AI comparison" />

<p>Watch your data movement. In many systems you fetch documents in one place, run embeddings in another, and run vector search in a third place. Each hop adds latency and failure modes. If you keep these steps inside one network, you cut variance and you debug faster.</p>

<h2>Architecture overview</h2>

<img src="assets/ai-with-database-on-prem/archtect.png" alt="On premises AI architecture overview" />

<p>Keep the data plane local. Store documents and metadata in Postgres. Store embeddings next to the rows they describe. Build vector indexes in the same database. Run retrieval queries over private links. Expose results through your app services.</p>

<p>Keep three paths clear. Ingest is write heavy. Retrieval is read heavy. Admin work is rare but sensitive. Split these paths by network rules and by roles.</p>

<img src="assets/ai-with-database-on-prem/diagram-on-prem-architecture.svg" alt="On premises architecture diagram" />

<p>Put ingestion on a schedule. Batch it. Keep queries stable. Do not let ad hoc scripts write to the main database. Use a queue or a worker process. Record each run.</p>

<h2>What you run</h2>

<p>Keep the component list short. Assign an owner to each part. If you cannot name the host and the pager, you are not done.</p>

<ul>
  <li>Postgres with NeuronDB for storage, embeddings, indexes, retrieval</li>
  <li>Ingestion workers for cleaning, chunking, and loads</li>
  <li>Embedding execution on CPU or GPU, batch jobs, steady throughput</li>
  <li>App services that call Postgres and return citations</li>
  <li>Monitoring for latency, load, pool use, lag, backups</li>
</ul>

<h2>Deployment patterns</h2>

<p>Start simple. Prove retrieval quality. Prove latency. Add resilience only when you need it. Keep changes small so you can reverse them.</p>

<h3>Single server</h3>

<img src="assets/ai-with-database-on-prem/diagram-single-server.svg" alt="Single server deployment diagram" />

<p>Use this for your first release. You get one host to secure. You get one Postgres instance to tune. You get clear failure handling. Add backups and dashboards before you add more servers.</p>

<pre><code class="language-sql">CREATE EXTENSION neurondb;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
    embedding vector(384)
);

INSERT INTO documents (content, embedding)
VALUES ('Document content', embed_text('Document content', 'sentence-transformers/all-MiniLM-L6-v2'));

SELECT
  content
FROM documents
ORDER BY embedding &lt;=&gt; embed_text('query', 'sentence-transformers/all-MiniLM-L6-v2')
LIMIT 10;</code></pre>

<p>Add filters early. It keeps results stable. It keeps cost stable. It keeps latency stable.</p>

<h3>Data model and chunking</h3>

<p>Store chunks, not whole files. Keep the original document id. Store offsets. Store a version. Keep chunk size stable. Start with 300 to 800 tokens per chunk. Start with 50 to 150 token overlap. Measure answer quality. Then change one variable.</p>

<pre><code class="language-sql">CREATE TABLE doc_chunks (
  doc_id BIGINT NOT NULL,
  chunk_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  source TEXT,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding vector(384)
);

CREATE INDEX doc_chunks_tenant_doc_idx
  ON doc_chunks (tenant_id, doc_id);</code></pre>

<p>Track a content hash. It lets you skip re embedding on retries. It lets you detect duplicates. Use a text hash or a stable id from your upstream system.</p>

<h3>Hybrid search with metadata and vectors</h3>

<p>Filter with metadata, then rank by vector distance. Use this per tenant. Use it per source. Use it per time window.</p>

<pre><code class="language-sql">ALTER TABLE documents
  ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default',
  ADD COLUMN source TEXT,
  ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT now();

SELECT id, content
FROM documents
WHERE tenant_id = 'acme'
  AND (source IS NULL OR source &lt;&gt; 'spam')
ORDER BY embedding &lt;=&gt; embed_text('query', 'sentence-transformers/all-MiniLM-L6-v2')
LIMIT 10;</code></pre>

<h3>Ingestion workflow</h3>

<p>Use one workflow. Keep it the same in development, test, and live. Run it in batches. Track each run. Start with these steps.</p>

<ol>
  <li>Fetch raw documents</li>
  <li>Normalize text, strip boilerplate</li>
  <li>Split into chunks, keep offsets</li>
  <li>Insert rows without embeddings</li>
  <li>Compute embeddings in batches of 32 to 256</li>
  <li>Update embeddings</li>
  <li>Build or refresh indexes</li>
  <li>Run a sample query set, record p95</li>
</ol>

<p>Set one target. Ingest 100k chunks in under 30 minutes. Then tune. If you cannot hit that target, reduce batch size, increase worker count, or move embedding execution to GPU.</p>

<h3>Primary and replicas</h3>

<img src="assets/ai-with-database-on-prem/diagram-multi-server-cluster.svg" alt="Multi server cluster diagram" />

<p>Use this when you need uptime and read scale. Keep writes on the primary. Send retrieval reads to replicas. Use a pooler. Track replication lag. Set a rule for stale reads.</p>

<pre><code class="language-sql">CREATE EXTENSION neurondb;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(384)
);

SELECT hnsw_create_index('documents', 'embedding', 'documents_embedding_hnsw', 16, 200);</code></pre>

<h2>Connection pooling</h2>

<p>Use a pooler for app traffic. Set a hard limit on connections. Keep idle connections low. Track pool saturation. Start with 20 to 50 connections per app node. Raise it only after you measure.</p>

<p>Keep one rule. Do not let each app pod open hundreds of direct connections to Postgres. It will fail under load.</p>

<h2>Indexing and maintenance</h2>

<p>Indexes drift. Stats drift. Tables bloat. Plan for it. Batch ingestion. Refresh stats. Watch index size. Watch vacuum behavior.</p>

<pre><code class="language-sql">ANALYZE documents;</code></pre>

<p>Check query plans. Do it before and after each major ingest. You want an index scan for retrieval queries. You do not want a full table scan.</p>

<pre><code class="language-sql">EXPLAIN (ANALYZE, BUFFERS)
SELECT id, content
FROM documents
ORDER BY embedding &lt;=&gt; embed_text('query', 'sentence-transformers/all-MiniLM-L6-v2')
LIMIT 10;</code></pre>

<h2>Replication checks</h2>

<p>Track lag. Track replay delay. Set an alert. Use a number. Start with 5 seconds for p95 lag. Use reads from the primary if lag exceeds your limit.</p>

<pre><code class="language-sql">SELECT
    application_name,
    state,
  write_lag,
  flush_lag,
  replay_lag
FROM pg_stat_replication;</code></pre>

<h2>Sizing</h2>

<p>Start with three numbers. Vector count. Embedding dimension. Peak reads per second. Then add headroom. For raw float storage use vectors times dims times 4 bytes. Ten million vectors at 384 dims is about 15.4 GB for floats. Plan for more once you add row overhead and indexes.</p>

<p>Use a simple table. It keeps planning honest.</p>

<ul>
  <li>1 million vectors at 384 dims, about 1.5 GB floats</li>
  <li>10 million vectors at 384 dims, about 15.4 GB floats</li>
  <li>10 million vectors at 768 dims, about 30.7 GB floats</li>
</ul>

<h2>Security</h2>

<img src="assets/ai-with-database-on-prem/diagram-security-architecture.svg" alt="Security architecture diagram" />

<p>Keep the database private. Restrict inbound. Restrict outbound. Limit roles. Log access. Keep backups protected.</p>

<ul>
  <li>Put the database in private subnets</li>
  <li>Use a bastion or VPN for admin access</li>
  <li>Use TLS on internal links</li>
  <li>Use disk encryption at rest</li>
  <li>Use least privilege roles for apps</li>
</ul>

<h3>Roles</h3>

<p>Create one app role per service. Grant only what it needs. Avoid superuser. Avoid owner roles in apps.</p>

<pre><code class="language-sql">CREATE ROLE app_reader NOINHERIT;
GRANT CONNECT ON DATABASE postgres TO app_reader;
GRANT USAGE ON SCHEMA public TO app_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO app_reader;</code></pre>

<h2>Performance</h2>

<p>Start with measurements. Confirm index use. Batch embeddings. Filter early. Keep result sets small. Track pool saturation.</p>

<pre><code class="language-sql">CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SELECT 
    calls,
  ROUND(mean_exec_time::numeric, 2) AS mean_ms,
  ROUND(max_exec_time::numeric, 2) AS max_ms,
  LEFT(query, 120) AS query_preview
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;</code></pre>

<p>Pick two numbers. Retrieval p95. Ingest throughput. Track them daily. Change one thing at a time.</p>

<h2>Backups and recovery</h2>

<p>Set RPO and RTO. Run restore drills. Write a steps document. Test failover in test. Keep the process repeatable.</p>

<p>Run a restore drill each month. Time it. Record it. Fix the slow steps. Keep one target. Restore in under 60 minutes for your core dataset.</p>

<h2>Migration from cloud</h2>

<img src="assets/ai-with-database-on-prem/diagram-migration.svg" alt="Cloud to on premises migration diagram" />

<p>Move the data plane first. Export docs and embeddings. Import into Postgres. Rebuild indexes. Mirror traffic. Compare answers and latency. Cut over with a rollback plan.</p>

<pre><code class="language-sql">CREATE TABLE documents (
  id BIGINT PRIMARY KEY,
    content TEXT,
  embedding vector(384)
);

SELECT hnsw_create_index('documents', 'embedding', 'documents_embedding_hnsw', 16, 200);</code></pre>

<h2>Cost model</h2>

<p>Use break even months. Use capex divided by cloud monthly minus on premises monthly. Include staff time, power, support, and depreciation. Include egress and API fees on the cloud side.</p>

<p>Use one example with numbers. Keep it simple.</p>

<ul>
  <li>Capex 120000</li>
  <li>Cloud monthly 18000</li>
  <li>On premises monthly 9000</li>
  <li>Break even months is 120000 divided by 9000, about 13.3</li>
</ul>

<h2>Checklist</h2>

<ol>
  <li>Pick a pattern, single server, cluster, hybrid, edge</li>
  <li>Set targets for p95 latency, QPS, RPO, RTO</li>
  <li>Lock down networks, subnets, firewall, bastion</li>
  <li>Add TLS and disk encryption</li>
  <li>Add a pooler</li>
  <li>Build indexes and check query plans</li>
  <li>Add monitoring and alerts</li>
  <li>Set backups and run a restore drill</li>
</ol>

<h2>Conclusion</h2>

<p>Running AI on premises with Postgres gives you control, stability, and cost savings at scale. Start with a single server. Lock down security. Measure latency and throughput. Add resilience as you grow. Use NeuronDB to keep embeddings, indexes, and retrieval inside your network. Keep it simple. Keep it monitored. Keep it backed up.</p>

<h2>Related resources</h2>

<ul>
  <li><a href="/docs">NeuronDB documentation</a></li>
  <li><a href="/blog/neurondb">NeuronDB overview</a></li>
  <li><a href="/blog/rag-complete-guide">RAG guide</a></li>
  <li><a href="/blog/neurondb-semantic-search-guide">Semantic search guide</a></li>
</ul>