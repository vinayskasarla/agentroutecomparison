# LLM Path Lab

A single-page app that sends the **same questions through several different ways of calling an LLM** and
compares them on **accuracy, cost, latency, confidence, reliability and governance**. It helps you pick
the one route every app in the company should go through.

Pick a model from a dropdown: GPT (OpenAI), Claude (Anthropic), Grok (xAI) or Gemini (Google).

## The routes

| Route | Chain | What it tests |
|---|---|---|
| Direct | Agent → LLM | The baseline: each app calls the provider SDK itself |
| Redis cache | Agent → Redis → LLM | An exact-match response cache |
| Jev only | Agent → Jev | [Jev](https://typesafe.ai), TypeSafe AI's "System One" model, answers as a typed `choice` with calibrated probabilities, picking from candidate answers. It doesn't write text. |
| Jev → LLM cascade | Agent → Jev → (unsure) LLM | Jev answers when its confidence clears a threshold (default 0.8); anything else escalates to the LLM. With a free-form prompt, Jev picks the small or large model tier instead. |
| AI Gateway | Agent → AI Gateway → LLM | Central keys, quotas, guardrails, cost metering and provider failover |
| Agent Runtime → AI Gateway → LLM | Agent Runtime → AI Gateway → LLM | The **enablement platform** and candidate company standard. The runtime adds memory, policy and tracing, plus injected context tokens. The gateway adds exact and similar-question caching, and uses Jev to route each request to the small or large model. |

## What's real and what's modeled

- **Real:** every service hop. The gateway and the runtime are separate HTTP endpoints
  (`/svc/*`) called over real HTTP, so their serialization and processing time is measured, not guessed.
  A configurable network RTT is added to each hop. The PII redaction, the injection guard, rate limiting,
  the Redis cache (real Redis when it's reachable) and failover all run as real code.
- **Jev** is called at `POST https://api.typesafe.ai/v1/systemone` with `TYPESAFE_API_KEY`, using the request
  and response shape from TypeSafe's official `typesafe-sdk`. Without a key it's simulated. The simulated
  Jev is strong on simple items and weaker on multi-step ones, and gives low confidence when it's wrong, so
  the cascade threshold catches most misses. Its price ($0.042 per 1M input tokens, output free) is the
  launch figure from third-party write-ups, so check it against your own contract.
- **The LLM step** is live when the provider's API key is set. Without a key, or with **Force simulation**
  on, it's simulated from a per-model profile covering time to first token, tokens/sec and accuracy by
  difficulty. The UI always shows which mode ran.
- **Provider outages** are injected faults at the LLM step, so you can see which routes survive them.

## Two pages

- **Benchmark** (`/`): runs the graded eval suite, or your own prompt, through every route with replays, and
  ranks the routes on accuracy, cost, latency, confidence, reliability and governance.
- **Ask a question** (`/ask`): type one question in plain English. It runs once through each route and
  shows that single request's answer, time per step, cost, confidence, tokens, the model that actually
  answered, and flags for cache hits, Jev decisions, routing, failover and PII. Asking the same question
  again shows the caches answering. Every question you ask is kept in a table on the page.

## Metrics

Accuracy is scored against a 10-question eval suite with known answers. Confidence is self-reported by the
model, and the calibration gap is |confidence − accuracy|. The table also shows p50/p95 latency, overhead
added by the route, cost per 1K requests (tokens plus infra) and monthly cost at your traffic level. It
also covers cache-hit rate, success rate, answer consistency across repeats, whether raw PII reached the
provider, and a 10-point capability matrix. Four weight sliders set the overall ranking.

The suite includes near-duplicate questions on purpose. "Capital of Australia" and "capital city of
Australia" produce a correct semantic-cache hit. If you lower the threshold to 0.7, the cache answers the
"Austria" question with Canberra, which shows the accuracy risk of a loose semantic cache.

## Run it

```bash
cd llm-path-lab
pip install -r requirements.txt
redis-server --daemonize yes          # optional; falls back to an in-process cache
export ANTHROPIC_API_KEY=...          # any of these enable live calls for that provider
export OPENAI_API_KEY=...  XAI_API_KEY=...  GEMINI_API_KEY=...
export TYPESAFE_API_KEY=...           # live Jev calls
uvicorn app:app --port 8088           # then open http://localhost:8088
```

You can also copy `.env.example` to `.env`, fill in your keys, and load it with `set -a; source .env; set +a`
before starting.

If you run it on a different port, set `SELF_URL=http://127.0.0.1:<port>` so the service hops can reach
each other.

Model IDs, list prices, simulator profiles, the eval suite and each route's capabilities are all in
`catalog.py`. Prices are defaults: override the selected model's price in the UI under **Assumptions**, or
edit the catalog to match your contracts and the latest model versions.

## Deploy to AWS (Terraform)

`infra/aws/` creates everything needed to run the app on AWS App Runner:

| Resource | Purpose |
|---|---|
| ECR repository | Holds the Docker image. Terraform builds and pushes it from your machine. |
| Secrets Manager: one secret per key | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `GEMINI_API_KEY`, `TYPESAFE_API_KEY`, and optional `REDIS_URL`, injected as environment variables. |
| IAM roles | One lets App Runner pull the image. The other lets the running app read only these secrets. |
| App Runner service | Public HTTPS URL, one instance (0.5 vCPU / 1 GB by default), with health checks and logs in CloudWatch. |
| Optional: WAF rate limit | Per-IP request limit (`enable_rate_limit = true`). Recommended because the page has no login. |
| Optional: AWS Budget | Emails you at 80% forecast and 100% actual of a monthly limit (`budget_email`). |

**You need:** Terraform 1.5 or later, the AWS CLI logged in to your account, and Docker running. On Apple
Silicon the image is built for `linux/amd64` automatically.

```bash
cd infra/aws
cp terraform.tfvars.example terraform.tfvars   # optional: region, size, budget email, rate limit
terraform init
terraform apply
```

**Add your API keys.** Terraform creates each secret with the placeholder `NOT_SET`, so your keys never
land in Terraform state. The app treats `NOT_SET` as "no key" and simulates that provider. Store the keys
you have, then redeploy so App Runner picks them up:

```bash
aws secretsmanager put-secret-value --secret-id agentroutecomparison/ANTHROPIC_API_KEY --secret-string 'sk-ant-...'
aws apprunner start-deployment --service-arn <from the next_steps output>
```

**Updating the app:** run `terraform apply` again. The image tag is a hash of the app's source files, so a
new image is built and deployed only when the code has changed.

**Removing everything:** `terraform destroy`. It deletes the secrets immediately and the ECR images too.

Notes:
- The app runs as a single instance on purpose. Its caches and the Jev router's similarity index live in
  memory, so extra instances would each have their own copy.
- App Runner cuts off requests after 120 seconds. The Ask page and simulated benchmarks finish well within
  that. A live benchmark on a slow model can exceed it, so set **Traffic repeats** to 1× for live runs.
- Approximate cost: about $5–25/month for App Runner, $2.40/month for the six secrets, plus about $6–10/month
  if the WAF rate limit is on. LLM API usage is billed separately by each provider.
