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
| Direct · small model | Agent → small LLM | The provider's small model, called directly. Shows whether the task needs the big one. |
| AI Gateway | Agent → AI Gateway → LLM | Central keys, quotas, guardrails, cost metering and provider failover |
| AI Gateway → Jev → LLM | Agent → AI Gateway → Jev → LLM if unsure | The gateway's controls, plus Jev answering fixed-answer decisions when it's confident. Free-text requests use Jev to pick the model tier. |
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

## Three pages

- **Architecture advisor** (`/`): the tool's main page, which acts as an AI architect. You describe the agent
  in plain English and it does four things:
  1. **Understands the goal.** It works out the task type, fixed labels, whether it needs your documents
     (and how big they are), tools, memory or several steps, plus personal data, risk, latency, volume and
     repeat rate. It also writes 12–16 test cases with gradable expected answers. This is done by Claude
     (`ARCHITECT_MODEL`, default `claude-opus-5`, using structured JSON output) when `ANTHROPIC_API_KEY` is
     set; otherwise built-in rules use sample cases from a similar task.
  2. **Tests every model** on those cases (plus Jev when there are fixed labels). It picks a **primary model**
     (the cheapest that meets the accuracy and hallucination targets), a **fallback model** from a different
     provider, and a **small model** for light steps.
  3. **Builds candidate architectures** from the patterns that fit the task: single call, small-to-large
     cascade, decision model (Jev) with LLM fallback, whole documents in the prompt, RAG, fixed workflow,
     tool-using agent, orchestrator with specialist agents, and batch pipeline. Add-ons are layered on as
     needed: AI Gateway, Agent Runtime, response cache, human review and a grounding check.
  4. **Ranks the designs and shows the top 3.** Designs that meet every requirement rank first, then by
     accuracy, cost, latency and simplicity; latency doesn't count for background jobs. The page explains
     why #1 won, the trade-offs of #2 and #3 against it, and "pick this if". It also includes the model
     table, a build plan, the architectures ruled out and why, and a downloadable ADR.

  Every number is labelled **measured** (single-call results on your cases), **from measurements** (e.g. the
  cascade replayed per case from measured confidences) or **estimated** (extra agent steps, tool calls and
  retrieval, which can't be run here without your systems). You can adjust any requirement or test case and
  re-run.

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
git clone https://github.com/vinayskasarla/agentroutecomparison.git && cd agentroutecomparison
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
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

## Keeping the advice trustworthy

The advisor is meant to be a single source of truth, so what it knows is kept in reviewed, dated files. The
recommendation comes from rules plus measurements; no LLM writes it.

**What the dashboard says for a goal**
1. **The answer**, in one sentence: the best model, the architecture, the monthly cost, the accuracy, and
   how fast 95% of answers arrive; plus the fallback model and what it costs.
2. **Best model and fallback cards:** per month, accuracy, hallucinations, 95% latency, and cost per 1,000
   correct answers (the efficiency figure).
   - **Fallback** is the cheapest qualifying model on a *different platform*, from a different maker where
     possible, so one outage can't take both down.
3. **Top 3 architectures for this goal**, each with its own best model, why #1 wins, and the trade-offs of
   #2 and #3. If fewer than three patterns fit, it says so rather than padding the list.
4. **Every model for this goal, priced inside architecture 1, cheapest first:** about 38 offerings across the
   Anthropic API, OpenAI API, Gemini API, xAI API, Amazon Bedrock, Azure AI Foundry and Google Vertex AI.
   - Each row shows price status, whether it was measured live or simulated (or estimated for "reported"
     models), accuracy, hallucinations, latency, cost per 1K correct answers, monthly cost, and the result.
   - The best-model row always matches the headline.

**Prices** (`knowledge/models.json`)
- **Verified automatically on every start** (`pricing_sync.py`, or "Sync prices now"):
  - Anthropic, from its official pricing page. This also covers Claude on Azure AI Foundry, which Anthropic
    bills at standard rates.
  - Amazon Bedrock, from the public AWS Price List API.
  - Azure OpenAI, from the public Azure Retail Prices API.
- **Needs owner review:** OpenAI, Google and xAI publish no machine-readable price list, and Claude on
  Bedrock and Vertex is billed through the marketplaces. Confirm these on the providers' pricing pages and
  set `status`/`verified_on`.
- **Reported:** newer models seen on third-party trackers. Their API IDs aren't confirmed, so they're priced
  as an estimate but never called.
- **Selection rule:** best-model and fallback picks prefer verified prices. If a cheaper unverified model
  qualified, the dashboard says so.

**Live calls**
- Anthropic, OpenAI, Gemini and xAI use API keys.
- Bedrock uses the AWS credentials or IAM role: Claude through the Anthropic SDK's Bedrock client, other
  models through Converse, with model IDs resolved by name.
- Azure uses `AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_API_KEY` for GPT models, and
  `AZURE_FOUNDRY_RESOURCE`/`AZURE_FOUNDRY_API_KEY` for Claude.
- Vertex uses `GOOGLE_CLOUD_PROJECT` for Claude.

Anything unreachable is simulated and labelled. When live results qualify, the best model and fallback are
picked from those.

**Patterns** (`knowledge/patterns.json`): 12 patterns with the conditions each fits, components and design
principles, each tied to a published source (Anthropic, OpenAI, AWS Well-Architected Generative AI and
Agentic AI lenses). The cross-check flags any disagreement between Claude's reading of the goal and the
keyword rules, for the user to confirm.

## Production harness

Tick **Production harness** next to the goal to wrap every suggested architecture in the controls it needs in production. The controls change with the architecture and the goal:

| Layer | Controls |
|---|---|
| Before the model | Input guardrail (prompt injection, jailbreaks, off-topic use) |
| After the model | Output validation and retry; grounding check when answers come from your documents |
| While it runs | Timeouts, retries and failover; tracing and cost metering; tool permissions and approval (agents that act); step, time and spend limits (loops and multi-step designs); escalation monitor (cascades) |
| Keeping it good | Regression test suite (this run's test cases); live quality sampling |

The harness cost and latency are added to every figure: the ranking, the winner, the fallback and the every-model table. So a simpler architecture can overtake one that needs more controls. Controls that call a model are priced from the run's measured results, and each row shows how its cost was worked out:
- the guardrail and the grounding check cost one small-model call each;
- retries are priced at the measured failure rate;
- live sampling grades 2% of traffic, an assumption you can change in `knowledge/harness.json`.

All harness figures are labelled estimates. With the box unticked, the page shows prototype numbers, and it warns when the goal takes actions, carries high risk or handles personal data. The controls and their sources are in `knowledge/harness.json`.

## Agentic systems: tools, MCP, orchestration, multiple models, retrieval

The advisor picks the pattern from the shape of the goal, and shows the reasoning as a **decision trace**. Each fact about the goal is listed with the rule it triggered, the published guidance behind the rule, and what the rule led to (the rank each pattern got, or why it was ruled out).

| Fact about the goal | Effect |
|---|---|
| Takes actions in your systems | Designs that only answer (single call, RAG, whole documents, cascade, batch, agentic retrieval) are ruled out |
| Steps are the same every time | Fixed workflow or router preferred over an autonomous agent |
| Steps vary per request | Fixed workflow ruled out; tool-using agent |
| 3+ distinct request types | Router to specialised handlers |
| Specialist domains, or 15+ tools | Orchestrator with specialist agents considered |
| 20+ tools | Tool search: load tool definitions on demand |
| Uses documents and takes actions | Retrieval becomes a tool or step inside the agent (vector store priced in) |
| Answers combine several documents or sources, or facts live in tables | Agentic retrieval (search, read, search again; text-to-SQL for tables); single-pass RAG ruled out |
| Multi-model off | One model for every role; two-model designs ruled out |

**How agentic designs are priced:**
- Each tool call is one more model turn, and each turn reads the tool results so far.
- Tool definitions are counted in every call, and cached where the platform allows it.
- API latency is added per call.
- MCP servers you host are priced as infrastructure.
- Accuracy, cost and speed per call are measured; the number of turns comes from your inputs. Whether the agent picks the right tool isn't tested yet, and the page says so.

**What the page takes and shows:**
- **Multi-model checkbox:** allows a different model per role. A separate small model is used only where it is cheaper and good enough.
- **Agent system inputs:** number of tools, MCP or direct integration, MCP servers, tool calls per request, API latency, request types, whether steps vary, specialists, multi-document questions, changing documents, and data in tables. Leave them blank to work them out from the goal.
- **System design card:** the decision trace, models per role, a tools and integration plan (MCP or direct tools, tool definitions per call, tool search, governance controls), and a retrieval plan. The retrieval plan covers strategy, chunking, hybrid search, reranking, filters and permissions, re-indexing, citations, text-to-SQL and retrieval evaluation, each with the reason it applies.

Rules, thresholds and defaults live in `knowledge/agentic.json`. Each cites its source, and the thresholds are this tool's defaults, not vendor limits.

## Beyond accuracy: what else decides the answer

| Question an architect asks | How the advisor handles it | Where to change it |
|---|---|---|
| "Those aren't my test cases." | Upload up to 60 real cases (CSV: `input,expected`) plus a reference document under *Adjust*. Every accuracy figure shows a 95% range. Passes whose range dips below the target are marked ≈, and cheaper models that missed only by noise are listed. | page |
| "Real prompts are much longer." | System prompt + tool definitions, retrieved text, whole documents and chat history are added to every call's tokens, with sensible defaults per task that you can edit. | page (*Adjust*) |
| "What about prompt caching?" | The static part of the prompt is priced at each platform's cached-input rate when it is over the platform minimum. That covers the system prompt, plus the documents in the whole-documents design. | `knowledge/capabilities.json` |
| "Can it take the load?" | Shows peak requests and tokens per minute (busiest hour × model calls per request). It is checked against your quotas if you enter them, and each platform's quota page and provisioned option are named. | `knowledge/quotas.json` |
| "Can this data go there?" | *Data class* and *region* rule models out before testing. Models that can't call tools, or can't read images the goal needs, are ruled out too. | `policy.json` (`data_classes`, `regions`) |
| "We already run a gateway / vector store." | Reused at no extra cost. A platform you have a spend commitment with wins when it is within 15% of the cheapest. | page (*Constraints*) |
| "What does it really cost?" | Adds fixed infrastructure (e.g. the vector store) and build effort in engineer-weeks, plus first-year cost when you give an engineer-week rate. Ranking uses total monthly cost. | `knowledge/tco.json` |
| "Will the model still be around?" | Retirement dates from `models.json` (`retires_on`) or retirement notices in the news feed are flagged on the winner, the fallback and the table. | `models.json`, news feed |
| "How sure are you?" | *Does the answer change?* re-ranks the same measurements under six what-ifs, with no extra model calls. | — |

Capabilities, caching rates, data rules and TCO figures are starting points marked for review. Unconfirmed context windows are left empty and skipped, not guessed.

### Your decision stays with you

Advice is computed on the fly. Nothing about a decision is stored per user, and there is no database.
- **Decision record:** choosing *Accept* or *Choose differently* (with a reason and who decided) only affects the decision record you download.
- **Result cache:** results are kept per **browser session**. Asking the same thing again in the same browser session (or opening the result link, `?r=…`) is instant. A new browser session always gets a fresh run. The cache lives in server memory only, for at most `ADVICE_CACHE_TTL` (24 h) and `ADVICE_CACHE_MAX` (200) results, and a restart clears it. *Run it fresh* bypasses it. Note that browsers set to restore the previous session also restore its cookie.

**First run vs full comparison:** a new goal tests a **shortlist** of about 4 models: the cheapest in each tier, preferring ones this server can call live, plus one on another platform so a fallback is measured. The model table still prices every model, marking the rest "not compared yet". **Compare all models** (under the table) tests every eligible model on the same test cases. It's limited to `COMPARE_LIMIT_PER_DAY` (default 5) per client IP per rolling 24 hours, in memory. The IP is the one App Runner's proxy appends to `X-Forwarded-For` (`TRUSTED_PROXY_HOPS`). Reopening a cached comparison doesn't count against the limit.

**Architect model:** reading the goal and writing test cases uses `ARCHITECT_MODEL` (default `claude-sonnet-5`) at `ARCHITECT_EFFORT` (default `medium`). On four sample goals, Sonnet 5 matched Opus 5's reading on 42 of 44 decisions for about $0.023 per goal vs $0.053. Set `ARCHITECT_MODEL=claude-opus-5` for the strongest reading.

**Optional password:** set `APP_PASSWORD` to require HTTP basic auth (any username) until SSO is in front of the app.

The audit trail in CloudWatch still records requests and actions as before.

## Company AI vendor policy

`knowledge/policy.json`, or the file named by `POLICY_FILE`, lists the approved platforms. Right now those
are the OpenAI API, Anthropic API (Claude), xAI API (Grok), Gemini API, Amazon Bedrock and Azure AI Foundry.
It can also block individual model makers and approve extra services such as Jev.

The server enforces the policy everywhere:
- **Evaluation:** only approved models are tested, so test cases never reach another vendor.
- **Recommendations:** only approved models are recommended or shown in the cost table.
- **Ask and Benchmark:** a request for a blocked model gets a 403, and a `policy_blocked` audit event is logged.
- **Failover:** it only switches to approved models.
- **Jev:** it stays off until `"jev"` is added to `approved_services`. That removes the decision-model pattern,
  the Jev routes and the Jev router.

The header's **Policy** pill shows what's approved and what's excluded, and why.

## Model news tab

`/news` fetches the providers' official feeds when you open the tab (cached for 30 minutes; **Refresh now**
forces a fetch). It filters them to model launches, pricing changes, retirements and features, and checks
every model name mentioned against the catalog.

- Sources are in `knowledge/news_sources.json`:
  - Anthropic API release notes
  - OpenAI news
  - Google AI and DeepMind blogs
  - AWS What's New for Bedrock, and the AWS ML blog
  - Azure updates for AI Foundry
  - the Hugging Face blog
- Models mentioned in the news but missing from `knowledge/models.json` are listed at the top as **Not in our
  catalog yet**. That's the to-do list for keeping the advisor current.
- A source that can't be reached from the server is shown as unreachable instead of being silently skipped.

## Signed-in user

Once SSO is in front of the app, the header shows the user's name, email and initials, read from the same
headers the audit trail uses (`/api/me`). Set `LOGOUT_URL` to show a *Sign out* link. Without SSO it shows
*Guest*.

## Audit trail (CloudWatch)

Every user action is written as one JSON line on stdout with `"log_type": "audit"`; App Runner ships it to
CloudWatch Logs. Events:

- `page_view`
- `advise_requested` and `advise_completed` (goal, how it was understood, models tested, top 3 with models
  and cost), or `advise_failed`
- `ask_question` and `benchmark_run`
- `adr_downloaded`, `adjust_opened`, `example_used`, `knowledge_opened`, `details_opened`
- `knowledge_checked`

Each event carries `user` (from SSO), `session_id`, `request_id`, IP and user agent. Personal data in goals and
prompts (emails, card, phone and SSN numbers) is redacted before logging (`AUDIT_REDACT_PII=true`).

**SSO:** the user is read from headers set by the SSO layer in front of the app (`AUDIT_USER_HEADERS`,
first match wins):
- ALB or Cognito OIDC sets `x-amzn-oidc-data`, a JWT whose `email` and `sub` claims are used.
- oauth2-proxy sets `x-forwarded-email`.

Only trust these headers if the app can be reached solely through that layer. App Runner's public URL
bypasses it, so either make the service private behind your SSO proxy, or run it on ECS behind an ALB with
OIDC authentication.

`terraform apply` adds these CloudWatch resources:
- saved Logs Insights queries: *who asked what*, *recommended architectures*, *activity by user*,
  *all actions* and *failures*;
- `AdviceCompleted` and `AdviceFailed` metrics, with an alarm on failures;
- a **usage dashboard**.

The `audit` output prints the log group, the dashboard link and a command to set log retention.

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
