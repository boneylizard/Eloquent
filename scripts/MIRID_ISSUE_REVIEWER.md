# Mirid issue reviewer

This is a manual maintainer tool. It reviews one GitHub issue at a time with
whichever model you have chosen and loaded in LM Studio.

It does not run continuously and it does not take action on GitHub.

## What it does

1. Reads one GitHub issue and its comments.
2. Optionally reads diagnostic or log files that you name explicitly.
3. Lets the selected model find, search, and read source files in the local
   Mirid repository.
4. Asks whether the report contains enough evidence to identify the issue.
5. Writes one Markdown report that can be given directly to Codex.

Every source search and file read is printed while the review runs and recorded
at the end of the report.

## What it cannot do

- edit any source file
- execute code or commands
- run tests
- post comments or labels to GitHub
- close issues
- contact the reporter
- read `.env`, settings, personal data, models, runtime packages, logs, build
  output, or other excluded repository folders

The only file it creates is the final Markdown report under
`artifacts\triage`, unless you choose another output path.

## Run a review

1. Open LM Studio.
2. Load the model you want to use.
3. Open the Developer page and start the local server.
4. From the Mirid repository, run:

```powershell
python scripts\review_github_issue.py https://github.com/boneylizard/Eloquent/issues/4
```

Include a diagnostic report or log file:

```powershell
python scripts\review_github_issue.py 4 `
  --diagnostic "C:\path\to\mirid-diagnostic.txt"
```

If you deliberately have several models loaded, name the model instance:

```powershell
python scripts\review_github_issue.py 4 --model "model-instance-id"
```

The script otherwise uses the single LLM currently loaded in LM Studio. This
means changing models does not require changing the reviewer.

LM Studio normally listens on `http://127.0.0.1:1234`. If you changed the port:

```powershell
python scripts\review_github_issue.py 4 --server "http://127.0.0.1:5678"
```

If LM Studio API authentication is enabled, place the token in the
`LM_STUDIO_API_TOKEN` environment variable. The token is used only to call the
local LM Studio server and is never included in the review prompt or report.

## The output

The report states:

- whether the evidence is sufficient, partially sufficient, or insufficient
- what the user actually reported
- useful error and log lines
- relevant source evidence
- what can and cannot be concluded
- the smallest amount of missing information
- a bounded likely cause, when evidence supports one
- a self-contained Codex handoff
- every source file inspected

Issue text, comments, diagnostics, and source files are treated as untrusted
evidence rather than instructions.
