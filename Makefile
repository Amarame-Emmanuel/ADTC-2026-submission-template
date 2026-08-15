# Àgbẹ̀ - offline Yoruba/English crop advisory
#
# Every target that executes model code runs inside the constrained container,
# so nothing is ever measured or demoed with more resources than the ADTC
# Standard Laptop provides.
#
# Usage from Windows: run these from WSL2 or Git Bash with Docker Desktop up.

IMAGE       ?= agbe:latest
MEM_LIMIT   ?= 7g
CPUS        ?= 4
PORT        ?= 8000

# Git Bash / MSYS on Windows rewrites arguments that look like Unix paths into
# Windows paths before the program sees them. `-v "$(PWD):/app"` becomes
# `-v "C:/...:C:/Program Files/Git/app"`, so Docker mounts the host directory
# at a path nothing reads from.
#
# It fails SILENTLY. The container starts, the image's own /app is used, work
# is written into the container's ephemeral layer, and everything is discarded
# on exit. A 2 GB model download completed, verified its checksums, reported
# success, and left nothing behind.
#
# Harmless on Linux and macOS, where the variable is simply unused.
export MSYS_NO_PATHCONV=1

# --memory-swap == --memory disables swap entirely. Without this the kernel
# would page instead of OOM-killing, which would let us silently exceed the
# 7GB ceiling and still "pass". We want a hard failure if we overrun.
#
# CPU: --cpuset-cpus, NOT --cpus. This distinction is worth 30x.
#
# `--cpus=4` caps CPU *quota* but leaves os.cpu_count() reporting every core on
# the host - 24 here. Every threaded library (OpenBLAS, OpenMP, llama.cpp)
# therefore spawns 24 threads, which then timeshare 4 CPUs worth of quota. The
# contention costs far more than the work: measured 6.0 GFLOPS against 115
# unrestricted, when the fair share should have been ~19.
#
# `--cpuset-cpus=0-3` restricts the visible CPU set, so sched_getaffinity
# reports 4 and libraries size their pools correctly: 170 GFLOPS. It is also
# the more faithful emulation of a 4-core laptop - a quota can burst, a cpuset
# cannot.
#
# The thread variables are belt-and-braces for libraries that read cpu_count()
# rather than the affinity mask.
#
# This defect made a 20-minute index build project to 106 HOURS, and would have
# made every benchmark ~30x pessimistic - reporting a system that looks
# unusable on the hardware it was designed for.
CPUSET      ?= 0-3
CONSTRAIN = --memory=$(MEM_LIMIT) --memory-swap=$(MEM_LIMIT) \
            --cpuset-cpus=$(CPUSET) \
            -e OMP_NUM_THREADS=$(CPUS) \
            -e OPENBLAS_NUM_THREADS=$(CPUS)

# Models and index are bind-mounted so they survive image rebuilds; a fresh
# `make build` should never force a multi-GB re-download.
#
# $(CURDIR), not $(PWD). PWD is an environment variable exported by bash and
# NOT by PowerShell, so `make run` from a PowerShell prompt expanded it to
# nothing: the mounts became "-v /models:/app/models", Docker silently created
# empty directories at the container root, and every request failed with "no
# index at /app/index" while the index sat untouched on disk.
#
# $(CURDIR) is a GNU Make built-in holding the working directory. It is always
# set, on every shell and platform.
#
# The whole repository is mounted rather than four subdirectories: it keeps the
# running code in step with the working tree, so an edit does not require an
# image rebuild to take effect.
MOUNTS = -v "$(CURDIR):/app"

.PHONY: help build fetch-models convert-models fetch-corpus index run shell \
        bench coverage answers test verify-no-torch submission verify-submission clean

help:
	@echo "Àgbẹ̀ - make targets"
	@echo ""
	@echo "  build          Build the ubuntu:22.04 reference image"
	@echo "  fetch-models   Download + checksum-verify all model weights"
	@echo "  convert-models Convert NLLB to CTranslate2 int8 (throwaway torch image)"
	@echo "  fetch-corpus   Harvest vetted extension documents + write manifest"
	@echo "  index          Build the RAG index from corpus/"
	@echo "  run            Serve the app under the 7GB / 4-CPU cap"
	@echo "  run-offline    Same, with NO network interface - proves offline operation"
	@echo "  verify-offline Headless offline proof, suitable for CI/audit"
	@echo "  bench          Run the benchmark harness, write bench/results/"
	@echo "  coverage       Measure retrieval coverage + refusal accuracy"
	@echo "  shell          Interactive shell inside the constrained container"
	@echo "  test           Run the test suite"
	@echo ""
	@echo "  Constraints:   MEM_LIMIT=$(MEM_LIMIT)  CPUS=$(CPUS)"

build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

# Model download is NOT memory-constrained - it is pure I/O and constraining it
# only makes the download flaky.
#
# Mounts the repository root rather than just models/, because the script
# writes models.lock.json at the root. With only models/ mounted, the lock was
# written inside the container and discarded on exit - the checksums were
# computed, reported as written, and then lost.
#
# The HF cache is mounted too so an interrupted multi-gigabyte download resumes
# instead of restarting.
fetch-models:
	docker run --rm \
		-v "$(CURDIR):/app" \
		-v "agbe-hf-cache:/root/.cache/huggingface" \
		$(IMAGE) python scripts/fetch_models.py $(ARGS)

# One-shot conversion of NLLB-200 to CTranslate2 int8, in a throwaway image
# that carries PyTorch. Torch is a BUILD dependency here, like a compiler: the
# runtime image never contains it, which is what makes the memory budget in
# REPORT.md a description of what ships rather than an aspiration.
#
# The HF cache volume is mounted for the same reason fetch-models mounts it:
# NLLB-200-distilled-600M is a ~2.5 GB download, and a stall partway through
# should resume rather than start over. This was learned by watching a run sit
# at exactly 1,000,000,000 bytes for an hour.
#
# MSYS_NO_PATHCONV=1 is not optional on Windows. Git Bash rewrites arguments
# that look like Unix paths, and it mangles the -v argument into
#
#   C:\...\LLM\models;C  ->  \Program Files\Git\work\models
#
# which Docker accepts silently. The conversion then runs to completion and
# writes its output into a directory nobody will ever look in. This has now
# cost two multi-gigabyte downloads; the export makes the target safe to run
# from any shell.
convert-models:
	docker build -f docker/Dockerfile.convert -t agbe-convert:latest .
	MSYS_NO_PATHCONV=1 docker run --rm \
		-v "$(CURDIR)/models:/work/models" \
		-v "agbe-hf-cache:/root/.cache/huggingface" \
		agbe-convert:latest

# Corpus harvesting is network-bound and runs unconstrained. It writes
# corpus/manifest.json, which pins provenance, licence and SHA-256 for every
# document. The documents themselves are never committed.
fetch-corpus:
	docker run --rm $(MOUNTS) $(IMAGE) python scripts/fetch_corpus.py $(ARGS)

index:
	docker run --rm $(CONSTRAIN) $(MOUNTS) $(IMAGE) python -m agbe.rag.build_index

run:
	docker run --rm -it $(CONSTRAIN) $(MOUNTS) -p $(PORT):8000 $(IMAGE)

# The offline claim, proved rather than asserted.
#
# --network none gives the container NO network interface at all: no DNS, no
# loopback to the host, no route anywhere. If any part of this system quietly
# depended on reaching the internet - a model download, a tokenizer fetch, a
# font, a CDN script, telemetry - it fails here, immediately and visibly.
#
# This is the target to run on camera. "It works offline" is a claim; a
# container with no network device answering a farmer's question is evidence.
#
# Port publishing still works because it is handled by the host's proxy, not
# by the container's (nonexistent) network stack.
run-offline:
	@echo ">>> starting with NO network interface (--network none)"
	@echo ">>> any hidden internet dependency will fail immediately"
	docker run --rm -it --network none $(CONSTRAIN) $(MOUNTS) -p $(PORT):8000 $(IMAGE)

# Same proof, headless: answers a question with no network and exits non-zero
# if anything reached for the internet. Suitable for CI and for the audit.
verify-offline:
	docker run --rm --network none $(CONSTRAIN) $(MOUNTS) $(IMAGE) \
		python -c "\
from agbe.advisor import AdvisoryEngine; \
e = AdvisoryEngine(); \
a = e.advise('My cassava leaves are yellow and twisted'); \
print(a.answer[:400]); \
print(); \
print('sources:', a.citations()); \
print(); \
print('OK: answered with no network interface')"

shell:
	docker run --rm -it $(CONSTRAIN) $(MOUNTS) $(IMAGE) bash

# The headline evidence artifact. Runs under the same cap as `run`, records
# peak RSS and throughput, and labels results with the host CPU/RAM so rows
# from different machines stay comparable.
bench:
	docker run --rm $(CONSTRAIN) $(MOUNTS) $(IMAGE) python -m bench.run

# Sustained-load thermal behaviour. Twenty minutes of continuous generation -
# a load heavier than advisory use ever produces - to test for the one symptom
# of throttling that is visible without a thermometer: throughput decay.
#
# Reports peak core temperature as null on hosts that do not expose sensors,
# which the official profiler's schema explicitly permits, rather than
# substituting a comforting number. See bench/thermal.py on why a temperature
# from a development machine would not transfer to the target laptop anyway.
thermal:
	docker run --rm $(CONSTRAIN) $(MOUNTS) $(IMAGE) python -m bench.thermal $(ARGS)

# Assemble submission/model/ from the pinned weights, and verify that the three
# places naming the model agree: models.lock.json, agbe/config.py and
# submission/metadata.json.
#
# The bundle used to be filled by hand and drifted the moment the model changed
# - it declared Q4_0 and contained Q4_K_M, which are exactly the two files whose
# 2.7x throughput difference the S_perf case rests on. `verify-submission` is
# the check; run it before packaging anything.
submission:
	python scripts/build_submission.py

verify-submission:
	python scripts/build_submission.py --check

# Gate 1 screenshots, captured by driving the real UI in a real browser.
#
# The browser lives in a throwaway image (docker/Dockerfile.shots), never the
# runtime. Both containers join a private network so Playwright can reach the
# server by name without publishing a port on the host.
#
# The server is started detached and torn down in all cases, including failure,
# so a crashed capture cannot leave a container holding port 8000.
screenshots:
	docker build -f docker/Dockerfile.shots -t agbe-shots:latest .
	docker network create agbe-shots-net 2>/dev/null || true
	MSYS_NO_PATHCONV=1 docker run -d --rm --name agbe-ui \
		--network agbe-shots-net $(CONSTRAIN) $(MOUNTS) $(IMAGE)
	-MSYS_NO_PATHCONV=1 docker run --rm --network agbe-shots-net \
		-v "$(CURDIR)/docs/assets:/out" agbe-shots:latest --url http://agbe-ui:8000
	-docker stop agbe-ui
	-docker network rm agbe-shots-net

# Answers "is the corpus enough?" with a measurement rather than an opinion.
# Reports retrieval coverage on in-scope questions AND refusal accuracy on
# deliberately out-of-scope ones; the two constrain each other, so the
# similarity floor cannot be tuned to flatter one number alone.
coverage:
	docker run --rm $(CONSTRAIN) $(MOUNTS) $(IMAGE) python -m bench.coverage --save $(ARGS)

# Answer-level evaluation. `coverage` above measures whether the right passage
# was RETRIEVED; this measures whether the ANSWER said the right thing, which is
# not the same and came apart badly enough to need its own tool - coverage sat
# unmoved through changes that made answers worse and the one that made them
# right. See REPORT.md 6.9.
#
# Costs a generation per question (~20s against coverage's ~70ms), so it is
# separate rather than a flag: coverage stays cheap enough to run constantly,
# this runs when answers might have changed.
answers:
	docker run --rm $(CONSTRAIN) $(MOUNTS) $(IMAGE) python -m bench.answers --save $(ARGS)

test:
	docker run --rm $(CONSTRAIN) $(MOUNTS) $(IMAGE) python -m pytest tests/ -v

verify-no-torch:
	@docker run --rm $(IMAGE) python -c "\
import importlib.util, sys; \
sys.exit(1) if importlib.util.find_spec('torch') else print('OK: no torch in runtime image')"

clean:
	rm -rf index/* bench/traces/*
