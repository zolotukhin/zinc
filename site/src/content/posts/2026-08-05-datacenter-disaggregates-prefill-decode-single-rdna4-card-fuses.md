---
title: "The datacenter disaggregates prefill and decode; a single RDNA4 card fuses them"
seoTitle: "Prefill-Decode Disaggregation vs Chunked-Prefill Fusion on One RDNA4 Card"
date: "2026-08-05"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - disaggregation
  - prefill
  - decode
  - chunked-prefill
  - distserve
  - splitwise
  - nvlink
  - pcie
  - agents
  - local-llm
  - llm-inference
keywords:
  - prefill decode disaggregation local LLM
  - DistServe Splitwise single GPU
  - chunked prefill vs disaggregation RDNA4
  - RX 9070 XT PCIe 5.0 KV cache transfer
  - NVLink bandwidth KV cache transfer
  - prefill decode interference consumer GPU
  - single-card agent swarm scheduling
  - disaggregated serving desktop inference
excerpt: "The best datacenter answer to prefill and decode fighting over one GPU is to stop sharing the GPU: run prefill on one pool of cards and decode on another, and ship the KV cache between them over a fast fabric. DistServe and Splitwise both report large goodput wins from that split. A single RX 9070 XT running an agent swarm has no second card to move work onto and no NVLink to move KV over, so it cannot disaggregate. It has to fuse, and chunked prefill is what makes fusion good enough that the split is not worth wanting."
seoDescription: "Prefill and decode want opposite things from a GPU, and colocating them on one card makes them interfere. The datacenter fix is disaggregation: DistServe (OSDI 2024) and Splitwise (ISCA 2024) run prefill and decode on separate GPU pools and transfer the KV cache between them over NVLink-class fabric, reporting up to 7.4x more requests within SLO and 1.4x throughput at 20% lower cost. A single 16 GB RX 9070 XT running an eight-agent local swarm cannot copy that design: it is one GPU on PCIe 5.0 with no peer fabric, so there is nowhere to send the decode work and no cheap path for the KV. This post explains why disaggregation is a multi-GPU technique bought with request volume and a fast interconnect, why a single-user swarm has neither, and why chunked-prefill fusion is the correct answer on a consumer card. It also models the one number people assume kills the desktop case, the KV transfer, and shows the transfer itself is cheap at 9B; the real cost is the second card and its duplicated weights."
faqs:
  - question: "What is prefill-decode disaggregation?"
    answer: "It is a serving design that runs the two phases of an LLM request on separate GPUs. Prefill, which reads the whole prompt at once and saturates compute, runs on one pool of cards; decode, which emits one memory-bound token at a time, runs on another. The KV cache produced by prefill is transferred to the decode card over the cluster interconnect. DistServe (OSDI 2024) and Splitwise (ISCA 2024) introduced this split to remove the interference that colocating the phases causes, and it now underpins several production serving stacks."
  - question: "Why can't a single consumer GPU disaggregate prefill and decode?"
    answer: "Disaggregation needs at least two GPUs, one for each phase, plus a fast link to move the KV cache between them. A desktop with one RX 9070 XT has neither: there is no second card to place the decode pool on, and consumer RDNA4 cards have no NVLink-class peer fabric, only PCIe 5.0 to the host. With one GPU the phases must share the same silicon, so the only lever left is when each phase runs, not where. That lever is chunked prefill."
  - question: "Isn't the KV cache transfer the reason disaggregation fails on a desktop?"
    answer: "No, and it is worth being precise. For a 9B model with a roughly 250 MB KV cache per short prompt, the transfer is about 0.28 ms over NVLink and about 4 ms over PCIe 5.0, both negligible next to an 8.3-second prefill. The transfer is cheap. The real costs are structural: you need a whole second 16 GB card, it has to hold its own copy of the model weights, and a single-user swarm cannot generate enough prefill volume to keep a dedicated prefill card busy. The transfer only becomes a first-order tax at much larger models and contexts."
  - question: "What does a single-card swarm do instead of disaggregating?"
    answer: "It fuses. Chunked prefill splits an incoming prompt into small slices and runs one slice per iteration alongside the in-flight decodes, so a new prompt never freezes the decoding agents. That buys the same isolation disaggregation buys, but in the time domain instead of across separate hardware, at a cost of roughly 3 percent added time-to-first-token for the incoming agent. On one GPU, fusion is not a compromise forced by poverty; it is the right tool."
  - question: "Would two RX 9070 XT cards make disaggregation worth it?"
    answer: "Rarely, for a single user. A dedicated prefill card sits idle whenever no prompt is arriving, and a single-user agent swarm has bursty, low-volume prefill compared to what one card can process, so utilization is poor. The same second card does more good running tensor parallelism to speed every token, or holding more agents' KV cache, than sitting in a prefill-only role. Disaggregation earns its keep on clusters that aggregate thousands of concurrent requests, which is exactly what a desktop does not have."
draft: false
---

For two weeks this series has treated prefill as the thing that goes wrong in an [eight-agent swarm](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/). A ninth agent shows up with a prompt, the prompt saturates the GPU, and the eight agents already decoding [go silent for 8.3 seconds](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/). Yesterday's post fixed it on a single card with [chunked prefill](/blog/2026-08-04-chunked-prefill-keeps-an-rdna4-swarm-decoding/): slice the prompt, let the decodes ride between the slices, pay about 3 percent more time-to-first-token, and the freeze is gone.

There is a different fix, and it is the one the datacenter actually reaches for first. Do not share the GPU at all. Run prefill on one set of cards and decode on another, and ship the KV cache from the prefill card to the decode card when the prompt is done. If the two phases never touch the same silicon, they cannot interfere, and you get to size each pool for its own bottleneck. This is prefill-decode disaggregation, and on a cluster it is a large, well-measured win.

The interesting part is that a single RX 9070 XT cannot do it, and once you see why, the reason is not the one most people guess. It is not that the KV cache is too slow to move. It is that disaggregation is a technique you buy with a second GPU and a room full of requests, and a single-user swarm has neither. Which is exactly why chunked prefill is the right answer on a desktop rather than a consolation prize.

## Two phases that never wanted to share a card

The reason disaggregation exists is the same split this series keeps returning to. Prefill reads the entire prompt in parallel and is compute-bound; it wants the newest, fastest matrix hardware and it fills the machine. Decode emits one token per step, touches the whole weight set to do it, and is memory-bound; it leaves most of the compute idle and mostly wants bandwidth. The [Splitwise paper](https://arxiv.org/abs/2311.18677) at ISCA 2024 put a fine point on it: token generation does not need the compute capability of the latest GPUs and can run on cheaper, lower-power hardware.

When both phases share a card, they collide. A prefill hogs the compute a decode does not need and starves the decodes of the step they do need. [DistServe](https://arxiv.org/abs/2401.09670), from Peking University and UC San Diego at OSDI 2024, calls this prefill-decoding interference and shows it also couples two decisions that should be independent: how much parallelism prefill wants is not how much decode wants, and colocating forces one answer on both.

Disaggregation cuts the knot by putting the phases on different GPUs. DistServe reports that doing so, and then right-sizing each pool separately for its own latency target, serves 7.4x more requests or meets a 12.6x tighter latency bound than a colocated system while keeping over 90 percent of requests inside their constraints. Splitwise, splitting the phases across separate machines, reports 1.4x higher throughput at 20 percent lower cost, or 2.35x more throughput at the same cost and power. These are not rounding errors. On a cluster, not sharing the card is one of the biggest levers there is.

## The move that makes it work is a KV cache transfer

Splitting the phases has a catch the papers are careful about: the KV cache built during prefill lives on the prefill card, and decode needs it. So every request has to move its KV from one GPU to another before the first token can come out. Splitwise implements this transfer over the fast back-plane interconnects that GPU clusters already have, and DistServe explicitly places the two phases according to the cluster's bandwidth so the communication stays small.

That interconnect is the hidden load-bearing assumption. Inside an eight-GPU H100 node the cards talk over NVLink at roughly 900 GB/s aggregate per GPU. A desktop card talks to the rest of the world over PCIe. The RX 9070 XT is a [PCIe 5.0 x16](https://overclock3d.net/news/gpu-displays/amd-radeon-rx-9070-xt-listed-as-a-pcie-5-0-gpu-by-multiple-sources/) card, which is about 64 GB/s in one direction, and consumer RDNA4 has no NVLink-class peer fabric at all, the same missing [Infinity Fabric link](/blog/2026-05-21-tensor-parallelism-two-r9700s-pays-for-the-infinity-fabric-rdna4-left-out/) that made two-card tensor parallelism awkward back in May. So the fabric the whole design leans on is either fourteen times slower or simply absent.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-05-disaggregation-vs-fusion-topology.svg" alt="A blueprint-style topology diagram on a deep navy grid background titled 'Two answers to prefill/decode interference, set by your GPU count'. The left half, labelled 'Datacenter, disaggregate across GPUs', shows a stack of small bars labelled 'thousands of requests' feeding an amber-outlined box 'Prefill GPU pool, compute-bound, runs flat out, holds its own copy of the weights'. A cyan arrow labelled 'KV cache transfer over NVLink, about 900 GB/s fabric' points down from the prefill box to a green-outlined box 'Decode GPU pool, memory-bound, streams weights, also holds a full copy of the weights'. Captions read 'two or more GPUs, phases right-sized independently', '7.4x more requests within SLO, DistServe OSDI 2024', and 'a dedicated prefill pool only pays off when request volume keeps it busy'. The right half, labelled 'Single card, fuse in time', shows one cyan-outlined box 'RX 9070 XT, 16 GB, one weight copy' containing a horizontal stream of alternating amber prefill-chunk blocks and thin green decode ticks, labelled 'prefill chunk' and 'eight decodes ride every iteration', with a note 'KV stays in place, no transfer, no second weight copy'. Its captions read '1 GPU, isolation in the time domain', 'worst-case decode stall 8.3 s to 0.66 s with 128-token chunks', and 'the incoming agent pays only about 3 percent more time-to-first-token'. A bottom strip titled 'The transfer disaggregation depends on rides a fabric a desktop does not have' shows two horizontal bars: a long cyan bar 'NVLink 4 (H100), about 900 GB/s' and a short amber bar 'PCIe 5.0 x16 (RX 9070 XT), about 64 GB/s per direction', with a footnote that one 1,730-token prompt's KV is about 250 MB, 0.28 ms on NVLink versus about 4 ms on PCIe, cheap at 9B but the second card's duplicated weights are not." loading="lazy" />
  <figcaption>Two structurally different answers to the same interference. The datacenter separates the phases onto different GPUs and pays with a KV transfer over a fast fabric. A single RX 9070 XT keeps both phases on one card and separates them in time with chunked prefill. The bottom strip is the fabric gap the desktop cannot close.</figcaption>
</figure>

Here is where the obvious conclusion is wrong. Run the numbers for the swarm's actual model, Qwen3.5-9B, where a [KV token costs about 147 KB](/blog/2026-07-29-a-decode-weight-stream-has-no-reuse-so-rdna4-infinity-cache-runs-cold/) in fp16. A fresh 1,730-token prompt, the same one the chunked-prefill post used, has a KV cache of about 250 MB. Moving 250 MB takes about 0.28 ms over NVLink and about 4 ms over PCIe 5.0. Against an 8.3-second prefill, 4 ms is nothing. The transfer is not the problem. At 9B on one prompt, PCIe is more than fast enough.

## What actually stops the desktop is arithmetic about GPUs, not bytes

If the transfer is cheap, why can't a two-card desktop just disaggregate anyway? Because disaggregation does not spend bandwidth, it spends whole GPUs, and it only breaks even when those GPUs stay busy.

Start with the card count. Disaggregation needs a prefill pool and a decode pool, so the minimum viable system is two GPUs. Most desktops running a local swarm have one. That alone ends the conversation for the common case: there is no second card to move decode onto, so the phases share the one card whether they like it or not, and the only question left is how to schedule them in time.

Now suppose you do have a second RX 9070 XT. Disaggregation still asks each card to hold its own full copy of the model weights, because both the prefill card and the decode card run the network. On a 16 GB card where the [KV budget is already tight enough](/blog/2026-08-01-reserve-for-max-kv-cache-fits-four-of-eight-agents-on-rdna4/) that only four of eight agents fit under a reserve-for-max policy, spending a second card's 16 GB on a duplicate of a model you already have resident is a poor trade. That second card, pointed at decode or at [tensor parallelism](/blog/2026-05-21-tensor-parallelism-two-r9700s-pays-for-the-infinity-fabric-rdna4-left-out/), could hold more agents' KV or make every token faster. In a prefill-only role it mostly waits.

That waiting is the deepest reason. A dedicated prefill card is worth its cost only if prompts arrive fast enough to keep it working. One RX 9070 XT prefills at roughly 208 tokens per second. A datacenter keeps a prefill pool near saturation because it aggregates thousands of concurrent users, so prompts never stop arriving. A single person running eight agents produces prefill in bursts, a tool observation here, a new session there, with long stretches where nothing needs prefilling at all. The dedicated card would sit idle through all of it. Disaggregation converts spare request volume into latency isolation, and a single user has no spare request volume to convert.

| | Disaggregation (DistServe, Splitwise) | Single-card fusion (chunked prefill) |
| --- | --- | --- |
| GPUs required | Two or more, split into pools | One |
| How interference is removed | Separate silicon per phase | Interleave chunks in time |
| Model weights resident | One copy per pool | One copy total |
| KV cache path | Prefill GPU to decode GPU over the fabric | Never moves |
| Interconnect assumed | NVLink-class, about 900 GB/s | None |
| What keeps prefill hardware busy | Aggregated cluster request volume | Not applicable |
| Cost on this workload | A whole second 16 GB card, mostly idle | About 3 percent added time-to-first-token |

The table is the argument. Read down the two columns and the split is not datacenter-versus-desktop by prestige, it is a fork determined by two facts: how many GPUs you have and how fast they can talk. Have many cards and a fast fabric and a flood of requests, and disaggregation is the better tool. Have one card, PCIe, and one user, and every row that makes disaggregation win turns into a cost you cannot pay.

## Fusion is the right answer, not the fallback

It would be easy to file this under "the desktop is too small to do the real thing." That gets it backwards. The single card is not failing to disaggregate. It is solving the same interference problem with a technique that fits its constraints, and the technique is good.

Chunked prefill removes prefill-decode interference the way disaggregation does, by making sure a prefill never gets to monopolize the resource a decode needs. Disaggregation does it in space, by putting the phases on different silicon. Fusion does it in time, by never letting a whole prompt occupy a single iteration, so the eight decodes always get their slice. The cost is the roughly 3 percent time-to-first-token the [chunked-prefill post](/blog/2026-08-04-chunked-prefill-keeps-an-rdna4-swarm-decoding/) measured, paid entirely by the incoming agent, and in exchange the swarm never freezes. No second card, no duplicated weights, no KV crossing PCIe, no prefill silicon sitting idle between bursts.

There is even an RDNA4 bonus that disaggregation would throw away. Fusing a compute-bound prefill chunk with eight memory-bound decodes in one iteration is precisely the mixed workload that uses the card on both axes at once, the matrix cores the [decode loop leaves idle](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop/) doing prefill math while the memory system streams weights for the decodes. Disaggregation, by design, keeps those two kinds of work on two different cards and never lets them fill each other's gaps. On one card, the fusion the desktop is forced into is also the more efficient use of the silicon.

The pattern across this whole series has held again. The big wins in a local swarm are scheduling and placement, not new arithmetic. Continuous batching, on-demand KV allocation, chunked prefill, and now the decision to fuse rather than split, are all choices about when and where work runs on one fixed card. The datacenter gets to answer prefill-decode interference by buying more hardware and a faster fabric. The desktop answers it with a scheduler, and on a single RX 9070 XT that is not the weaker answer. It is the one that fits.
