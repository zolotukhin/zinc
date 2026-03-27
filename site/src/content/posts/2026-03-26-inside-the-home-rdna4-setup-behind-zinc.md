---
title: "Building a local AI rig: from trading workstation to home AI server"
date: "2026-03-26"
tags:
  - amd
  - amdgpu
  - home-lab
  - pc-build
  - local-ai
  - ai-rig
  - home-ai
excerpt: "How I built the home RDNA4 node behind ZINC — from a Ryzen 9800X3D trading workstation to an AMD Radeon AI PRO R9700 inference rig. Parts list, Noctua fan swap, assembly photos, and why a $1300 GPU turned an overclocking platform into a local AI server."
---

This was supposed to be a trading machine.

I had been running high-frequency trading bots and needed a platform I could overclock hard — tight memory timings, fast single-thread, stable under sustained load. So I picked parts for exactly that: an [AMD Ryzen 7 9800X3D](https://www.amazon.com/dp/B0DKFMSMYK) for the best single-thread on AM5 and that massive 3D V-Cache, [96 GB of G.SKILL DDR5-6000 CL26](https://www.amazon.com/dp/B0F79YGMX1) because it overclocks well and holds timings, and an [ASRock X870E Taichi](https://www.amazon.com/dp/B0DFP2Q3TM) because I wanted real VRMs and a BIOS I could actually tune with.

No GPU plans. No AI ambitions. Just a fast, quiet box for running trading strategies.

![Amazon order for AMD Ryzen 9800X3D, ASRock X870E Taichi, DDR5-6000, Noctua fans, ASUS ProArt PA602 case — $2,009 before the GPU](/blog/order.jpg)

Then I started thinking about local LLM inference on AMD hardware — the gap I wrote about in [why we are building ZINC](/blog/2026-03-25-why-we-are-building-zinc). The CPU, memory, and motherboard were already sitting on my desk, tuned and stable. All I needed was a GPU.

That one decision changed the whole project.

## Picking the case

The compact enclosure I had been using for trading was not going to cut it. A big GPU means real heat, and I wanted something I could run overnight without it turning my office into a sauna.

I landed on the [ASUS ProArt PA602](https://www.amazon.com/dp/B0CPP3DWLX). It is not a flashy gaming case. It is a big, quiet, well-ventilated E-ATX box designed for workstation builds — room for a 420 mm radiator on top, space for oversized fans, and enough GPU clearance that you do not have to play Tetris with your components.

<div class="img-pair">

![The ProArt PA602 — clean lines, mesh front, no RGB anywhere](/blog/build_screen1.jpg)
![Side panel off, mid-build — room to work](/blog/build_screen3.jpg)

</div>

I know it is not the sexiest case in the world. That is the point. I wanted a case I would forget about once it was closed up.

## Ripping out the stock fans

The PA602 ships with its own fans. They are fine for a quiet office machine. They are not fine when a GPU is pulling sustained power for hours while you sleep in the next room.

<div class="img-pair">

![Stock fans removed — first things I pulled out](/blog/IMG_8037.jpg)
![Still mounted in the front panel — about to go](/blog/IMG_8036.jpg)

</div>

I replaced everything with Noctua. Two [NF-A20 PWM chromax](https://www.amazon.com/dp/B07ZP46RNR) 200 mm fans went into the front. These move serious air at barely audible RPM. The difference was immediate — I went from "I can hear the machine" to "wait, is it on?"

![Noctua NF-A20 Chromax fans installed in ProArt PA602 front intake — the noise floor dropped dramatically](/blog/IMG_8041.jpg)

The [ProArt LC 420](https://www.amazon.com/dp/B0CXLJ2N5B) CPU cooler radiator got three [Noctua NF-A14 industrialPPC-2000 PWM](https://www.amazon.com/dp/B00KESSUDW) fans. These are the high static pressure variant — they actually push air through a radiator instead of just spinning next to it.

![Noctua NF-A14 industrial fans on ProArt LC 420 radiator — overkill for a 9800X3D, perfect for overnight AI inference runs](/blog/IMG_8035.jpg)

Total fan budget was probably more than some people spend on their GPU. I do not care. Silence during a 12-hour optimization loop is worth every dollar.

## The motherboard and CPU

The Taichi went into the case first. The board is massive — serious VRM heatsinks, tons of I/O, four M.2 slots. I originally picked it because I wanted the overclocking headroom for trading. Turns out that same VRM headroom is exactly what you want when a machine is compiling Zig, syncing over SSH, and managing GPU workloads at the same time.

Then the 9800X3D dropped into the socket. I used a [Thermalright AM5 contact frame](https://www.amazon.com/dp/B0D1V45DSL) for even mounting pressure — five dollars of insurance for repeatable thermals. [Thermal Grizzly Kryonaut](https://www.amazon.com/dp/B0F48FLCRX) on top.

<div class="img-pair">

![X870E Taichi — VRM heatsinks, empty socket, waiting for the CPU](/blog/IMG_8042.jpg)
![9800X3D seated — chosen for trading, turned out perfect for this too](/blog/IMG_8070.jpg)

</div>

Here is the thing about the 9800X3D in an AI build: nobody picks it for inference. It is "a gaming CPU." But when your actual workload is compiling a Zig codebase, running helper scripts, syncing code over SSH, tokenizing prompts, scanning logits on the CPU side, and keeping the whole machine responsive while the GPU does the heavy lifting — having the fastest single-thread chip on the platform is not a luxury. It is the reason the machine never feels slow even when the GPU is floored.

## The power supply

A [Corsair HX1000i](https://www.amazon.com/dp/B0BZ2CRW8H). 1000 watts of clean power, fully modular, and way more headroom than this single-GPU build actually needs. That is deliberate. I wanted the PSU to be the one component I never think about.

![Corsair HX1000i 1000W PSU for AMD GPU workstation — enough headroom for sustained local LLM inference](/blog/IMG_8047.jpg)

## The GPU that changed the whole project

The previous card in my daily machine was an NVIDIA GeForce RTX 4080 Founders Edition. If you have ever held one, you know the thing: a massive triple-slot slab of metal that weighs like a brick and dominates whatever case it lives in. You build *around* a 4080 FE.

So when the [AI PRO R9700](https://www.bhphotovideo.com/c/product/1927021-REG/gigabyte_gv_r9700ai_top_32gd_radeon_ai_pro_r9700.html) arrived and I pulled it out of the anti-static bag, I genuinely paused. This cannot be right. It is a compact dual-slot card with a single blower fan. No RGB. No backplate theatrics. It looks like something you would find in a Dell workstation, not in a build where you are trying to push the limits of local AI inference.

<div class="img-pair">

![Holding the R9700 — my hand for scale. Compare to any triple-slot gaming card.](/blog/gpu_2.jpg)
![The back — "AMD RADEON AI PRO" branding. Workstation-plain. Tiny.](/blog/gpu_3.jpg)

</div>

I held it in one hand and thought: this small thing has 32 GB of VRAM and 576 GB/s of memory bandwidth. This is supposed to run 35-billion-parameter models.

After years of GPUs trying to look like they belong in a spaceship, there is something deeply satisfying about a card that just looks like a tool.

![AMD Radeon AI PRO R9700 on desk — dual-slot RDNA4 card with 32 GB VRAM and 576 GB/s bandwidth](/blog/gpu_1.jpg)

It slotted into the X870E Taichi and looked almost lost. The ProArt case was designed for cards twice this size. All that empty space turned out to be a feature, not a waste — the blower has room to breathe, and the whole thermal situation is dramatically calmer than anything I have ever built with triple-slot gaming cards.

<div class="img-pair">

![R9700 installed — practically disappears inside the PA602](/blog/gpu_4.jpg)
![Full interior — motherboard, RAM, GPU, room to breathe](/blog/IMG_8039.jpg)

</div>

32 GB of VRAM is what makes this build actually useful for [ZINC](https://github.com/zolotukhin/zinc). It is the difference between running toy demos and fitting a real 35B-class model with room for KV cache, concurrent sessions, and TurboQuant experiments. For that, 32 GB is not a luxury. It is room to work.

## The finished machine

<div class="img-pair">

![Completed home AI inference setup with AMD Radeon AI PRO R9700 — ProArt PA602 under the desk](/blog/build_screen1.jpg)
![ASUS ProArt PA602 top and I/O — mesh exhaust for Noctua radiator fans](/blog/build_screen2.jpg)

</div>

Here is the full parts list with links:

| Part | Choice | Why |
| --- | --- | --- |
| CPU | [AMD Ryzen 7 9800X3D](https://www.amazon.com/dp/B0DKFMSMYK) | Originally for HFT — best single-thread AM5, 3D V-Cache |
| Motherboard | [ASRock X870E Taichi](https://www.amazon.com/dp/B0DFP2Q3TM) | VRM headroom for overclocking, strong I/O, 4x M.2 |
| GPU | [AMD Radeon AI PRO R9700](https://www.bhphotovideo.com/c/product/1927021-REG/gigabyte_gv_r9700ai_top_32gd_radeon_ai_pro_r9700.html) | 32 GB VRAM, RDNA4, 576 GB/s — the reason ZINC exists |
| Memory | [G.SKILL Trident Z5 Neo 96 GB DDR5-6000 CL26](https://www.amazon.com/dp/B0F79YGMX1) | Overclocks well, tight timings, real workstation capacity |
| Case | [ASUS ProArt PA602](https://www.amazon.com/dp/B0CPP3DWLX) | Big airflow, 420 mm rad support, GPU clearance |
| CPU cooler | [ASUS ProArt LC 420](https://www.amazon.com/dp/B0CXLJ2N5B) | Oversized AIO for sustained loads |
| Radiator fans | [3x Noctua NF-A14 industrialPPC-2000](https://www.amazon.com/dp/B00KESSUDW) | High static pressure through the radiator |
| Case fans | [Noctua NF-A20 chromax](https://www.amazon.com/dp/B07ZP46RNR) + [NF-A12x25 chromax](https://www.amazon.com/dp/B09C6DQDNT) | Quiet high-airflow replacements |
| PSU | [Corsair HX1000i](https://www.amazon.com/dp/B0BZ2CRW8H) | Clean power, headroom |
| OS drive | [WD_BLACK SN8100 2 TB](https://www.amazon.com/dp/B0F3BD1W6R) | Fast boot + builds |
| Model storage | [2x WD_BLACK SN850X 8 TB](https://www.newegg.com/western-digital-8tb-black/p/N82E16820250270) RAID0 | 14.6 TB fast scratch for GGUF models |
| Archive | [WD Ultrastar DC HC550 16 TB](https://www.newegg.com/wd-0f38462-16tb/p/N82E16822234479) | Cold storage for old runs |

## What it does now

The node runs Ubuntu 24.04.3 LTS with the Vulkan driver pinned to Mesa 25.0.7 (newer versions caused a 14% RADV regression), `RADV_PERFTEST=coop_matrix` for cooperative matrix support, and GPU ECC disabled for extra bandwidth. The llama.cpp baseline on this hardware is 107 tok/s decode on Qwen3.5-35B-A3B Q4_K_XL.

I edit code on my laptop and push to this machine over SSH. The [ZINC optimization loop](/blog/2026-03-25-why-we-are-building-zinc) rsyncs source, builds, runs, measures, and iterates — sometimes overnight, sometimes for dozens of cycles. The machine needs to be thermally stable for that. Quiet enough that I forget it is running. And producing numbers I can trust.

The fact that this started as a trading workstation turned out to be a feature. An overclocking-grade platform — fast CPU, tuned memory, real VRMs — is exactly the kind of foundation you want for sustained GPU inference work. The 9800X3D handles Zig compilation, SSH orchestration, and host-side decode overhead without breaking a sweat. The DDR5 stays out of the way — and the price of that RAM kit has tripled since I bought it, so I am glad I did not wait. The VRMs stay cool.

One $1300 GPU card turned a platform I already had into the machine that is building [ZINC](https://github.com/zolotukhin/zinc). That is the kind of leverage local AI should feel like.
