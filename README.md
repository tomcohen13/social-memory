<div align="center">

# Midterm Report

**Evaluation of uni- and multi-modal models on Social-IQ 2.0**

<br/>

</div>

---

## Overview

This repository implements **unimodal evaluation pipelines** for social-scene understanding: models answer multiple-choice questions using either **transcript-only** (language) or **native video** (vision-language) inputs. A small **pipeline abstraction** keeps shared concerns—configuration, prompting, model wiring, logging, and result export—in one place so new modalities or baselines can plug in through a single registry.

---

## Architecture

### Pipeline abstraction

All runs implement a common `Pipeline` base class: load config and prompts, bind the chat model with retries, stream async inference over the dataset, and write structured results under `results/`. Concrete pipelines register by CLI name in `PIPELINE_REGISTRY` (`language`, `video`).

| Piece | Role |
| ------ | ----- |
| [`Pipeline`](src/social_memory/pipelines/base.py) | Shared lifecycle: config, logging, model runner, output paths |
| [`PIPELINE_REGISTRY`](src/social_memory/pipelines/registry.py) | Maps CLI `--pipeline` values to implementations |
| [`LanguagePipeline`](src/social_memory/pipelines/lang_baseline.py) | Transcript-only inputs; zero-shot MCQ on text |
| [`VideoPipeline`](src/social_memory/pipelines/video_baseline.py) | Video inputs; zero-shot MCQ with visual context |

### Unimodal pipelines

<table>
<tr>
<td width="50%" valign="top">

**Language** · `language`

- Uses **transcript-only** context (no pixels).
- Suited for text LLMs and chat APIs that do not accept video.
- Implemented in `LanguagePipeline`.

</td>
<td width="50%" valign="top">

**Video** · `video`

- Uses **native video** (and matching VLMs / video-capable APIs).
- Keeps the same task and evaluation hook as the language path for fair comparison.
- Implemented in `VideoPipeline`.

</td>
</tr>
</table>

---

## Results · validation accuracy

Five models across **Google (Gemini)**, **OpenAI**, **Meta (Llama)**, and **Qwen** were evaluated on the validation split. Accuracies below are **validation** scores; the **pipeline** column indicates whether the run used the **language** (transcript) or **video** pipeline.

<table>
<thead>
<tr>
<th align="left">Provider</th>
<th align="left">Model</th>
<th align="center">Modalities</th>
<th align="right">1 min (original)</th>
<th align="right">1-5 min</th>
<th align="right"> 5-10 min</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><img src="assets/google.svg" width="22" height="22" alt="Google" valign="middle" /></td>
<td align="left"><code>gemini-2.5-pro</code></td>
<td align="center"><code>video</code><code>audio</code></td>
<td align="right"><strong>81.76%</strong></td>
<td align="right"><strong>78.62%</strong></td>
<td align="right"><strong>77.65%</strong></td>
</tr>
<tr>
<td align="left"><img src="assets/google.svg" width="22" height="22" alt="Google" valign="middle" /></td>
<td align="left"><code>gemini-2.5-pro</code></td>
<td align="center"><code>audio</code></td>
<td align="right"><strong>77.84%</strong></td>
<td align="right">—</td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/google.svg" width="22" height="22" alt="Google" valign="middle" /></td>
<td align="left"><code>gemini-2.5-pro</code></td>
<td align="center"><code>video</code></td>
<td align="right"><strong>74.14%</strong></td>
<td align="right"><strong>70.13%</strong></td>
<td align="right"><strong>68.83%</strong></td>
</tr>
<tr>
<td align="left"><img src="assets/google.svg" width="22" height="22" alt="Google" valign="middle" /></td>
<td align="left"><code>gemini-2.5-flash</code></td>
<td align="center"><code>text</code></td>
<td align="right"><strong>73.81%</strong></td>
<td align="right">—</td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/openai.svg" width="22" height="22" alt="OpenAI" valign="middle" /></td>
<td align="left"><code>gpt-4o-preview</code></td>
<td align="center"><code>audio</code></td>
<td align="right"><strong>70.86%</strong></td>
<td align="right">—</td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/meta.svg" width="22" height="22" alt="Meta" valign="middle" /></td>
<td align="left"><code>Llama 3.3 70B</code></td>
<td align="center"><code>text</code></td>
<td align="right"><strong>69.65%</strong></td>
<td align="right">—</td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/google.svg" width="22" height="22" alt="Google" valign="middle" /></td>
<td align="left"><code>gemini-2.5-flash</code></td>
<td align="center"><code>video</code></td>
<td align="right">—</td>
<td align="right"><strong>61.21%</strong></td>
<td align="right">58.55%</td>
</tr>
<tr>
<td align="left"><img src="assets/google.svg" width="22" height="22" alt="Google" valign="middle" /></td>
<td align="left"><code>Gemma 4 31B</code></td>
<td align="center"><code>video</code></td>
<td align="right"><strong>66.83%</strong></td>
<td align="right"><strong>63.65%</strong></td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/qwen.svg" width="22" height="22" alt="Qwen" valign="middle" /></td>
<td align="left"><code>Qwen3 VL 32B</code></td>
<td align="center"><code>video</code></td>
<td align="right"><strong>62.80%</strong></td>
<td align="right"><strong>59.36%</strong></td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/openai.svg" width="22" height="22" alt="OpenAI" valign="middle" /></td>
<td align="left"><code>gpt-4.1-nano</code></td>
<td align="center"><code>text</code></td>
<td align="right"><strong>58.31%</strong></td>
<td align="right">—</td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/nvidia.svg" width="22" height="22" alt="NVIDIA" valign="middle" /></td>
<td align="left"><code>Nemotron Nano 12B 2 VL</code></td>
<td align="center"><code>video</code></td>
<td align="right"><strong>51.53%</strong></td>
<td align="right">—</td>
<td align="right">—</td>
</tr>
<tr>
<td align="left"><img src="assets/xiaomi.svg" width="22" height="22" alt="NVIDIA" valign="middle" /></td>
<td align="left"><code>Xiaomi MiMo V2.5</code></td>
<td align="center"><code>video</code></td>
<td align="right"><strong>57.09%</strong></td>
<td align="right"><strong>53.6%</strong></td>
<td align="right"><strong>49.1%</strong></td>

</tr>
</tbody>
</table>

*Percentages are rounded to two decimal places (raw: Gemini Flash `0.7381…`, Llama `0.6965…`, Qwen VL `0.6280…`).*

---

## Update: composable transforms

Each pipeline now supports a `self.transforms` list — ordered `Callable[[dict], dict]` functions applied concurrently to every input row before the model runs. Transforms receive the full row dict (video id, oracle timestamps, question, options, etc.) and return an updated version of it; config params are pre-bound via `functools.partial`.

**Example — `video_clipped`:** clips each video to a 5-minute window around the ground-truth oracle segment before sending it to the model.

```python
class VideoClippedPipeline(VideoPipeline):
    def __init__(self, configs):
        super().__init__(configs)
        self.transforms = [
            load_video,
            partial(clip_around_oracle, **configs.transform_configs.get("clip_around_oracle", {})),
            encode_video,
        ]
```

Transform parameters (e.g. `output_length`) are set in the experiment YAML under `transform_configs` and passed through `PipelineConfig`.

---

## Quick start

Requires **Python 3.12–3.13** (see `pyproject.toml`). Configure API keys in `.env`, then run a pipeline via `scripts/run_pipeline.py`:

```bash
python scripts/run_pipeline.py --pipeline language --model "google_genai:gemini-2.5-flash" --split val
python scripts/run_pipeline.py --pipeline video --model "your_provider:your-video-model" --split val
```

Use `--help` on the script for concurrency, split options, and other flags.

---

<div align="center">

<sub>Built for reproducible unimodal baselines and straightforward extension via the pipeline registry.</sub>

</div>


Best results per adapter, 10-30 epochs, 1e-5 learning rate:

- early fusion:
    - video-only: 0.25 
    - text-only: 0.1603
    - fused: 0.1795
- late interaction:
    - video-only: 0.2308
    - text-only: 0.2115
    - fused: 0.2308
- additive / mean:
    - fused: 0.2596
    - text-only: 0.2212
    - video-only: 0.1827

  ┌────────────┬────────┬─────────────┬───────────────────┬─────────────────────────┐
  │   Metric   │   LI   │ EarlyFusion │ Mean-fuse lr=1e-5 │ Mean-fuse lr=1e-4 (new) │
  ├────────────┼────────┼─────────────┼───────────────────┼─────────────────────────┤
  │ fused top1 │ 0.2308 │ 0.1795      │ 0.2179            │ 0.2596 (e3)             │
  ├────────────┼────────┼─────────────┼───────────────────┼─────────────────────────┤
  │ fused top3 │ 0.5641 │ 0.5288      │ 0.5321            │ 0.5769 (e2)             │
  ├────────────┼────────┼─────────────┼───────────────────┼─────────────────────────┤
  │ text top1  │ 0.2115 │ 0.1603      │ 0.2372            │ 0.2212                  │
  ├────────────┼────────┼─────────────┼───────────────────┼─────────────────────────┤
  │ text top3  │ 0.5385 │ 0.4135      │ 0.4679            │ 0.4936                  │
  ├────────────┼────────┼─────────────┼───────────────────┼─────────────────────────┤
  │ video top1 │ 0.2308 │ 0.2500      │ 0.1635            │ 0.1827                  │
  ├────────────┼────────┼─────────────┼───────────────────┼─────────────────────────┤
  │ video top3 │ 0.5481 │ 0.5064      │ 0.5417            │ 0.5673                  │
  └────────────┴────────┴─────────────┴───────────────────┴─────────────────────────┘