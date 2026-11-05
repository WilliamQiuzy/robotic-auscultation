# Astra + local auscultation skills

**Status:** runnable API-request prototype with local validation tests. No live API call, Isaac execution, or Astra auscultation success rate is claimed by this release.

We propose **GPT-6 Astra as a skill supervisor**, with the existing analytic planner or ACT approach policy below it. The robot's contact control remains local. This is an additional policy track; it is not a replacement for training data or physical evaluation.

```
RGB + depth-derived target points + joints + Hall voltage + local phase
                              ↓
                  Astra: request one bounded skill
        approach_target / continue_local_control / retract / abort
                              ↓
          native local executor → IK / motion / force checks
                              ↓
                actual simulator measurements → evaluation
```

The input is multimodal when an RGB image is supplied: an image plus structured depth-derived geometry, proprioception, Hall voltage and phase. This prototype **does not encode a raw depth map** and does not establish that Astra makes useful use of each sensor. That needs controlled perturbation and actual task tests.

## Try the interface without an API call

```sh
python -m demo.astra_policy \
  --observation examples/astra_observation.json \
  --output /tmp/astra-request.json
```

The example is explicitly synthetic interface data, not measured robot state. The default path is a dry run: it constructs a Responses API request but does not call a model or execute motion.

For an actual model proposal, use an API key in your environment and a current **simulation** observation:

```sh
pip install openai
export OPENAI_API_KEY=...  # local environment only
python -m demo.astra_policy \
  --observation /data/current-sim-observation.json \
  --rgb /data/current-robot-camera.png \
  --output /data/astra-proposal.json --live
```

The live path returns a validated proposal, API usage and wall time. It **does not dispatch** the skill to Isaac. To execute it, connect a simulator-side skill bridge that binds the proposal to the current observation ID and invokes the existing guarded executor. Do not turn a requested skill into an execution or success label.

Local validation rejects invented targets, stale observation IDs, force/gain override fields and a new approach during HOLD. The model has no tool for changing force, duration requirements, collision checks or success labels. During contact it can request continuation, retreat or abort; the local controller owns the high-frequency force loop.

## Why this architecture

[NVIDIA's Astranomy study](https://research.nvidia.com/labs/srl/projects/astranomy/) evaluates direct, agentic and orchestrated Astra policies on manipulation. It demonstrates that the integration role matters, and reports substantial inference latency. That is motivation to evaluate a supervisory layer in auscultation; it is **not evidence of auscultation performance**. Our choice to keep force control local is an engineering inference from these latency requirements and our contact task.

The [official GPT-6 Astra documentation](https://developers.openai.com/api/docs/models/gpt-6-astra) lists image input, function calling and structured outputs. [`demo/astra_policy.py`](../demo/astra_policy.py) uses the [Responses API function-calling format](https://developers.openai.com/api/docs/guides/function-calling) with a strict schema and one requested skill at a time. Model availability and account access must be checked when making a live request.

After the supervisory baseline, a **direct end-effector chunk** track would be useful, with the same IK, collision, joint, force and time gates. It is not implemented by this skill-request prototype and should have a distinct benchmark name. Both tracks must be tested against the same frozen scenes, with latency/cost reported and sensor-use ablations, before claiming an improvement over ACT or the analytic expert.
