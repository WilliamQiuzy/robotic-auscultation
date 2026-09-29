"""Publish the concise Chinese report only after full10 and media gates exist.

Run after verification, before the package seal. Never overwrites a report and
never declares mirror synchronization, clinical validity or training readiness.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGETS = tuple(side+str(i) for i in range(1, 6) for side in ("R", "L"))
FILES = {
    "runner": "results/physics_selected_all10/runner_summary.json",
    "outcome": "results/physics_selected_all10/outcome.json",
    "quality": "results/quality_selected_all10/quality_report.json",
    "free": "results/quality_selected_all10/free_motion_report.json",
    "comparison": "results/compare_selected_all10/comparison.json",
    "body": "results/body_pose_selected_all10.json",
    "media": "media_verification_linux.json",
    "render": "video_selected/review_manifest.json",
    "compositor": "video_comparison/comparison_video_manifest.json",
    "search": "results/search_v1/search_summary.json",
    "pair": "results/pair_rank_v1.json",
    "selection": "results/screen_selection.json",
    "plans": "results/candidate_b_plans/selection.json",
    "screen_a": "results/physics_a_screen/runner_summary.json",
    "screen_b": "results/physics_b_screen/runner_summary.json",
}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): digest.update(block)
    return digest.hexdigest()


def read(path):
    def nonfinite(value): raise ValueError("nonfinite report input: " + value)
    return json.loads(Path(path).read_text(), parse_constant=nonfinite)


def _child(directory, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "unsafe artifact path")
    directory = Path(directory).resolve(); path = (directory/relative).resolve()
    require(directory in path.parents, "artifact escaped its source directory")
    return path


def _same_recorded_path(recorded, expected, project, label):
    """Allow host relocation, while retaining the exact project-relative identity."""
    require(isinstance(recorded, str) and recorded and ".." not in Path(recorded).parts,
            label+" has unsafe recorded path")
    expected_parts = Path(expected).resolve().relative_to(project).parts
    parts = Path(recorded).parts
    require(len(parts) >= len(expected_parts) and parts[-len(expected_parts):] == expected_parts,
            label+" references a different episode/artifact")


def verify_bound_episodes(data, root, *, provenance_validator=None):
    """Check both complete live artifact inventories and each report/render identity.

    The optional validator supports bounded fixtures; production always uses the
    unchanged curobo provenance adapter against each fixed episode directory.
    Returned paths are project-relative, including the sealed baseline package.
    """
    if provenance_validator is None:
        from sim.curobo_expert_v1.quality import validate_episode_provenance
        provenance_validator = validate_episode_provenance
    root = Path(root).resolve(); project = root.parents[1]
    episodes = {"baseline": project/"sim/curobo_expert_v1/results/physics_revised_all10_v1",
                "selected": root/"results/physics_selected_all10"}
    hashes = {}
    def pin(path, expected):
        path = Path(path).resolve(); relative = str(path.relative_to(project))
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in "0123456789abcdef" for c in expected), "invalid artifact digest: "+relative)
        if relative not in hashes:
            require(path.is_file() and sha(path) == expected, "artifact hash mismatch: "+relative)
            hashes[relative] = expected
        require(hashes[relative] == expected, "conflicting artifact bindings: "+relative)
        return expected
    compositor, media = data["compositor"], data["media"]["comparison"]
    panels = compositor.get("panel_sources", [])
    require(len(panels) == 2 and [p.get("role") for p in panels] == ["baseline", "alternative"],
            "comparison requires explicit ordered panel identities")
    require(media.get("panel_sources") == panels and len(media.get("source_renders", [])) == 2,
            "verified panel identities differ from compositor")
    renders = {}
    for index, side in enumerate(("baseline", "selected")):
        episode = episodes[side]; panel = panels[index]; verified = media["source_renders"][index]
        manifest_path = _child(project, panel["manifest"])
        require(manifest_path.name == "review_manifest.json", "unexpected source render manifest")
        if side == "selected":
            require(manifest_path == root/FILES["render"], "selected panel uses another render")
        render = read(manifest_path); renders[side] = render
        digest = pin(manifest_path, compositor["source_sha256"][panel["manifest"]])
        require(verified.get("status") == "passed" and verified.get("manifest_sha256") == digest
                and verified.get("schedule", {}).get("targets") == list(TARGETS),
                side+" media receipt does not bind the complete ten-point render")
        _same_recorded_path(verified.get("manifest"), manifest_path, project, side+" verified render")
        _same_recorded_path(verified.get("episode"), episode, project, side+" media episode")
        _same_recorded_path(render.get("episode"), episode, project, side+" render episode")
        require(render.get("schema") == "continuous_recorded_route_v1", "unexpected source render schema")
        video = _child(manifest_path.parent, render["video"])
        require(_child(project, panel["video"]) == video, side+" panel video path mismatch")
        digest = pin(video, render["video_sha256"])
        require(verified.get("video_sha256") == digest and compositor["source_sha256"].get(panel["video"]) == digest,
                side+" media video binding mismatch")
        for relative, expected in render["source_sha256"].items(): pin(_child(project, relative), expected)
        outcome = read(episode/"outcome.json")
        require(outcome.get("success") is True and outcome.get("requested_points") == list(TARGETS),
                side+" canonical episode is incomplete")
        require(render.get("point_intervals") == [{k: p[k] for k in
            ("target", "start_physics_row", "end_physics_row")} for p in outcome["points"]],
                side+" render point coverage differs from its episode")
    require(renders["selected"] == data["render"], "selected render changed during report binding")
    for side, episode in episodes.items():
        fresh = provenance_validator(episode)
        require(fresh.get("status") == "passed", side+" current episode provenance failed")
        expected = fresh.get("artifact_sha256")
        require(isinstance(expected, dict) and expected, side+" live artifact inventory missing")
        comparison = data["comparison"]["source_bindings"]["baseline" if side == "baseline" else "alternative"]
        body = data["body"]["episodes"][side]
        _same_recorded_path(comparison.get("episode"), episode, project, side+" comparison")
        _same_recorded_path(body.get("episode"), episode, project, side+" body review")
        records = [("comparison", comparison), ("comparison provenance", comparison["provenance"]),
                   ("body provenance", body["provenance"]), ("render provenance", renders[side]["episode_provenance"])]
        if side == "selected":
            records += [("quality", data["quality"]), ("free provenance", data["free"]["provenance"])]
            for label in ("quality", "free"):
                _same_recorded_path(data[label].get("episode"), episode, project, "selected "+label)
            for name in ("outcome.json", "expert_trace.jsonl", "expert_replay.npz"):
                require(data["runner"]["evidence_sha256"].get(name) == expected.get(name),
                        "selected runner artifact binding mismatch: "+name)
        for label, record in records:
            require(record.get("artifact_sha256") == expected,
                    side+" "+label+" differs from complete current artifact inventory")
        for relative, digest in expected.items(): pin(_child(episode, relative), digest)
        pin(episode/"runner_summary.json", body["runner_sha256"])
    return hashes


def gate_reports(data):
    """Pure publication gates; input byte/episode bindings are checked by build."""
    r, o, q, free, comparison, body, media = (data[k] for k in
        ("runner", "outcome", "quality", "free", "comparison", "body", "media"))
    require(r.get("accepted") is True and r.get("child_returncode") == 0, "full10 runner not accepted")
    require(o.get("success") is True and o.get("requested_points") == list(TARGETS)
            and [p.get("target") for p in o.get("points", [])] == list(TARGETS)
            and all(p.get("success") is True for p in o["points"]), "full10 outcome incomplete")
    require(q.get("hard_failures") == [] and q.get("provenance_status") == "passed", "full10 quality hard gate failed")
    require(free.get("status") == "passed" and free.get("failures") == [], "full10 free-motion audit failed")
    full = comparison.get("full10_comparison")
    require(isinstance(full, dict) and full.get("status") == "matched"
            and full.get("window_kind") == "full10_complete_route", "full10 comparison absent")
    upper = comparison.get("upper_pair_comparison", {})
    require(upper.get("status") == "matched" and upper.get("window_kind") == "upper_pair_through_R2_HOLD",
            "matched upper R1 to R2 HOLD window absent")
    require(comparison.get("unexpected_nonupper_plan_changes") == []
            and comparison.get("fixed_plan_comparison_status") == "matching_captured_nonupper_plans"
            and all(comparison.get("shared_plan_hashes_equal", {}).get(n) is True for n in TARGETS[2:]),
            "unchanged eight plans not confirmed")
    require(body.get("schema") == "upper_pair_recorded_body_pose_review_v1"
            and {"baseline", "selected"}.issubset(body.get("episodes", {}))
            and "selected" in body.get("alternative_minus_baseline", {}), "selected full10 body review absent")
    selected = body["episodes"]["selected"]
    require(selected.get("recorded_runner_accepted") is True and selected.get("recorded_outcome_success") is True,
            "selected body review is not from a completed accepted episode")
    for name in TARGETS:
        hold = comparison["contact_by_point"]["alternative"][name]["hold"]
        require(abs(hold["duration_s"]-3.) < 1e-7 and abs(hold["longest_continuous_s"]-3.) < 1e-7,
                "continuous 3 s HOLD missing: " + name)
    require(media.get("status") == "passed" and media.get("comparison", {}).get("status") == "passed",
            "Linux comparison media verification absent")
    require(data["search"].get("completed") is True and data["search"].get("source_inputs_unchanged") is True,
            "finite search incomplete")
    require(data["selection"].get("chosen") == "candidate_b_plans", "selected candidate changed")
    require(all(data[k].get("accepted") is True for k in ("screen_a", "screen_b")), "two physics screens not accepted")
    for point, expected in (("R1", 20.), ("L1", 10.)):
        identifier = data["plans"]["selected_plans"][point]["search_candidate_id"]
        candidates = [c for c in data["search"]["candidates"] if c["id"] == identifier]
        require(len(candidates) == 1 and abs(candidates[0]["relative_roll_deg"]-expected) < 1e-6,
                "selected audited tool-axis roll differs: " + point)


def metric(window):
    actual = window["actual"]
    return dict(duration_s=window["window"]["duration_s"], total_travel_rad=actual["total_travel_rad"],
        wrist_travel_rad=actual["wrist_joints_5_7_travel_rad"], joint4_travel_rad=actual["elbow_flexion_joint_4_travel_rad"],
        travel_per_joint_rad=actual["travel_rad"],
        raw_jerk_peak_rad_s3=max(actual["raw"]["jerk"]["max_abs_per_joint"]),
        filtered_jerk_peak_rad_s3=max(actual["filtered"]["jerk"]["max_abs_per_joint"]),
        commanded_raw_jerk_peak_rad_s3=max(window["commanded"]["raw"]["jerk"]["max_abs_per_joint"]),
        commanded_filtered_jerk_peak_rad_s3=max(window["commanded"]["filtered"]["jerk"]["max_abs_per_joint"]),
        hall_peak_N=window["force"]["hall_peak_N"], front_sum_peak_N=window["force"]["front_sum_peak_N"],
        elbow_max_z_m=window["posture_measured_q_fk"]["elbow_max_z_m"])


def summarize(data):
    comparison = data["comparison"]
    windows = {label: {side: metric(comparison[key][side]) for side in ("baseline", "alternative")}
               for label, key in (("upper_R1_through_R2_HOLD", "upper_pair_comparison"), ("full10", "full10_comparison"))}
    hold = {}
    for target in ("R1", "L1"):
        values = [data["body"]["episodes"][side]["HOLD"][target] for side in ("baseline", "selected")]
        hold[target] = dict(baseline_median_elbow_z_m=values[0]["recorded_body_landmarks"]["elbow"]["median_xyz_m"][2],
            selected_median_elbow_z_m=values[1]["recorded_body_landmarks"]["elbow"]["median_xyz_m"][2],
            delta_median_elbow_mm=data["body"]["alternative_minus_baseline"]["selected"][target]["delta_median_elbow_z_m"]*1000,
            selected_minimum_wrist_minus_elbow_z_m=values[1]["minimum_wrist_minus_elbow_z_m"])
    selected_ids = [data["plans"]["selected_plans"][n]["search_candidate_id"] for n in ("R1", "L1")]
    baseline_pair = next(p for p in data["pair"]["records"] if p["baseline"])
    selected_pair = next(p for p in data["pair"]["records"] if [p["r1"], p["l1"]] == selected_ids)
    require(baseline_pair["feasible"] and selected_pair["feasible"], "nominal selected pair not feasible")
    proxy = 100*(1-selected_pair["metrics"]["wrist_travel_rad"]/baseline_pair["metrics"]["wrist_travel_rad"])
    attempts = sum(len(p["endpoint_screen"]["attempts"]) for p in data["search"]["points"])
    require(attempts == 576 and len(data["search"]["candidates"]) == 12, "finite search coverage differs from report contract")
    drag = data["quality"]["contact"]["loaded_drag"]
    retraction = [{"target": event["from_target"], "tangential_distance_mm": event["tangential_distance_m"]*1000,
                   "draft_distance_limit_mm": event["distance_limit_m"]*1000,
                   "start_row": event["start_row"], "end_row": event["end_row"]}
                  for event in drag["events"] if event.get("category") == "retraction"
                  and event["from_target"] == event["to_target"] and event.get("exceeded") is True]
    return {"schema": "upper_pair_final_evidence_summary_v1", "selected_candidate": "B",
        "selected_tool_axis_roll_deg": {"R1": 20., "L1": 10.}, "full10_mechanical_run_accepted": True,
        "unchanged_other_plan_count": 8, "search": {"IK_attempts": attempts, "complete_geometric_candidates": 12,
        "physical_screen_candidates": 2, "bounded_search_only": True}, "windows": windows, "HOLD_body_pose": hold,
        "nominal_proxy_wrist_reduction_percent": proxy,
        "quality": {k: data["quality"].get(k) for k in ("hard_failures", "pending", "simulation_quality_decision",
                    "training_demonstration_readiness", "contact_site_calibration_status")},
        "free_motion_status": data["free"]["status"], "media_verification_status": data["media"]["status"],
        "loaded_retraction": {"status": drag["status"], "exceeded_same_point_events": retraction,
            "loaded_force_threshold_N": drag["loaded_force_N"],
            "point_change_event_count": drag["by_category"]["point_change"]["event_count"],
            "free_transfer_event_count": drag["by_category"]["free_transfer"]["event_count"],
            "scope": "Same-point unloading/retraction tangential TCP motion; not loaded inter-point transfer or tissue shear force."},
        "mirror_status": "Not asserted here; inspect post-seal mirror_receipts on each host.",
        "claims": {"final_expert_accepted": False, "training_ready": False, "clinical_validity": False,
                   "acoustic_validity": False, "spatial_elbow_flip_classified": False},
        "scope": "Selected engineering comparison completed; necessary smooth joint motion remains allowed. "
                 "Nominal proxy and actual upper/full10 windows are distinct. No impossibility proof."}


def markdown(summary):
    upper, full = summary["windows"]["upper_R1_through_R2_HOLD"], summary["windows"]["full10"]
    reduction = 100*(1-upper["alternative"]["wrist_travel_rad"]/upper["baseline"]["wrist_travel_rad"])
    q = summary["quality"]
    lines = ["# 上背两点姿态比较", "",
        "推荐候选 B 作为下一轮人工审阅的工程方案：完整十点已完成接近、接触、连续 HOLD、撤离、点间转移及最终回位，机械硬检查与自由运动检查通过。患者、安装、目标、接触法向及已批准相机固定；其余八点计划字节不变。R1/L1 的 20°/10°是工具绕自身轴旋转，不是相机转角。", "",
        "有限搜索包含 576 次 IK、12 条完整几何候选及两组实际物理筛查。下表分别比较上部共同窗口与完整十点，避免把不同收尾混为总量。必要的平滑腕肘运动仍然允许；各关节存在取舍，不能宣称全部少转。", ""]
    labels = (("时长（s）", "duration_s"), ("七关节总行程（rad）", "total_travel_rad"),
        ("腕组 q5–q7（rad）", "wrist_travel_rad"), ("q4 屈曲行程（rad）", "joint4_travel_rad"),
        ("区间肘最高（m，实测 q 的 FK）", "elbow_max_z_m"),
        ("实测原始 jerk 峰（rad/s³）", "raw_jerk_peak_rad_s3"),
        ("实测约 0.1 s 滤波 jerk 峰（rad/s³）", "filtered_jerk_peak_rad_s3"),
        ("命令原始 jerk 峰（rad/s³）", "commanded_raw_jerk_peak_rad_s3"),
        ("命令约 0.1 s 滤波 jerk 峰（rad/s³）", "commanded_filtered_jerk_peak_rad_s3"),
        ("Hall / 前端接触力峰（N）", None))
    for title, window in (("R1 开始 → R2 HOLD 结束", upper), ("完整十点，包含最终回位", full)):
        lines += ["**"+title+"**", "", "| 实测指标 | 原路线 B | 候选 B |", "|---|---:|---:|"]
        for label, key in labels:
            cells = [f'{window[side][key]:.5g}' if key else
                     f'{window[side]["hall_peak_N"]:.4f} / {window[side]["front_sum_peak_N"]:.4f}'
                     for side in ("baseline", "alternative")]
            lines.append("| "+label+" | "+" | ".join(cells)+" |")
        lines += [""]
    lines += ["| 关节行程（rad） | 上部原路线 | 上部候选 | 上部变化 | 全十点变化 |",
              "|---|---:|---:|---:|---:|"]
    for j in range(7):
        b, a = (upper[s]["travel_per_joint_rad"][j] for s in ("baseline", "alternative"))
        delta = full["alternative"]["travel_per_joint_rad"][j]-full["baseline"]["travel_per_joint_rad"][j]
        lines.append(f"| q{j+1} | {b:.5f} | {a:.5f} | {a-b:+.5f} | {delta:+.5f} |")
    lines += ["", "| HOLD 刚体肘高 | 原路线中位高度（m） | 候选中位高度（m） | 变化（mm） |",
              "|---|---:|---:|---:|"]
    for target, h in summary["HOLD_body_pose"].items():
        lines.append(f'| {target} | {h["baseline_median_elbow_z_m"]:.5f} | {h["selected_median_elbow_z_m"]:.5f} | {h["delta_median_elbow_mm"]:+.2f} |')
    joint7_degrees = math.degrees(upper["alternative"]["travel_per_joint_rad"][6]-upper["baseline"]["travel_per_joint_rad"][6])
    command_same = all(math.isclose(full["baseline"][key], full["alternative"][key], rel_tol=0, abs_tol=1e-8)
                       for key in ("commanded_raw_jerk_peak_rad_s3", "commanded_filtered_jerk_peak_rad_s3"))
    command_note = "完整十点命令 jerk 的原始、滤波峰均与基线相同；" if command_same else "完整十点命令 jerk 峰另见表；"
    events = summary["loaded_retraction"]["exceeded_same_point_events"]
    if events:
        targets = "/".join(event["target"] for event in events)
        distances = "/".join(f'{event["tangential_distance_mm"]:.3f}' for event in events)
        limits = sorted({event["draft_distance_limit_mm"] for event in events})
        limit_text = "/".join(f"{value:g}" for value in limits)
        unloading_note = f"{targets} 卸载期切向累计移动分别约 {distances} mm，超过草案 {limit_text} mm。"
    else:
        unloading_note = "同点卸载期未记录草案切向移动超限。"
    if summary["loaded_retraction"]["point_change_event_count"] == summary["loaded_retraction"]["free_transfer_event_count"] == 0:
        unloading_note += "这是同点卸载期移动，审计未记录带力跨点转移。"
    else:
        unloading_note += "跨点及自由转移事件另列于 JSON，不能与卸载期混同。"
    lines += ["", f"上部 q7 行程变化 {joint7_degrees:+.2f}°；HOLD 肘高中位数与区间最高肘位是不同指标。十点均连续 HOLD 3 s，力峰如表。原始及滤波 jerk 的既有偏好问题仍需复核，不能用滤波值替代原始峰值。质量结论为 `"+str(q["simulation_quality_decision"])+"`，偏好、装配或接触部位待决项保留于审计。q4 屈曲不等于空间翻肘；采样 SO(3) 也不是翻肘分类器。", "",
        command_note+"实测原始 jerk 峰下降不代表平滑性问题已解决。"+unloading_note, "",
        f'名义腕部代理约 {summary["nominal_proxy_wrist_reduction_percent"]:.1f}% 与实测上部约 {reduction:.1f}% 使用不同路径窗口：前者截止 R2 预接近位，后者包含实际接触与释放后的重规划，不能解读为“收益损失”。有限搜索不证明其他姿态不可能。', "",
        "物理执行与渲染均在 Linux；这是工程比较，不代表最终专家、临床、声学或训练就绪验收。核心文件的 Mac/Linux 一致性须在封存后分别查看镜像回执，本报告不提前声称同步完成。", "",
        "下一步先固定本轮较低肘位候选，专门处理接触进入及卸载撤离时的关节变化连续性，再进行最终专家与训练示范验收。本轮保持控制器不变，不直接启动 IL。", "",
        "[原速并排视频](video_comparison/upper_pair_comparison_1x.mp4) · [完整十点视频](video_selected/alternative_full_1x.mp4) · [实测比较](results/compare_selected_all10/comparison.md) · [刚体姿态](results/body_pose_selected_all10.json) · [质量审计](results/quality_selected_all10/quality_report.json) · [自由运动](results/quality_selected_all10/free_motion_report.json) · [媒体验证](media_verification_linux.json) · [输入哈希与摘要](results/final_summary.json)", ""]
    return "\n".join(lines)


def build(root=ROOT, output=None):
    root = Path(root).resolve()
    require(root == ROOT.resolve(), "report root must be this new comparison package")
    destination = root/"results/final_summary.json" if output is None else Path(output).resolve()
    require(destination == root/"results/final_summary.json", "summary output must use the fixed new path")
    readme = root/"README.md"
    require(not destination.exists() and not readme.exists(), "final report already exists; no overwrite")
    generator_before = sha(__file__)
    paths = {name: root/relative for name, relative in FILES.items()}
    before = {relative: sha(root/relative) for relative in FILES.values()}
    data = {name: read(path) for name, path in paths.items()}
    gate_reports(data)
    episode_before = verify_bound_episodes(data, root)
    require(data["body"]["episodes"]["selected"]["runner_sha256"] == before[FILES["runner"]], "body runner binding mismatch")
    require(data["render"]["point_intervals"] == [{k: p[k] for k in
            ("target", "start_physics_row", "end_physics_row")} for p in data["outcome"]["points"]],
            "render does not cover selected full10")
    media_comparison = data["media"]["comparison"]
    require(media_comparison["manifest_sha256"] == before[FILES["compositor"]], "media comparison manifest changed")
    selected_media = [r for r in media_comparison["source_renders"] if r["manifest_sha256"] == before[FILES["render"]]]
    require(len(selected_media) == 1 and selected_media[0]["schedule"]["targets"] == list(TARGETS),
            "media verification lacks selected full10 render")
    for relative, digest in (("video_selected/alternative_full_1x.mp4", data["render"]["video_sha256"]),
                             ("video_comparison/upper_pair_comparison_1x.mp4", data["compositor"]["sha256"])):
        require(sha(root/relative) == digest, "video changed since verification: "+relative)
        before[relative] = digest
    require(media_comparison["video_sha256"] == data["compositor"]["sha256"], "comparison video receipt changed")
    summary = summarize(data); text = markdown(summary)
    after = {relative: sha(root/relative) for relative in before}
    episode_after = {relative: sha(root.parents[1]/relative) for relative in episode_before}
    generator_after = sha(__file__)
    require(before == after and episode_before == episode_after and generator_before == generator_after,
            "report input/episode/generator changed")
    summary.update(input_sha256_before=before, input_sha256_after=after,
                   episode_and_render_sha256_before=episode_before, episode_and_render_sha256_after=episode_after,
                   episode_binding_path_base="project root; includes canonical baseline and selected full10",
                   generator_sha256_before=generator_before, generator_sha256_after=generator_after,
                   readme_sha256=hashlib.sha256(text.encode()).hexdigest())
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    with readme.open("x") as stream: stream.write(text)
    with destination.open("x") as stream: stream.write(encoded)
    return {"readme": str(readme), "summary": str(destination), "status": "written"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="must be ROOT/results/final_summary.json")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.root, args.output)), flush=True)


if __name__ == "__main__": main()
